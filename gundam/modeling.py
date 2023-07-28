import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import skip_init
import inspect

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from gundam.config import GundamConfig

from .config import GundamConfig

class Embedding(nn.Module):

    def __init__(self, config: GundamConfig, device = None, dtype = None):
        super(Embedding, self).__init__()
        self.hidden_size = config.hidden_size
        self.word_embeddings = nn.Embedding(
            num_embeddings = config.vocab_size,
            embedding_dim = self.hidden_size,
            device=device,
            dtype=dtype
        )

    def forward(self, input):
        # Embeddings.
        embeddings = self.word_embeddings(input)
        #embeddings = embeddings.transpose(0, 1).contiguous()
        return embeddings


class PositionEmbedding(nn.Module):
    def __init__(self, config:GundamConfig, device = None,  dtype = None) -> None:
        super(PositionEmbedding, self).__init__()
        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_seq_length

        pe = torch.zeros(self.max_seq_len, self.hidden_size)

        for pos in range(self.max_seq_len):
            for i in range(0, self.hidden_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.hidden_size)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.hidden_size)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (b, s, h)
        x = x * math.sqrt(self.hidden_size) 
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False) # (b, s, h) -> (b, s, h)
        return x
    
class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(shape, device=device, dtype=dtype))
        self.eps = eps
        self.bias = torch.nn.Parameter(torch.empty(shape, device=device, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor): # b, s, h
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)
    
class SelfAttention(nn.Module):
    def __init__(self, config: GundamConfig, device = None,  dtype = None):
        super(SelfAttention, self).__init__()
        
        self.hidden_size = config.hidden_size
        self.n_attn_heads = config.n_attn_heads
        self.head_size  = self.hidden_size // self.n_attn_heads
        self.max_seq_length = config.max_seq_length
        self.dropout_p = config.attn_dropout
        self.dropout = nn.Dropout(config.attn_dropout)
        
        self.qkv_attn = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias =config.add_qkv_bias, device=device)

        self.proj = nn.Linear(self.hidden_size, self.hidden_size, device = device)

        self.torch_major_version = int(torch.__version__.split('.')[0])

        if self.torch_major_version < 2:
            self.register_buffer("bias", torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
                                        .view(1, 1, config.max_seq_length, config.max_seq_length))
        
    def forward(self, x:torch.Tensor, mask=None, cache = None, use_cache = False):

        batch_size, seq_length, hidden_size = x.shape
        assert hidden_size == self.hidden_size, "The input lenght is not equal to the configuration"

        def split_heads(x):
            q, k, v = x.split(self.hidden_size, dim = 2)
            k = k.view(batch_size, seq_length, self.n_attn_heads, self.head_size).transpose(1, 2)
            q = q.view(batch_size, seq_length, self.n_attn_heads, self.head_size).transpose(1, 2)
            v = v.view(batch_size, seq_length, self.n_attn_heads, self.head_size).transpose(1, 2)
            return q, k, v

        attn_output = self.qkv_attn(x)
        
        q, k, v = split_heads(attn_output)

        if use_cache:
            if cache is None:
                cache = (k, v)
            else:
                cache_k, cache_v = cache
                cache = (torch.cat((cache_k, k), dim = 0), 
                         torch.cat((cache_v, v), dim = 0))
        else:
            cache = None

        if self.torch_major_version >= 2:
            if mask is None:
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                                 attn_mask = mask, 
                                                                 is_causal = True, 
                                                                 dropout_p = self.dropout_p if self.training else 0)
            else:
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                    attn_mask=mask,
                                                                    is_causal=False,
                                                                    dropout_p = self.dropout_p if self.training else 0)
        else:
            # b, s, h
            def scaled_dot_product_attention(Q, K, V, mask=None):
                # Q, K, V : b, nh, s, hd
                attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.n_head_d)
                attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
                if mask is not None:
                    attn = attn.masked_fill(mask == 0, -1e9)
                else:
                    attn = attn.masked_fill(self.bias[:, :, :seq_length, :seq_length] == 0, float('-inf'))
                # context: b, nh, s, s
                attn = torch.nn.functional.softmax(attn, dim=-1)
                # context: b, nh, s, s
                attn = self.dropout(attn)
                # context: b, nh, s, hd
                attn = torch.matmul(attn, V)
                return attn

            x = scaled_dot_product_attention(q, k, v, mask=mask)
        
        #B, _, S, _ = output.shape
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

        x = self.proj(x)

        return x, cache

        
class MLP(nn.Module):

    def __init__(self, config : GundamConfig, device = None,  dtype = None) -> None:
        super(MLP, self).__init__()
        self.size = config.hidden_size
        self.ffn_size = config.ffn_hidden_size
        ## by using the swiglu * 2
        self.feed_forward_1 = nn.Linear(self.size, self.ffn_size * 2, bias=config.add_mlp_bias, device = device, dtype = dtype)
        self.feed_forward_2 = nn.Linear(self.ffn_size, self.size, bias=config.add_mlp_bias, device = device, dtype = dtype)
        self.dropout = nn.Dropout(config.mlp_dropout)

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]
        
        self.relu = swiglu

    def forward(self, x):
        x = self.feed_forward_1(x)
        x = self.relu(x)
        x = self.feed_forward_2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config : GundamConfig, layer_number, device = None, dtype = None) -> None:
        super(TransformerBlock, self).__init__()
        self.layer_number = layer_number
        self.hidden_size = config.hidden_size
        self.residual_conn = config.layer_residual_conn
        self.atten_layer = SelfAttention(config, device = device, dtype = dtype)
        self.feed_forward = MLP(config, device = device, dtype = dtype)
        LayerNorm = RMSNorm if config.rmsnorm else nn.LayerNorm ##if config.RMSNorm nn.LayerNorm(self.hidden_size, device = device, dtype = dtype)
        self.input_layer_norm = LayerNorm(self.hidden_size, config.layer_norm_eps, bias=config.add_norm_bias, device=device, dtype=dtype)

        self.post_attn_norm  = LayerNorm(self.hidden_size, config.layer_norm_eps, bias=config.add_norm_bias, device = device, dtype = dtype)
        
        self.dropout_1 = nn.Dropout(config.attn_dropout)
        self.dropout_2 = nn.Dropout(config.mlp_dropout)

    def forward(self, x, mask = None, cache = None, use_cache = False):
        
        inp = x 
        
        input_norm_output = self.input_layer_norm(x)
        attn_output, cache = self.atten_layer(input_norm_output, mask, cache, use_cache)

        first_dropout_output = self.dropout_1(attn_output)

        if self.residual_conn:
            output1 = inp + first_dropout_output
        else:
            output1 = input_norm_output + first_dropout_output

        
        output_norm_output = self.post_attn_norm(output1)
        mlp_output = self.feed_forward(output_norm_output)
        sencond_dropout_output = self.dropout_2(mlp_output)

        if self.residual_conn:
            output2 = output1 + sencond_dropout_output
        else:
            output2 = output_norm_output + sencond_dropout_output

        return output2, attn_output, cache

class GundamTransformer(nn.Module):
    def __init__(self, config: GundamConfig, device = None, dtype = None) -> None:
        super(GundamTransformer, self).__init__()
        self.n_layer = config.n_layer
        self.residual_conn = config.transformer_residual_conn
        self.post_layer_norm = config.transformer_post_layer_norm
        self.use_cache = config.use_cache
        
        self.blocks = torch.nn.ModuleList([TransformerBlock(config, i, device, dtype) for i in range(self.n_layer)])
        
        if self.post_layer_norm:
            layer_norm_func = RMSNorm if config.rmsnorm else nn.LayerNorm
            self.layer_norm = layer_norm_func(config.hidden_size, config.layer_norm_eps, device=device, dtype=dtype)
    
    def forward(self, x, mask = None, kv_caches = None, ):
        all_states = ()
        all_self_attn = ()
        if self.use_cache and kv_caches is None:
            kv_caches = [None for _ in range(self.n_layer)]

        for index in range(self.n_layer):
            all_states = all_states + (x,)
            block = self.blocks[index]
            cache = kv_caches[index]
            x, attn, cache = block(x, mask, cache, self.use_cache)
            kv_caches[index] = cache
            all_self_attn = all_self_attn + (attn,)

        all_states = all_states + (x,)

        if self.post_layer_norm:
            x = self.layer_norm(x)

        return x, all_states, all_self_attn, kv_caches
    

class GundamPreTrainModel(PreTrainedModel):
    def __init__(self, config: GundamConfig, device=None, dtype = None, init = None):
        super(GundamPreTrainModel, self).__init__(config)
        self.embedding = Embedding(config=config, device=device, dtype=dtype)
        self.pos_embedding = PositionEmbedding(config=config, device=device, dtype=dtype)
        self.n_layer = config.n_layer
        self.transformers = GundamTransformer(config, device=device, dtype=dtype)
        self.dropout = nn.Dropout(config.global_dropout)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
        self.use_cache = config.use_cache

        self.apply(self._init_weights)

        print(f"number of parameters {self.num_parameters / 1e6 : .3f}M")
    
    @property
    def num_parameters(self):
        if '_num_params' not in dir(self):
            self._num_params = sum([p.numel() for p in self.parameters()])
        return self._num_params       

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, mean = 0.0, std = 0.0002)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas=betas, fused=use_fused)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def forward(self, ids, attn_masks = None):
        b, s = ids.shape
        
        embeds = self.embedding(ids) # (b, s, v) -> (b, s, h)

        embeds = self.pos_embedding(embeds) 

        x = self.dropout(embeds)

        output, states, self_attn, kv_caches = self.transformers(x, attn_masks)

        y = self.output_layer(output)
        
        return BaseModelOutputWithPast(
            last_hidden_state=y,
            hidden_states= states,
            attentions = self_attn,
            past_key_values=kv_caches
        )

    def gen_prompt(self, batch_size, device, dtype = torch.float16):
        pass

    # def gen_masks(self, input_ids, padding = None):
    #     batch_size, seq_length = input_ids.shape
    #     attn_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
    #     attn_mask.tril_()
    #     attn_mask = (attn_mask < 0.5).bool()
    #     attn_mask.unsqueeze_(1)
    #     return attn_mask
    
    # def estimate_mfu(self, fwdbwd_per_iter, dt):
    #     """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
    #     # first estimate the number of flops we do per iteration.
    #     # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    #     N = self.num_parameters()
    #     cfg: GundamConfig = self.config
    #     L, H, Q, T = cfg.n_layer, cfg.n_attn_heads, cfg.hidden_size//cfg.n_attn_heads, cfg.seq_length
    #     flops_per_token = 6*N + 12*L*H*Q*T
    #     flops_per_fwdbwd = flops_per_token * T
    #     flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    #     # express our flops throughput as ratio of A100 bfloat16 peak flops
    #     flops_achieved = flops_per_iter * (1.0/dt) # per second
    #     flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
    #     mfu = flops_achieved / flops_promised
    #     return mfu



    
class GundamChat(GundamPreTrainModel):
    def __init__(self, config: GundamConfig, device=None, dtype=None, init=None):
        super().__init__(config, device, dtype, init)


    def build_inputs(self, request, tokenizer, history = None):
        if history is None:
            history = []
        
        prompt = ""
        for i, (old_request, response) in enumerate(history):
            prompt += f"[Round {i}]\n\n Ask: {old_request}\n\n Answer: {response}\n\n]"

        prompt += f"[Round {len(history) + 1}\n\n Ask: {request}]"
        prompt = tokenizer.encode(prompt, )
