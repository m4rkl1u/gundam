from transformers import PretrainedConfig
from dataclasses import dataclass

@dataclass
class GundamConfig(PretrainedConfig):
    device = 'cuda'
    
    ## tokenizer
    ## tiktoken 50257
    ## sentencepiece homemade 65024 ja + zh + en

    vocab_size = 65024 ## see the data_prepare.py 65024

    ## position embeding
    #embeding_method = 'embedding' ## 

    ## attention
    n_layer = 5
    n_attn_heads = 4
    head_dim = 128
    hidden_size = int(head_dim * n_attn_heads)

    seq_length = 1024 ## max_seq_lenght, block size
    max_seq_length = 1024
    attn_dropout = 0.001
    add_qkv_bias = False

    kv_channels = 8

    ## mlp
    ffn_hidden_size = 8096
    add_mlp_bias = False
    mlp_dropout = 0.001


    ## layer norm & rms norm
    layer_norm_eps = 1e-5
    rmsnorm = False
    add_norm_bias = True
    
    embedding_residual_conn = True
    
    layer_residual_conn = True

    transformer_residual_conn = False
    transformer_post_layer_norm = True ## transformer

    ## train
    global_dropout = 0.001
    batch_size = 10
    max_iters = 10
    learning_rate = 0.01
    grad_clip = 1.0
    compile = True

    warmup_iters = 1000 
    learning_rate_decay_iters = 100000
    min_learning_rate = 1e-5

    ## adamw
    weight_decay = 1e-1
    beta1= 0.9
    beta2 = 0.95

    use_cache = True
    
    quantization_bit = 16
