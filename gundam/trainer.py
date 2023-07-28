import os, inspect, time
import numpy as np
import math, random
import fire

from transformers import Trainer
from datasets import load_dataset
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from contextlib import nullcontext


path = __file__
file_path = os.path.dirname(path)
import sys, os
sys.path.append(os.path.dirname(file_path))


from gundam.config import GundamConfig
from gundam.modeling import GundamPreTrainModel
from gundam.tokenizer import GundamTokenizer

torch.cuda.empty_cache()

seed = 1287
s_dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[s_dtype]
out_dir = 'out'
os.makedirs(out_dir, exist_ok=True)

wandb_log = False # disabled by default
wandb_project = 'gundam'
wandb_run_name = 'run' + str(time.time())

g_config = GundamConfig()
tokenizer = GundamTokenizer(path_to_model = os.path.join(file_path, 'tokenizer.model'))

ctx = nullcontext() if g_config.device == 'cpu' else torch.amp.autocast(device_type=g_config.device, dtype=dtype)

datasets = ['20230601.en', '20230601.zh', '20230601.ja']
train_data = [os.path.join(file_path, 'train.' + name + '.bin') for name in datasets]
train_data = [np.memmap(path, dtype=np.uint32, mode='r') for path in train_data]

val_data = [os.path.join(file_path, 'val.' + name + '.bin') for name in datasets]
val_data = [np.memmap(path, dtype=np.uint32, mode='r') for path in val_data]
n_data = len(datasets)


class DataGenerator():

    idx = {}

    def __init__(self) -> None:
        self.idx.update(
            {data : 0 for data in train_data}
        )
        self.idx.update(
            {data : 0 for data in val_data}
        )

    def create_batch(self, split, sel = -1, device = 'cuda'):
        if sel < 0:
            sel = random.randint(0, n_data - 1)
        
        data = train_data[sel] if split == 'train' else val_data[sel]
        start_pos = self.idx[data]
        if start_pos + g_config.max_length * g_config.batch_size  > len(data):
            start_pos = 0

        self.idx[data] = self.idx[data] + g_config.max_length

def create_batch(split, sel = -1, device = 'cuda'):
    if sel < 0:
        sel = random.randint(0, n_data -1)
    
    data = train_data[sel] if split == 'train' else val_data[sel]
    ix = torch.randint(len(data) - g_config.max_seq_length, (g_config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+g_config.max_seq_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+g_config.max_seq_length]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

#trainer = Trainer(model)
def compute_loss(result, y):
    estimate_y = result.last_hidden_state
    loss = cross_entropy(estimate_y.view(-1, estimate_y.size(-1)), 
                         y.view(-1), 
                         ignore_index=-1)
    return loss
    
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    eval_iter = 100
    losses = torch.zeros(eval_iter)
    for split in ['train', 'val']:
        for k in range(eval_iter):
            X, Y = create_batch(split=split)
            with ctx:
                result = model(X)
                loss = compute_loss(result, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
        # 1) linear warmup for warmup_iters steps
    if it < g_config.warmup_iters:
        return g_config.learning_rate * it / g_config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > g_config.learning_rate_decay_iters:
        return g_config.min_learning_rate
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - g_config.warmup_iters) / (g_config.learning_rate_decay_iters - g_config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return g_config.min_learning_rate + coeff * (g_config.learning_rate - g_config.min_learning_rate)

def save_checkpoint(model, optimizer, config, iter_num, loss):
    checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': loss,
                    'config': config,
                }
    print(f"saving checkpoint to {out_dir}")
    
    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

def train(seed = seed, 
          init_from = 'scratch', ## scratch, resume, 'gpt2' etc 
          max_iter = g_config.max_iters):
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16)) ##https://github.com/karpathy/nanoGPT/issues/167

    config = GundamConfig()
    model = GundamPreTrainModel(config=config, device=config.device, dtype=dtype)

    if init_from == 'scratch':
        iter_num = 0
        best_val_loss = 10000.0

    elif init_from == 'resume':
        ckpath = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpath, map_location=config.device)
        state_dict = checkpoint['model']

        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        #iter_num = 0
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    else:
        pass
    
    model.to(config.device, dtype = dtype)

    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)


    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (config.beta1, config.beta2), config.device)

    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    checkpoint = None

    if config.compile:
        unoptimized_model = model
        model = torch.compile(model)

    X, Y = create_batch('train')
    t0 = time.time()
    simulate_batch_size = 40

    try: 
        while iter_num < max_iter:
        
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if iter_num % 1000 == 0: #make checkpoint 
                ## eval error
                losses = estimate_loss(model)
                print(f"step {iter_num}: loss {losses['train']:.4f}")

                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                    })

                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    print(f"current val error {best_val_loss:.4f}..")
                    save_checkpoint(model, optimizer, config, iter_num, best_val_loss)

            for micro_step in range(simulate_batch_size):
                with ctx:
                    result = model(X)
                    loss = compute_loss(result, Y)
                    loss = loss / simulate_batch_size
                    
                scaler.scale(loss).backward()

                X, Y = create_batch('train')

            if config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if iter_num % 10 == 0:
                lossf = loss.item() * simulate_batch_size
                print(f"iter {iter_num}: loss {lossf:.4f}. time {dt*1000:.2f}ms. learning rate {lr}")
            
            iter_num += 1

    except KeyboardInterrupt:
        ## close it smoothly
        print("catch KeybordInterrupt")
        
    except Exception as e:
        raise(e)
        
    finally:
        losses = estimate_loss(model)
        print(f"step {iter_num}: loss {losses['train']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            print(f"current val error {best_val_loss:.4f}..")
            save_checkpoint(model, optimizer, config, iter_num, best_val_loss)

if __name__ == '__main__':
    fire.Fire(train)
