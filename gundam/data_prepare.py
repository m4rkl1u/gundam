import os
from tqdm import tqdm
import numpy as np
import time
from datasets import load_dataset
import fire

from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from transformers import PreTrainedTokenizer

path = __file__
file_path = os.path.dirname(path)
import sys, os
sys.path.append(os.path.dirname(file_path))

from gundam.tokenizer import SPTokenizer

num_proc = 12
path = 'graelo/wikipedia'
name = '20230601.zh'
test_size = 0.001
seed = int(time.time()) % 10000

def main(path = path, name = name, test_size = test_size, seed = int(time.time()), path_to_model = 'tokenizer.model'):

    dataset = load_dataset(path, name, num_proc=num_proc)

    if test_size > 0:
        split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test') # rename the test split to val
    else:
        split_dataset = dataset

    tok = SPTokenizer(path_to_model)

    def process(example):
        ids = tok.encode(example['text'], eos = True, bos=True) # encode_ordinary ignores any special tokens
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.{name}.bin')
        dtype = np.uint32 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
    
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

if __name__ == '__main__':
    fire.Fire(main)