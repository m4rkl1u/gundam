from typing import List, Optional, Union, Dict, Any
import torch
import time
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset
import fire

class SPTokenizer(object):
    
    def __init__(self, path_to_model) -> None:
        self.path_to_model = path_to_model
        self.sp_model = SentencePieceProcessor(self.path_to_model)
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        self.index_special_tokens = {
            
        }

    def tokenize(self, s: str):
        return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.index_special_tokens or index in [self.eos_id, self.bos_id, self.pad_id] or index < 0:
            return ""
        return self.sp_model.IdToPiece(index)

class GundamTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path_to_model = kwargs.pop('path_to_model', None)
        if path_to_model is None:
            raise ValueError("Need to pass parameter --path_to_model")
        self.tokenizer_impl = SPTokenizer(path_to_model)
        self.n_words = self.tokenizer_impl.n_words
        special_tokens = ['<mask>', '<pmask>', '<smask>', '<bop>', '<eop>', '<pad>']
    
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1

    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]

    @property
    def pad_token(self) -> str:
        return "<unk>"

    @property
    def pad_token_id(self):
        return self.get_command("<pad>")

    @property
    def eos_token(self) -> str:
        return "</s>"

    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    @property
    def vocab_size(self):
        return self.n_words
    
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.tokenizer_impl.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.tokenizer_impl.decode_tokens(tokens)

    def _tokenize(self, text, **kwargs):
        return self.tokenizer_impl.tokenize(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.tokenizer_impl.convert_token_to_id(token)

    def get_vocab(self):
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def get_prefix_tokens(self):
        prefix_tokens = [self.get_command("<pmask>"), self.get_command("sop")]
        return prefix_tokens

    def build_prompt(self, query, history=None):
        if history is None:
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\nAsk: {}\n\nAnswer: {}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\nAsk: {}\n\nAnswer: ".format(len(history) + 1, query)
        return prompt



from datasets.arrow_dataset import Dataset
from datasets import concatenate_datasets, load_dataset
import os
from tqdm import tqdm

class MakeTokenizerModel(object):

    def __init__(self) -> None:
        pass

    def input_file(self, dataset_names = [('graelo/wikipedia', '20230601.ja'), 
                                     ('graelo/wikipedia', '20230601.zh'), 
                                     ('graelo/wikipedia', '20230601.en')], max_size = 1000, seed = time.time()):
        
        datasets = [load_dataset(n1, n2)['train'] for n1, n2 in dataset_names]

        dataset = concatenate_datasets(datasets)

        shuffle_ds = dataset.shuffle(seed=seed, writer_batch_size=1000)
        size = len(shuffle_ds)
        if max_size > 0:
            size = max_size

        with open(f"shuffle_corpus_{size}.text", "w", encoding='utf-8') as file:
            for i in tqdm(range(0, size)):
                text = shuffle_ds[i]['text'].strip()
                if len(text) > 0: 
                    file.write(text)
        
        print(f"finished training file shuffle_corpus_{size}.text")

    def train(self, input_file, vocab_size = 65024, model_name = "tokenizer", character_coverage = 0.999,
              input_sentence_size = 10000, shuffle_input_sentence = True, num_threads = 31):
        """
            -v, --vocab_size=VOCAB_SIZE
                Default: 32767
            -m, --model_name=MODEL_NAME
                Default: 'tokenizer'
            -c, --character_coverage=CHARACTER_COVERAGE
                Default: 0.999
            -i, --input_sentence_size=INPUT_SENTENCE_SIZE
                Default: 10000
            -s, --shuffle_input_sentence=SHUFFLE_INPUT_SENTENCE
                Default: True
            -n, --num_threads=NUM_THREADS
                Default: 31
        """
        SentencePieceTrainer.Train(
            f"--input={input_file} "
            f"--vocab_size={vocab_size} "
            f"--model_prefix={model_name} "
            f"--max_sentence_length=5000 "
            f"--shuffle_input_sentence={str(shuffle_input_sentence).lower()} "
            f"--input_sentence_size={input_sentence_size} "
            f"--num_threads={num_threads} "
            f"--train_extremely_large_corpus=true "
            f"--character_coverage={character_coverage} "
        )
        
if __name__ == "__main__":
    fire.Fire(MakeTokenizerModel)