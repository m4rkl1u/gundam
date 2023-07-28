from transformers import PretrainedConfig
from dataclasses import dataclass

@dataclass
class TrainGundamConfig(PretrainedConfig):
    max_iters = 10
    learning_rate = 6e-3
    grad_clip = 1.0
    compile = True