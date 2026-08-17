# model/config.py

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 8000
    block_size: int = 256

    n_embd: int = 256
    n_layer: int = 6

    n_head: int = 8
    n_kv_head: int = 2

    dropout: float = 0.0