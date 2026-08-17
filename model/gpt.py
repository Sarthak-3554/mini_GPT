import torch
import torch.nn as nn

from .config import GPTConfig
from .rmsnorm import RMSNorm
from .block import TransformerBlock


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layer)
        ])

        # Final normalization
        self.final_norm = RMSNorm(
            config.n_embd
        )

        # Language-model head
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        assert T <= self.config.block_size

        # Token embeddings
        x = self.token_embedding(idx)

        # Transformer
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Vocabulary logits
        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1)
            )

        return logits, loss