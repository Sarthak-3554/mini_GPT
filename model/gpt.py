# import torch
# import torch.nn as nn

# from .config import GPTConfig
# from .rmsnorm import RMSNorm
# from .block import TransformerBlock


# class MiniGPT(nn.Module):
#     def __init__(self, config: GPTConfig):
#         super().__init__()

#         self.config = config

#         # Token embeddings
#         self.token_embedding = nn.Embedding(
#             config.vocab_size,
#             config.n_embd
#         )

#         # Transformer blocks
#         self.blocks = nn.ModuleList([
#             TransformerBlock(config)
#             for _ in range(config.n_layer)
#         ])

#         # Final normalization
#         self.final_norm = RMSNorm(
#             config.n_embd
#         )

#         # Language-model head
#         self.lm_head = nn.Linear(
#             config.n_embd,
#             config.vocab_size,
#             bias=False
#         )

#         # Weight tying
#         self.lm_head.weight = (
#             self.token_embedding.weight
#         )

#         self.apply(self._init_weights)

#     def _init_weights(self, module):

#         if isinstance(module, nn.Linear):
#             nn.init.normal_(
#                 module.weight,
#                 mean=0.0,
#                 std=0.02
#             )

#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)

#         elif isinstance(module, nn.Embedding):
#             nn.init.normal_(
#                 module.weight,
#                 mean=0.0,
#                 std=0.02
#             )

#     def forward(self, idx, targets=None):

#         B, T = idx.shape

#         assert T <= self.config.block_size

#         # Token embeddings
#         x = self.token_embedding(idx)

#         # Transformer
#         for block in self.blocks:
#             x = block(x)

#         # Final normalization
#         x = self.final_norm(x)

#         # Vocabulary logits
#         logits = self.lm_head(x)

#         loss = None

#         if targets is not None:

#             loss = nn.functional.cross_entropy(
#                 logits.view(-1, self.config.vocab_size),
#                 targets.view(-1)
#             )

#         return logits, loss


import torch
import torch.nn as nn

from .config import GPTConfig
from .rmsnorm import RMSNorm
from .block import TransformerBlock


class MiniGPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # ====================================================
        # Token embeddings
        # ====================================================

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        # ====================================================
        # Transformer blocks
        # ====================================================

        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layer)
        ])

        # ====================================================
        # Final normalization
        # ====================================================

        self.final_norm = RMSNorm(
            config.n_embd
        )

        # ====================================================
        # Language model head
        # ====================================================

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(
            self._init_weights
        )

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        idx,
        targets=None,
        past_key_values=None,
        use_cache=False,
        start_pos=0,
    ):

        B, T = idx.shape

        assert (
            start_pos + T
            <= self.config.block_size
        )

        # ====================================================
        # Token embeddings
        # ====================================================

        x = self.token_embedding(
            idx
        )

        # ====================================================
        # Transformer
        # ====================================================

        present_key_values = []

        for layer_idx, block in enumerate(
            self.blocks
        ):

            past_kv = None

            if past_key_values is not None:

                past_kv = (
                    past_key_values[layer_idx]
                )

            x, present_kv = block(
                x,
                past_kv=past_kv,
                use_cache=use_cache,
                start_pos=start_pos,
            )

            if use_cache:

                present_key_values.append(
                    present_kv
                )

        # ====================================================
        # Final normalization
        # ====================================================

        x = self.final_norm(x)

        # ====================================================
        # Language model head
        # ====================================================

        logits = self.lm_head(x)

        # ====================================================
        # Training loss
        # ====================================================

        loss = None

        if targets is not None:

            loss = nn.functional.cross_entropy(
                logits.view(
                    -1,
                    self.config.vocab_size
                ),
                targets.view(-1)
            )

        if use_cache:

            return (
                logits,
                loss,
                present_key_values
            )

        return logits, loss