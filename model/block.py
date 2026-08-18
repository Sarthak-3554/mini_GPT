# import torch.nn as nn

# from .rmsnorm import RMSNorm
# from .attention import GroupedQueryAttention
# from .swiglu import SwiGLU


# class TransformerBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()

#         self.attention_norm = RMSNorm(
#             config.n_embd
#         )

#         self.attention = GroupedQueryAttention(
#             config
#         )

#         self.ffn_norm = RMSNorm(
#             config.n_embd
#         )

#         # Common approximation for SwiGLU hidden dimension.
#         hidden_dim = int(
#             8 * config.n_embd / 3
#         )

#         # Round to a multiple of 256
#         hidden_dim = (
#             (hidden_dim + 255) // 256
#         ) * 256

#         self.ffn = SwiGLU(
#             config.n_embd,
#             hidden_dim
#         )

#     def forward(self, x):

#         # Pre-normalization
#         x = x + self.attention(
#             self.attention_norm(x)
#         )

#         # Feed-forward network
#         x = x + self.ffn(
#             self.ffn_norm(x)
#         )

#         return x


import torch.nn as nn

from .rmsnorm import RMSNorm
from .attention import GroupedQueryAttention
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.attention_norm = RMSNorm(
            config.n_embd
        )

        self.attention = GroupedQueryAttention(
            config
        )

        self.ffn_norm = RMSNorm(
            config.n_embd
        )

        hidden_dim = int(
            8 * config.n_embd / 3
        )

        hidden_dim = (
            (hidden_dim + 255) // 256
        ) * 256

        self.ffn = SwiGLU(
            config.n_embd,
            hidden_dim
        )

    def forward(
        self,
        x,
        past_kv=None,
        use_cache=False,
        start_pos=0,
    ):

        # ====================================================
        # Attention
        # ====================================================

        attention_output, present_kv = (
            self.attention(
                self.attention_norm(x),
                past_kv=past_kv,
                use_cache=use_cache,
                start_pos=start_pos,
            )
        )

        x = x + attention_output

        # ====================================================
        # Feed-forward
        # ====================================================

        x = x + self.ffn(
            self.ffn_norm(x)
        )

        return x, present_kv