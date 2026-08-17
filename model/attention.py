import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import precompute_rope, apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        self.n_rep = config.n_head // config.n_kv_head

        # Query projection
        self.q_proj = nn.Linear(
            config.n_embd,
            config.n_head * self.head_dim,
            bias=False
        )

        # Key and Value projections use fewer heads
        self.k_proj = nn.Linear(
            config.n_embd,
            config.n_kv_head * self.head_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            config.n_embd,
            config.n_kv_head * self.head_dim,
            bias=False
        )

        # Output projection
        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False
        )

        self.dropout = config.dropout

        # Precompute RoPE frequencies
        cos, sin = precompute_rope(
            config.block_size,
            self.head_dim,
            device="cpu"
        )

        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape

        # --------------------------------------------------
        # Project Q, K, V
        # --------------------------------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, H*D] -> [B, H, T, D]

        q = q.view(
            B, T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T,
            self.n_kv_head,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T,
            self.n_kv_head,
            self.head_dim
        ).transpose(1, 2)

        # --------------------------------------------------
        # Rotary positional embeddings
        # --------------------------------------------------

        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)

        q, k = apply_rope(q, k, cos, sin)

        # --------------------------------------------------
        # GQA
        # --------------------------------------------------

        # Example:
        #
        # 8 query heads
        # 2 KV heads
        #
        # Each KV head is shared by 4 query heads.

        if self.n_rep > 1:
            k = k.repeat_interleave(
                self.n_rep,
                dim=1
            )

            v = v.repeat_interleave(
                self.n_rep,
                dim=1
            )

        # --------------------------------------------------
        # Causal self-attention
        # --------------------------------------------------

        # PyTorch's scaled_dot_product_attention:
        #
        # softmax(QK^T / sqrt(d)) V
        #
        # is_causal=True prevents looking at future tokens.

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        # [B, H, T, D]
        # ->
        # [B, T, H*D]

        y = y.transpose(1, 2).contiguous()

        y = y.view(
            B,
            T,
            C
        )

        return self.out_proj(y)