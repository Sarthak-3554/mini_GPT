import torch


def precompute_rope(seq_len, head_dim, device, theta=10000.0):
    position = torch.arange(
        seq_len,
        device=device,
        dtype=torch.float32
    )

    freq = 1.0 / (
        theta ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=device,
                dtype=torch.float32
            )
            / head_dim
        )
    )

    angles = torch.outer(position, freq)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(q, k, cos, sin):
    # q, k: [B, H, T, D]

    q1 = q[..., 0::2]
    q2 = q[..., 1::2]

    k1 = k[..., 0::2]
    k2 = k[..., 1::2]

    q_rotated = torch.stack(
        [
            q1 * cos - q2 * sin,
            q1 * sin + q2 * cos,
        ],
        dim=-1
    )

    k_rotated = torch.stack(
        [
            k1 * cos - k2 * sin,
            k1 * sin + k2 * cos,
        ],
        dim=-1
    )

    q_rotated = q_rotated.flatten(-2)
    k_rotated = k_rotated.flatten(-2)

    return q_rotated, k_rotated