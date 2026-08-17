import torch

from model.config import GPTConfig
from model.gpt import MiniGPT


def main():

    config = GPTConfig()

    model = MiniGPT(config)

    x = torch.randint(
        0,
        config.vocab_size,
        (4, 64)
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (4, 64)
    )

    logits, loss = model(
        x,
        targets
    )

    print("Input shape :", x.shape)
    print("Logits shape:", logits.shape)
    print("Loss        :", loss.item())

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters  : {params:,}"
    )

    print(
        f"Parameters  : {params / 1e6:.2f}M"
    )


if __name__ == "__main__":
    main()