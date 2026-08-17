import argparse

import torch

from model.gpt import MiniGPT
from tokenizer.tokenizer import BPETokenizer


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# Load model
# ============================================================

def load_model(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    model = MiniGPT(config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint from step "
        f"{checkpoint['step']}"
    )

    print(
        f"Checkpoint validation loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    return model


# ============================================================
# Generate
# ============================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
):

    device = next(
        model.parameters()
    ).device

    # Encode prompt
    token_ids = tokenizer.encode(
        prompt
    )

    idx = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new_tokens):

        # Keep only the model's context window
        idx_cond = idx[
            :, -model.config.block_size:
        ]

        # Forward pass
        logits, _ = model(
            idx_cond
        )

        # We only care about the last position
        logits = logits[:, -1, :]

        # Temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k is not None:

            values, _ = torch.topk(
                logits,
                min(top_k, logits.size(-1)),
            )

            threshold = values[:, -1].unsqueeze(-1)

            logits = torch.where(
                logits < threshold,
                torch.full_like(
                    logits,
                    float("-inf")
                ),
                logits,
            )

        # Convert logits → probabilities
        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        # Sample next token
        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        # Append token
        idx = torch.cat(
            [idx, next_token],
            dim=1,
        )

        # Stop at EOS
        if (
            tokenizer.eos_token_id is not None
            and next_token.item()
            == tokenizer.eos_token_id
        ):
            break

    # Decode complete sequence
    return tokenizer.decode(
        idx[0].tolist()
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
    )

    parser.add_argument(
        "--prompt",
        default="ROMEO:",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    device = get_device()

    print(
        f"Using device: {device}"
    )

    # Load tokenizer
    tokenizer = BPETokenizer(
        "tokenizer/tokenizer.json"
    )

    # Load model
    model = load_model(
        args.checkpoint,
        device,
    )

    print(
        f"Prompt: {args.prompt}"
    )

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\n--- GENERATED TEXT ---\n")
    print(output)


if __name__ == "__main__":
    main()