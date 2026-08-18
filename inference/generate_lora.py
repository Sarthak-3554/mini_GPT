import argparse

import torch

from model.gpt import MiniGPT
from tokenizer.tokenizer import BPETokenizer

from finetuning.lora import (
    replace_linear_with_lora,
)


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
# Load LoRA model
# ============================================================

def load_lora_model(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    model = MiniGPT(config)

    # --------------------------------------------------------
    # Recreate the exact LoRA architecture
    # --------------------------------------------------------

    replace_linear_with_lora(
        model,
        rank=checkpoint["lora_rank"],
        alpha=checkpoint["lora_alpha"],
        dropout=0.0,
        target_names=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
        ],
    )

    # --------------------------------------------------------
    # Load trained LoRA + base weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print(
        f"Loaded LoRA checkpoint "
        f"from step {checkpoint['step']}"
    )

    print(
        f"Checkpoint validation loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    print(
        f"LoRA rank: "
        f"{checkpoint['lora_rank']}"
    )

    print(
        f"LoRA alpha: "
        f"{checkpoint['lora_alpha']}"
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
    temperature=0.7,
    top_k=50,
):

    device = next(
        model.parameters()
    ).device

    # --------------------------------------------------------
    # Conversation prompt
    # --------------------------------------------------------

    user_tokens = tokenizer.encode(
        prompt
    )

    token_ids = (
        [tokenizer.user_token_id]
        + user_tokens
        + [tokenizer.assistant_token_id]
    )

    idx = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    for _ in range(max_new_tokens):

        idx_cond = idx[
            :, -model.config.block_size:
        ]

        logits, _ = model(
            idx_cond
        )

        logits = logits[:, -1, :]

        # Temperature
        logits = logits / max(
            temperature,
            1e-8,
        )

        # ----------------------------------------------------
        # Top-k
        # ----------------------------------------------------

        if top_k is not None:

            k = min(
                top_k,
                logits.size(-1),
            )

            values, _ = torch.topk(
                logits,
                k,
            )

            threshold = (
                values[:, -1]
                .unsqueeze(-1)
            )

            logits = torch.where(
                logits < threshold,
                torch.full_like(
                    logits,
                    float("-inf"),
                ),
                logits,
            )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        idx = torch.cat(
            [idx, next_token],
            dim=1,
        )

        # ----------------------------------------------------
        # Stop at <|end|>
        # ----------------------------------------------------

        if (
            next_token.item()
            == tokenizer.end_token_id
        ):
            break

    # --------------------------------------------------------
    # Remove conversation prefix
    # --------------------------------------------------------

    generated_tokens = idx[
        0,
        len(token_ids):,
    ].tolist()

    return tokenizer.decode(
        generated_tokens
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/lora.pt",
    )

    parser.add_argument(
        "--prompt",
        default="What is machine learning?",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
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

    tokenizer = BPETokenizer(
        "tokenizer/tokenizer.json"
    )

    model = load_lora_model(
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

    print(
        "\n--- GENERATED TEXT ---\n"
    )

    print(output)


if __name__ == "__main__":
    main()