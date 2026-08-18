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

def load_model(
    checkpoint_path,
    device
):

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
# Sampling
# ============================================================

def sample_next_token(
    logits,
    temperature,
    top_k,
):

    logits = logits / temperature

    if top_k is not None:

        values, _ = torch.topk(
            logits,
            min(
                top_k,
                logits.size(-1)
            ),
        )

        threshold = (
            values[:, -1]
            .unsqueeze(-1)
        )

        logits = torch.where(
            logits < threshold,
            torch.full_like(
                logits,
                float("-inf")
            ),
            logits,
        )

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    return torch.multinomial(
        probabilities,
        num_samples=1
    )


# ============================================================
# KV-cache generation
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

    # --------------------------------------------------------
    # Use EXACTLY the same prompt formatting as the
    # known-good generate.py
    # --------------------------------------------------------

    formatted_prompt = (
        "<|user|>\n"
        + prompt
        + "\n"
        + "<|assistant|>\n"
    )

    token_ids = tokenizer.encode(
        formatted_prompt
    )

    idx = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    # ========================================================
    # FIRST PASS
    # ========================================================

    logits, _, past_key_values = model(
        idx,
        use_cache=True,
        start_pos=0,
    )

    logits = logits[:, -1, :]

    next_token = sample_next_token(
        logits,
        temperature,
        top_k,
    )

    generated_tokens = [
        next_token.item()
    ]

    # Number of tokens already stored in cache
    current_pos = idx.shape[1]

    # ========================================================
    # SUBSEQUENT PASSES
    # ========================================================

    for _ in range(
        max_new_tokens - 1
    ):

        # IMPORTANT:
        #
        # Only the NEW token enters the model.
        #
        # Previous tokens are already represented
        # by past_key_values.

        logits, _, past_key_values = model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True,
            start_pos=current_pos,
        )

        logits = logits[:, -1, :]

        next_token = sample_next_token(
            logits,
            temperature,
            top_k,
        )

        generated_tokens.append(
            next_token.item()
        )

        current_pos += 1

        if (
            tokenizer.eos_token_id is not None
            and next_token.item()
            == tokenizer.eos_token_id
        ):
            break

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

    tokenizer = BPETokenizer(
        "tokenizer/tokenizer.json"
    )

    model = load_model(
        args.checkpoint,
        device
    )

    print(
        f"Prompt: {args.prompt}"
    )

    print(
        "\nGeneration mode: KV cache"
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