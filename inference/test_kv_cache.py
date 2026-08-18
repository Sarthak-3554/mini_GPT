import torch

from model.gpt import MiniGPT
from tokenizer.tokenizer import BPETokenizer


CHECKPOINT = "checkpoints/best.pt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"

PROMPT = "ROMEO:"
NUM_TOKENS = 20


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
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = MiniGPT(
        checkpoint["config"]
    )

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
        f"Validation loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    return model


# ============================================================
# Normal generation
# ============================================================

@torch.no_grad()
def generate_normal(
    model,
    tokenizer,
    prompt,
    num_tokens,
):

    device = next(
        model.parameters()
    ).device

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

    generated_logits = []

    for _ in range(num_tokens):

        idx_cond = idx[
            :,
            -model.config.block_size:
        ]

        logits, _ = model(
            idx_cond
        )

        # Save logits corresponding to
        # the next generated token.
        next_logits = logits[:, -1, :]

        generated_logits.append(
            next_logits.clone()
        )

        # Greedy decoding
        next_token = torch.argmax(
            next_logits,
            dim=-1,
            keepdim=True,
        )

        idx = torch.cat(
            [
                idx,
                next_token,
            ],
            dim=1,
        )

    return generated_logits


# ============================================================
# KV-cache generation
# ============================================================

@torch.no_grad()
def generate_cached(
    model,
    tokenizer,
    prompt,
    num_tokens,
):

    device = next(
        model.parameters()
    ).device

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

    # --------------------------------------------------------
    # Initial prompt
    # --------------------------------------------------------

    logits, _, past_key_values = model(
        idx,
        use_cache=True,
        start_pos=0,
    )

    next_logits = logits[:, -1, :]

    generated_logits = []

    generated_logits.append(
        next_logits.clone()
    )

    next_token = torch.argmax(
        next_logits,
        dim=-1,
        keepdim=True,
    )

    current_pos = idx.shape[1]

    # --------------------------------------------------------
    # Cached generation
    # --------------------------------------------------------

    for _ in range(
        num_tokens - 1
    ):

        logits, _, past_key_values = model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True,
            start_pos=current_pos,
        )

        next_logits = logits[:, -1, :]

        generated_logits.append(
            next_logits.clone()
        )

        next_token = torch.argmax(
            next_logits,
            dim=-1,
            keepdim=True,
        )

        current_pos += 1

    return generated_logits


# ============================================================
# Compare
# ============================================================

def compare_logits(
    normal_logits,
    cached_logits,
):

    print(
        "\n========================================"
    )

    print(
        "KV CACHE CORRECTNESS TEST"
    )

    print(
        "========================================"
    )

    overall_max_difference = 0.0
    overall_mean_difference = 0.0

    all_match = True

    for step, (
        normal,
        cached,
    ) in enumerate(
        zip(
            normal_logits,
            cached_logits,
        )
    ):

        difference = (
            normal - cached
        ).abs()

        max_difference = (
            difference.max().item()
        )

        mean_difference = (
            difference.mean().item()
        )

        overall_max_difference = max(
            overall_max_difference,
            max_difference,
        )

        overall_mean_difference += (
            mean_difference
        )

        # ----------------------------------------------------
        # Check top predicted token
        # ----------------------------------------------------

        normal_token = torch.argmax(
            normal,
            dim=-1,
        )

        cached_token = torch.argmax(
            cached,
            dim=-1,
        )

        same_token = (
            normal_token
            == cached_token
        ).all().item()

        print(
            f"Step {step:2d} | "
            f"max diff: {max_difference:.8f} | "
            f"mean diff: {mean_difference:.8f} | "
            f"same prediction: {same_token}"
        )

        if not same_token:
            all_match = False

    overall_mean_difference /= len(
        normal_logits
    )

    print(
        "\n----------------------------------------"
    )

    print(
        f"Overall max difference : "
        f"{overall_max_difference:.8f}"
    )

    print(
        f"Overall mean difference: "
        f"{overall_mean_difference:.8f}"
    )

    print(
        f"Greedy predictions match: "
        f"{all_match}"
    )

    # Small floating-point differences are expected.
    # The important thing is that predictions match.

    if all_match:

        print(
            "\nKV cache correctness: PASS"
        )

    else:

        print(
            "\nKV cache correctness: FAIL"
        )


# ============================================================
# Main
# ============================================================

def main():

    device = get_device()

    print(
        f"Using device: {device}"
    )

    tokenizer = BPETokenizer(
        TOKENIZER_PATH
    )

    model = load_model(
        CHECKPOINT,
        device,
    )

    print(
        f"\nPrompt: {PROMPT}"
    )

    print(
        f"Testing {NUM_TOKENS} generated tokens..."
    )

    # --------------------------------------------------------
    # Normal
    # --------------------------------------------------------

    print(
        "\nRunning normal generation..."
    )

    normal_logits = generate_normal(
        model,
        tokenizer,
        PROMPT,
        NUM_TOKENS,
    )

    # --------------------------------------------------------
    # KV cache
    # --------------------------------------------------------

    print(
        "Running KV-cache generation..."
    )

    cached_logits = generate_cached(
        model,
        tokenizer,
        PROMPT,
        NUM_TOKENS,
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    compare_logits(
        normal_logits,
        cached_logits,
    )


if __name__ == "__main__":
    main()