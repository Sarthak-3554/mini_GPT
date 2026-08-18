import time

import torch

from model.gpt import MiniGPT
from tokenizer.tokenizer import BPETokenizer


CHECKPOINT = "checkpoints/best.pt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"

PROMPT = "ROMEO:"

# Candidate generation lengths.
# The benchmark will automatically remove lengths
# that exceed the model's context window.
CANDIDATE_LENGTHS = [
    25,
    50,
    100,
    200,
    400,
]

WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


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
# Synchronization
# ============================================================

def synchronize(device):

    if device.type == "mps":
        torch.mps.synchronize()

    elif device.type == "cuda":
        torch.cuda.synchronize()


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
# Encode prompt
# ============================================================

def encode_prompt(
    tokenizer,
    prompt,
    device,
):

    # Keep exactly the same formatting used by
    # the working generation code.

    formatted_prompt = (
        "<|user|>\n"
        + prompt
        + "\n"
        + "<|assistant|>\n"
    )

    token_ids = tokenizer.encode(
        formatted_prompt
    )

    return torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )


# ============================================================
# Determine valid generation lengths
# ============================================================

def get_generation_lengths(
    model,
    prompt_tokens,
):

    context_length = (
        model.config.block_size
    )

    prompt_length = (
        prompt_tokens.shape[1]
    )

    max_generation_tokens = (
        context_length
        - prompt_length
    )

    if max_generation_tokens <= 0:

        raise RuntimeError(
            f"Prompt contains {prompt_length} tokens, "
            f"but model context window is only "
            f"{context_length} tokens."
        )

    generation_lengths = [
        n
        for n in CANDIDATE_LENGTHS
        if n <= max_generation_tokens
    ]

    # If even 25 doesn't fit, use the largest
    # possible generation length.

    if not generation_lengths:

        generation_lengths = [
            max_generation_tokens
        ]

    return (
        generation_lengths,
        context_length,
        max_generation_tokens,
    )


# ============================================================
# Normal generation
# ============================================================

@torch.no_grad()
def generate_normal(
    model,
    prompt_tokens,
    num_new_tokens,
):

    idx = prompt_tokens.clone()

    start_time = time.perf_counter()

    for _ in range(num_new_tokens):

        idx_cond = idx[
            :,
            -model.config.block_size:
        ]

        logits, _ = model(
            idx_cond
        )

        logits = logits[:, -1, :]

        # Greedy decoding.
        # This makes both implementations deterministic.

        next_token = torch.argmax(
            logits,
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

    synchronize(
        prompt_tokens.device
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return elapsed


# ============================================================
# KV-cache generation
# ============================================================

@torch.no_grad()
def generate_cached(
    model,
    prompt_tokens,
    num_new_tokens,
):

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # First pass
    # --------------------------------------------------------

    logits, _, past_key_values = model(
        prompt_tokens,
        use_cache=True,
        start_pos=0,
    )

    logits = logits[:, -1, :]

    next_token = torch.argmax(
        logits,
        dim=-1,
        keepdim=True,
    )

    current_pos = (
        prompt_tokens.shape[1]
    )

    # --------------------------------------------------------
    # Cached generation
    # --------------------------------------------------------

    for _ in range(
        num_new_tokens - 1
    ):

        # Only the newly generated token is
        # processed by the model.

        logits, _, past_key_values = model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True,
            start_pos=current_pos,
        )

        logits = logits[:, -1, :]

        next_token = torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

        current_pos += 1

    synchronize(
        prompt_tokens.device
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return elapsed


# ============================================================
# Benchmark helper
# ============================================================

def benchmark(
    fn,
    model,
    prompt_tokens,
    num_new_tokens,
):

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(
        WARMUP_RUNS
    ):

        fn(
            model,
            prompt_tokens,
            num_new_tokens,
        )

    # --------------------------------------------------------
    # Timed runs
    # --------------------------------------------------------

    times = []

    for _ in range(
        BENCHMARK_RUNS
    ):

        synchronize(
            prompt_tokens.device
        )

        elapsed = fn(
            model,
            prompt_tokens,
            num_new_tokens,
        )

        times.append(
            elapsed
        )

    return sum(times) / len(times)


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

    prompt_tokens = encode_prompt(
        tokenizer,
        PROMPT,
        device,
    )

    # --------------------------------------------------------
    # Context information
    # --------------------------------------------------------

    (
        generation_lengths,
        context_length,
        max_generation_tokens,
    ) = get_generation_lengths(
        model,
        prompt_tokens,
    )

    print(
        f"\nPrompt: {PROMPT}"
    )

    print(
        f"Prompt tokens: "
        f"{prompt_tokens.shape[1]}"
    )

    print(
        f"Context window: "
        f"{context_length}"
    )

    print(
        f"Maximum generation length: "
        f"{max_generation_tokens}"
    )

    print(
        f"Benchmark lengths: "
        f"{generation_lengths}"
    )

    print(
        f"Warmup runs: "
        f"{WARMUP_RUNS}"
    )

    print(
        f"Benchmark runs: "
        f"{BENCHMARK_RUNS}"
    )

    # ========================================================
    # Benchmark
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "KV CACHE BENCHMARK"
    )

    print(
        "=" * 90
    )

    print(
        f"{'Tokens':>8} | "
        f"{'No Cache (s)':>14} | "
        f"{'KV Cache (s)':>14} | "
        f"{'Speedup':>10} | "
        f"{'No Cache tok/s':>16} | "
        f"{'KV Cache tok/s':>16}"
    )

    print(
        "-" * 90
    )

    results = []

    for num_tokens in generation_lengths:

        # ----------------------------------------------------
        # Normal
        # ----------------------------------------------------

        normal_time = benchmark(
            generate_normal,
            model,
            prompt_tokens,
            num_tokens,
        )

        # ----------------------------------------------------
        # KV cache
        # ----------------------------------------------------

        cached_time = benchmark(
            generate_cached,
            model,
            prompt_tokens,
            num_tokens,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        speedup = (
            normal_time
            / cached_time
        )

        normal_tokens_per_sec = (
            num_tokens
            / normal_time
        )

        cached_tokens_per_sec = (
            num_tokens
            / cached_time
        )

        results.append(
            {
                "tokens": num_tokens,
                "normal_time": normal_time,
                "cached_time": cached_time,
                "speedup": speedup,
                "normal_tps": normal_tokens_per_sec,
                "cached_tps": cached_tokens_per_sec,
            }
        )

        print(
            f"{num_tokens:8d} | "
            f"{normal_time:14.4f} | "
            f"{cached_time:14.4f} | "
            f"{speedup:9.2f}x | "
            f"{normal_tokens_per_sec:16.2f} | "
            f"{cached_tokens_per_sec:16.2f}"
        )

    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 90
    )

    best = max(
        results,
        key=lambda x: x["speedup"]
    )

    average_speedup = (
        sum(
            result["speedup"]
            for result in results
        )
        / len(results)
    )

    print(
        f"Average speedup: "
        f"{average_speedup:.2f}x"
    )

    print(
        f"Best speedup: "
        f"{best['speedup']:.2f}x "
        f"at {best['tokens']} generated tokens"
    )

    print(
        f"Best no-cache throughput: "
        f"{best['normal_tps']:.2f} tokens/sec"
    )

    print(
        f"Best KV-cache throughput: "
        f"{best['cached_tps']:.2f} tokens/sec"
    )


if __name__ == "__main__":
    main()