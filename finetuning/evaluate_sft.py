import json

import torch
import torch.nn.functional as F

from model.gpt import MiniGPT
from tokenizer.tokenizer import BPETokenizer


TOKENIZER_PATH = "tokenizer/tokenizer.json"

PRETRAINED = "checkpoints/best.pt"
SFT = "checkpoints/sft.pt"

DATA = "data/sft.jsonl"

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


def load_example():

    with open(
        DATA,
        "r",
        encoding="utf-8",
    ) as f:

        example = json.loads(
            f.readline()
        )

    return example


def build_sequence(
    example,
    tokenizer,
    block_size,
):

    prompt = (
        "### Instruction:\n"
        + example["instruction"]
        + "\n\n"
        + "### Response:\n"
    )

    full_text = (
        prompt
        + example["response"]
    )

    prompt_ids = tokenizer.encode(
        prompt
    )

    full_ids = tokenizer.encode(
        full_text
    )

    full_ids = full_ids[
        : block_size + 1
    ]

    input_ids = torch.tensor(
        [full_ids[:-1]],
        dtype=torch.long,
        device=DEVICE,
    )

    labels = torch.tensor(
        [full_ids[1:]],
        dtype=torch.long,
        device=DEVICE,
    )

    # Ignore prompt portion
    prompt_length = min(
        len(prompt_ids),
        labels.shape[1],
    )

    labels[:, :prompt_length] = -100

    return input_ids, labels


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path,
    input_ids,
    labels,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = MiniGPT(
        checkpoint["config"]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    logits, _ = model(
        input_ids
    )

    loss = F.cross_entropy(
        logits.view(
            -1,
            logits.size(-1),
        ),
        labels.view(-1),
        ignore_index=-100,
    )

    return loss.item()


def main():

    tokenizer = BPETokenizer(
        TOKENIZER_PATH
    )

    example = load_example()

    input_ids, labels = build_sequence(
        example,
        tokenizer,
        block_size=256,
    )

    print(
        "Evaluating example:"
    )

    print(
        example["instruction"]
    )

    print()

    pretrained_loss = evaluate_checkpoint(
        PRETRAINED,
        input_ids,
        labels,
    )

    sft_loss = evaluate_checkpoint(
        SFT,
        input_ids,
        labels,
    )

    print(
        f"Pre-SFT response loss: "
        f"{pretrained_loss:.4f}"
    )

    print(
        f"Post-SFT response loss: "
        f"{sft_loss:.4f}"
    )

    print(
        f"Improvement: "
        f"{pretrained_loss - sft_loss:.4f}"
    )


if __name__ == "__main__":
    main()