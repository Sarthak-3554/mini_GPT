import torch

from tokenizer.tokenizer import BPETokenizer
from finetuning.dataset import SFTDataset


TOKENIZER_PATH = "tokenizer/tokenizer.json"
DATA_PATH = "data/sft.jsonl"


def main():

    import json

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        examples = [
            json.loads(line)
            for line in f
        ]

    tokenizer = BPETokenizer(
        TOKENIZER_PATH
    )

    dataset = SFTDataset(
        examples,
        tokenizer,
        block_size=256,
    )

    sample = dataset[0]

    input_ids = sample["input_ids"]
    labels = sample["labels"]

    print(
        "Input shape:",
        input_ids.shape,
    )

    print(
        "Labels shape:",
        labels.shape,
    )

    # Tokens participating in loss
    valid_labels = labels[
        labels != -100
    ]

    print(
        "Response tokens:",
        len(valid_labels),
    )

    print(
        "Total tokens:",
        len(input_ids),
    )

    print("\n--- INPUT ---")

    print(
        tokenizer.decode(
            input_ids.tolist()
        )
    )

    print("\n--- TARGET ---")

    target_ids = [
        x
        for x in labels.tolist()
        if x != -100
    ]

    print(
        tokenizer.decode(
            target_ids
        )
    )


if __name__ == "__main__":
    main()