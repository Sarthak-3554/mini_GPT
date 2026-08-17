import json
import random
from pathlib import Path

from datasets import load_dataset


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "yahma/alpaca-cleaned"

OUTPUT_PATH = "data/sft.jsonl"

NUM_EXAMPLES = 5000

SEED = 42


# ============================================================
# Load dataset
# ============================================================

def load_alpaca():

    print(
        f"Loading dataset: {DATASET_NAME}"
    )

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    print(
        f"Original examples: {len(dataset):,}"
    )

    return dataset


# ============================================================
# Clean examples
# ============================================================

def clean_examples(dataset):

    examples = []

    for row in dataset:

        instruction = (
            row["instruction"]
            or ""
        ).strip()

        input_text = (
            row["input"]
            or ""
        ).strip()

        output = (
            row["output"]
            or ""
        ).strip()

        # Ignore incomplete examples
        if not instruction:
            continue

        if not output:
            continue

        # Build the instruction.
        #
        # If an input/context exists, include it.
        if input_text:

            combined_instruction = (
                instruction
                + "\n\n"
                + "Input:\n"
                + input_text
            )

        else:

            combined_instruction = instruction

        examples.append(
            {
                "instruction":
                    combined_instruction,
                "response":
                    output,
            }
        )

    print(
        f"Valid examples: {len(examples):,}"
    )

    return examples


# ============================================================
# Select reproducible subset
# ============================================================

def select_examples(
    examples,
    num_examples,
    seed,
):

    rng = random.Random(seed)

    # Shuffle a copy so that the original list
    # remains unchanged.
    examples = examples.copy()

    rng.shuffle(examples)

    num_examples = min(
        num_examples,
        len(examples),
    )

    selected = examples[
        :num_examples
    ]

    return selected


# ============================================================
# Save JSONL
# ============================================================

def save_jsonl(
    examples,
    output_path,
):

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for example in examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"Saved {len(examples):,} examples"
    )

    print(
        f"Saved to: {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    dataset = load_alpaca()

    examples = clean_examples(
        dataset
    )

    examples = select_examples(
        examples,
        NUM_EXAMPLES,
        SEED,
    )

    save_jsonl(
        examples,
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()