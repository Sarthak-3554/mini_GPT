import json
from pathlib import Path


INPUT_PATH = "data/sft.jsonl"
OUTPUT_PATH = "data/sft_conversations.jsonl"


def convert():

    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)

    examples = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            examples.append({
                "conversations": [
                    {
                        "role": "user",
                        "content": row["instruction"],
                    },
                    {
                        "role": "assistant",
                        "content": row["response"],
                    },
                ]
            })

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
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

    print(f"Converted examples: {len(examples):,}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    convert()