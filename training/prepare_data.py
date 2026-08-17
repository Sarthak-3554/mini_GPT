from pathlib import Path

import torch

from tokenizer.tokenizer import BPETokenizer

def prepare_data(
    text_path,
    tokenizer_path,
    train_output,
    val_output,
    train_ratio=0.9,
):

    text = Path(text_path).read_text(
        encoding="utf-8"
    )

    tokenizer = BPETokenizer(
        tokenizer_path
    )

    token_ids = tokenizer.encode(text)

    print(
        f"Characters: {len(text):,}"
    )

    print(
        f"Tokens: {len(token_ids):,}"
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    split = int(
        len(token_ids) * train_ratio
    )

    train_tokens = torch.tensor(
        token_ids[:split],
        dtype=torch.long,
    )

    val_tokens = torch.tensor(
        token_ids[split:],
        dtype=torch.long,
    )

    torch.save(
        train_tokens,
        train_output
    )

    torch.save(
        val_tokens,
        val_output
    )

    print(
        f"Train tokens: "
        f"{len(train_tokens):,}"
    )

    print(
        f"Validation tokens: "
        f"{len(val_tokens):,}"
    )


if __name__ == "__main__":

    prepare_data(
        text_path="data/train.txt",
        tokenizer_path="tokenizer/tokenizer.json",
        train_output="data/train_tokens.pt",
        val_output="data/val_tokens.pt",
    )