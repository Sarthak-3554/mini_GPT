from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


VOCAB_SIZE = 8000

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]


def train_tokenizer(
    input_file: str,
    output_dir: str,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Byte-level preprocessing makes the tokenizer robust to
    # arbitrary UTF-8 text.
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )

    tokenizer.train(
        files=[input_file],
        trainer=trainer,
    )

    tokenizer.save(
        str(output_path / "tokenizer.json")
    )

    print("Tokenizer trained.")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Saved to: {output_path / 'tokenizer.json'}")


if __name__ == "__main__":
    train_tokenizer(
        input_file="data/train.txt",
        output_dir="tokenizer",
    )