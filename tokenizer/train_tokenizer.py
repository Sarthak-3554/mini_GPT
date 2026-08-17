from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# ============================================================
# Configuration
# ============================================================

VOCAB_SIZE = 8000

SPECIAL_TOKENS = [
    "<unk>",
    "<|pad|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]


# ============================================================
# Train tokenizer
# ============================================================

def train_tokenizer(
    input_file: str,
    output_dir: str,
):

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # BPE tokenizer
    # --------------------------------------------------------

    tokenizer = Tokenizer(
        BPE(
            unk_token="<unk>"
        )
    )

    # Byte-level preprocessing
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )
    tokenizer.decoder = ByteLevelDecoder()

    # --------------------------------------------------------
    # BPE trainer
    # --------------------------------------------------------

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )

    tokenizer.train(
        files=[input_file],
        trainer=trainer,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    tokenizer_path = (
        output_path / "tokenizer.json"
    )

    tokenizer.save(
        str(tokenizer_path)
    )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print("Tokenizer trained.")

    print(
        f"Vocabulary size: "
        f"{tokenizer.get_vocab_size()}"
    )

    print(
        "Special tokens:"
    )

    for token in SPECIAL_TOKENS:

        token_id = tokenizer.token_to_id(
            token
        )

        print(
            f"  {token}: {token_id}"
        )

    print(
        f"Saved to: {tokenizer_path}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    train_tokenizer(
        input_file="data/train.txt",
        output_dir="tokenizer",
    )