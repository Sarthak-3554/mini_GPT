from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel


class BPETokenizer:

    def __init__(self, tokenizer_path: str):
        path = Path(tokenizer_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}"
            )

        self.tokenizer = Tokenizer.from_file(
            str(path)
        )

        # Convert ByteLevel representations
        # such as Ġ and Ċ back to normal text.
        self.tokenizer.decoder = ByteLevel()

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def bos_token_id(self):
        return self.tokenizer.token_to_id("<bos>")

    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id("<eos>")

    @property
    def pad_token_id(self):
        return self.tokenizer.token_to_id("<pad>")

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)