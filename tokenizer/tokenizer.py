from tokenizers import Tokenizer


class BPETokenizer:

    def __init__(self, tokenizer_path: str):

        self.tokenizer = Tokenizer.from_file(
            tokenizer_path
        )

        # ----------------------------------------------------
        # Special token IDs
        # ----------------------------------------------------

        self.unk_token_id = self.tokenizer.token_to_id(
            "<unk>"
        )

        self.pad_token_id = self.tokenizer.token_to_id(
            "<|pad|>"
        )

        self.user_token_id = self.tokenizer.token_to_id(
            "<|user|>"
        )

        self.assistant_token_id = self.tokenizer.token_to_id(
            "<|assistant|>"
        )

        self.end_token_id = self.tokenizer.token_to_id(
            "<|end|>"
        )

        # Use <|end|> as EOS during generation.
        self.eos_token_id = self.end_token_id

        # ----------------------------------------------------
        # Validate special tokens
        # ----------------------------------------------------

        special_tokens = {
            "<unk>": self.unk_token_id,
            "<|pad|>": self.pad_token_id,
            "<|user|>": self.user_token_id,
            "<|assistant|>": self.assistant_token_id,
            "<|end|>": self.end_token_id,
        }

        for token, token_id in special_tokens.items():

            if token_id is None:
                raise ValueError(
                    f"Special token {token!r} "
                    "was not found in tokenizer."
                )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    @property
    def vocab_size(self):

        return self.tokenizer.get_vocab_size()

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    def encode(
        self,
        text: str,
    ):

        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
        ).ids

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    def decode(
        self,
        token_ids,
    ):

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
        )

    # --------------------------------------------------------
    # Token helpers
    # --------------------------------------------------------

    def encode_conversation(
        self,
        user_message: str,
        assistant_message: str | None = None,
    ):

        tokens = []

        # <|user|>
        tokens.append(
            self.user_token_id
        )

        # User message
        tokens.extend(
            self.encode(user_message)
        )

        if assistant_message is not None:

            # <|assistant|>
            tokens.append(
                self.assistant_token_id
            )

            # Assistant response
            tokens.extend(
                self.encode(assistant_message)
            )

            # End of assistant response
            tokens.append(
                self.end_token_id
            )

        return tokens


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    tokenizer = BPETokenizer(
        "tokenizer/tokenizer.json"
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size,
    )

    print(
        "Special tokens:"
    )

    print(
        "<unk>:",
        tokenizer.unk_token_id,
    )

    print(
        "<|pad|>:",
        tokenizer.pad_token_id,
    )

    print(
        "<|user|>:",
        tokenizer.user_token_id,
    )

    print(
        "<|assistant|>:",
        tokenizer.assistant_token_id,
    )

    print(
        "<|end|>:",
        tokenizer.end_token_id,
    )

    tokens = tokenizer.encode_conversation(
        "What is machine learning?",
        "Machine learning is a method of learning patterns from data.",
    )

    print(
        "\nConversation tokens:"
    )

    print(tokens)

    print(
        "\nDecoded:"
    )

    print(
        tokenizer.decode(tokens)
    )