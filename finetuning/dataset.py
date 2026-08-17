import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):

    def __init__(
        self,
        examples,
        tokenizer,
        block_size,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        conversation = self.examples[idx][
            "conversations"
        ]

        user_message = conversation[0]["content"]
        assistant_message = conversation[1]["content"]

        # ----------------------------------------------------
        # Build tokens
        #
        # <|user|>
        # user text
        # <|assistant|>
        # assistant text
        # <|end|>
        # ----------------------------------------------------

        user_tokens = self.tokenizer.encode(
            user_message
        )

        assistant_tokens = self.tokenizer.encode(
            assistant_message
        )

        input_tokens = (
            [self.tokenizer.user_token_id]
            + user_tokens
            + [self.tokenizer.assistant_token_id]
            + assistant_tokens
            + [self.tokenizer.end_token_id]
        )

        # ----------------------------------------------------
        # Need block_size + 1 because:
        #
        # input  = tokens[:-1]
        # target = tokens[1:]
        # ----------------------------------------------------

        input_tokens = input_tokens[
            : self.block_size + 1
        ]

        actual_length = len(input_tokens)

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        if actual_length < self.block_size + 1:

            input_tokens += [
                self.tokenizer.pad_token_id
            ] * (
                self.block_size
                + 1
                - actual_length
            )

        # ----------------------------------------------------
        # Input / target
        # ----------------------------------------------------

        input_ids = torch.tensor(
            input_tokens[:-1],
            dtype=torch.long,
        )

        labels = torch.tensor(
            input_tokens[1:],
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Mask everything before assistant response
        #
        # We want the model to learn:
        #
        # <|assistant|> → response
        #
        # but NOT calculate loss on the user's question.
        # ----------------------------------------------------

        response_start = (
            1
            + len(user_tokens)
        )

        # The assistant token itself can also be
        # predicted, so mask through the user portion.
        labels[
            :response_start
        ] = -100

        # ----------------------------------------------------
        # Mask padding
        # ----------------------------------------------------

        if actual_length < self.block_size + 1:

            labels[
                actual_length - 1:
            ] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
        }