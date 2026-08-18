import json
import random

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model.gpt import MiniGPT

from tokenizer.tokenizer import BPETokenizer

from finetuning.dataset import SFTDataset

from finetuning.lora import (
    replace_linear_with_lora,
    mark_only_lora_trainable,
    count_parameters,
)


# ============================================================
# Configuration
# ============================================================

TOKENIZER_PATH = (
    "tokenizer/tokenizer.json"
)

DATA_PATH = (
    "data/sft_conversations.jsonl"
)

PRETRAINED_CHECKPOINT = (
    "checkpoints/best.pt"
)

OUTPUT_CHECKPOINT = (
    "checkpoints/lora.pt"
)

BLOCK_SIZE = 256

BATCH_SIZE = 16

MAX_STEPS = 5000

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 0.01

VAL_RATIO = 0.1

SEED = 42

# LoRA
LORA_RANK = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.0


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# Dataset
# ============================================================

def load_examples():

    examples = []

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            examples.append(
                json.loads(line)
            )

    return examples


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    losses = []

    for batch in loader:

        input_ids = batch[
            "input_ids"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        logits, _ = model(
            input_ids
        )

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            labels.reshape(-1),
            ignore_index=-100,
        )

        losses.append(
            loss.item()
        )

    model.train()

    if not losses:
        return float("inf")

    return sum(losses) / len(losses)


# ============================================================
# Main
# ============================================================

def main():

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = BPETokenizer(
        TOKENIZER_PATH
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    examples = load_examples()

    print(
        f"SFT examples: "
        f"{len(examples):,}"
    )

    dataset = SFTDataset(
        examples,
        tokenizer,
        BLOCK_SIZE,
    )

    val_size = int(
        len(dataset) * VAL_RATIO
    )

    train_size = (
        len(dataset) - val_size
    )

    generator = (
        torch.Generator()
        .manual_seed(SEED)
    )

    train_dataset, val_dataset = (
        random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )
    )

    print(
        f"Train examples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation examples: "
        f"{len(val_dataset):,}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Load pretrained MiniGPT
    # --------------------------------------------------------

    checkpoint = torch.load(
        PRETRAINED_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    model = MiniGPT(
        config
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # Add LoRA
    # --------------------------------------------------------

    replace_linear_with_lora(
    model,
    rank=LORA_RANK,
    alpha=LORA_ALPHA,
    dropout=LORA_DROPOUT,
    target_names=[
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    ],
)

    # Freeze base model
    mark_only_lora_trainable(
        model
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Parameter statistics
    # --------------------------------------------------------

    total_params, trainable_params, percentage = (
        count_parameters(model)
    )

    print()
    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        f"Trainable percentage: "
        f"{percentage:.4f}%"
    )

    print()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")

    step = 0

    progress = tqdm(
        total=MAX_STEPS,
        desc="LoRA",
    )

    model.train()

    while step < MAX_STEPS:

        for batch in train_loader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            logits, _ = model(
                input_ids
            )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                labels.reshape(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                ],
                1.0,
            )

            optimizer.step()

            # ------------------------------------------------
            # Evaluation
            # ------------------------------------------------

            if step % 50 == 0:

                val_loss = evaluate(
                    model,
                    val_loader,
                    device,
                )

                print(
                    f"step {step} | "
                    f"train loss "
                    f"{loss.item():.4f} | "
                    f"val loss "
                    f"{val_loss:.4f}"
                )

                if val_loss < best_val_loss:

                    best_val_loss = val_loss

                    # Save LoRA adapter + config.
                    torch.save(
                        {
                            "model_state_dict":
                                model.state_dict(),
                            "config":
                                config,
                            "step":
                                step,
                            "val_loss":
                                val_loss,
                            "lora_rank":
                                LORA_RANK,
                            "lora_alpha":
                                LORA_ALPHA,
                        },
                        OUTPUT_CHECKPOINT,
                    )

                    print(
                        "Saved best LoRA checkpoint "
                        f"(val loss: "
                        f"{val_loss:.4f})"
                    )

            progress.update(1)

            step += 1

            if step >= MAX_STEPS:
                break

    progress.close()

    print()
    print("LoRA training complete.")

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )


if __name__ == "__main__":
    main()