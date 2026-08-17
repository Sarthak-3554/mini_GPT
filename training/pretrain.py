import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.config import GPTConfig
from model.gpt import MiniGPT
from training.dataset import LanguageModelDataset


# ============================================================
# Configuration
# ============================================================

SEED = 42

TRAIN_DATA = "data/train_tokens.pt"
VAL_DATA = "data/val_tokens.pt"

BATCH_SIZE = 32
BLOCK_SIZE = 256

MAX_STEPS = 3000

LEARNING_RATE = 3e-4
MIN_LR = 3e-5

WARMUP_STEPS = 100

EVAL_INTERVAL = 100
EVAL_ITERS = 20

GRAD_CLIP = 1.0

CHECKPOINT_DIR = "checkpoints"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
# Learning-rate scheduler
# ============================================================

def get_lr(step):

    # Linear warmup
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (
            (step + 1) / WARMUP_STEPS
        )

    # Cosine decay
    if step >= MAX_STEPS:
        return MIN_LR

    progress = (
        step - WARMUP_STEPS
    ) / (
        MAX_STEPS - WARMUP_STEPS
    )

    return (
        MIN_LR
        + 0.5
        * (LEARNING_RATE - MIN_LR)
        * (1 + math.cos(math.pi * progress))
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    train_loader,
    val_loader,
    device,
):

    model.eval()

    results = {}

    for name, loader in [
        ("train", train_loader),
        ("val", val_loader),
    ]:

        losses = []

        for i, (x, y) in enumerate(loader):

            if i >= EVAL_ITERS:
                break

            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    step,
    train_loss,
    val_loss,
    config,
    filename,
):

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }

    torch.save(
        checkpoint,
        filename,
    )


# ============================================================
# Training
# ============================================================

def train():

    set_seed(SEED)

    device = get_device()

    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Load tokenized data
    # --------------------------------------------------------

    train_tokens = torch.load(
        TRAIN_DATA,
        weights_only=True,
    )

    val_tokens = torch.load(
        VAL_DATA,
        weights_only=True,
    )

    print(
        f"Train tokens: {len(train_tokens):,}"
    )

    print(
        f"Validation tokens: {len(val_tokens):,}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = LanguageModelDataset(
        train_tokens,
        BLOCK_SIZE,
    )

    val_dataset = LanguageModelDataset(
        val_tokens,
        BLOCK_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )

    print(
        f"Training sequences: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation sequences: "
        f"{len(val_dataset):,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    config = GPTConfig(
        vocab_size=8000,
        block_size=BLOCK_SIZE,
        n_embd=256,
        n_layer=6,
        n_head=8,
        n_kv_head=2,
        dropout=0.0,
    )

    model = MiniGPT(config).to(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{num_params / 1e6:.2f}M"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.1,
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    import os

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    train_iterator = iter(
        train_loader
    )

    progress = tqdm(
        range(MAX_STEPS),
        desc="Training",
    )

    for step in progress:

        try:
            x, y = next(train_iterator)

        except StopIteration:

            train_iterator = iter(
                train_loader
            )

            x, y = next(
                train_iterator
            )

        x = x.to(device)
        y = y.to(device)

        # ----------------------------------------------------
        # Learning rate
        # ----------------------------------------------------

        lr = get_lr(step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits, loss = model(
            x,
            y,
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Progress bar
        # ----------------------------------------------------

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{lr:.2e}",
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if (
            step % EVAL_INTERVAL == 0
            or step == MAX_STEPS - 1
        ):

            losses = evaluate(
                model,
                train_loader,
                val_loader,
                device,
            )

            train_loss = losses["train"]
            val_loss = losses["val"]

            print(
                f"\nstep {step:5d} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | "
                f"lr {lr:.2e}"
            )

            # Save best model
            if val_loss < best_val_loss:

                best_val_loss = val_loss

                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    train_loss,
                    val_loss,
                    config,
                    f"{CHECKPOINT_DIR}/best.pt",
                )

                print(
                    f"Saved best checkpoint "
                    f"(val loss: {val_loss:.4f})"
                )

            # Periodic checkpoint
            if step > 0 and step % 500 == 0:

                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    train_loss,
                    val_loss,
                    config,
                    f"{CHECKPOINT_DIR}/step_{step}.pt",
                )

                print(
                    f"Saved checkpoint "
                    f"step_{step}.pt"
                )

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    save_checkpoint(
        model,
        optimizer,
        MAX_STEPS,
        loss.item(),
        best_val_loss,
        config,
        f"{CHECKPOINT_DIR}/final.pt",
    )

    print("\nTraining complete.")

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train()