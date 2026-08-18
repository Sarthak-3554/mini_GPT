import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Linear layer with a LoRA adapter.

    y = W(x) + scaling * B(A(x))

    W is frozen.
    A and B are trainable.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not isinstance(
            original_layer,
            nn.Linear,
        ):
            raise TypeError(
                "LoRALinear requires nn.Linear"
            )

        self.in_features = (
            original_layer.in_features
        )

        self.out_features = (
            original_layer.out_features
        )

        self.rank = rank
        self.alpha = alpha

        self.scaling = (
            alpha / rank
        )

        # ----------------------------------------------------
        # Original pretrained layer
        # ----------------------------------------------------

        self.original = original_layer

        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False

        # ----------------------------------------------------
        # LoRA A
        #
        # in_features → rank
        # ----------------------------------------------------

        self.lora_A = nn.Linear(
            self.in_features,
            rank,
            bias=False,
        )

        # ----------------------------------------------------
        # LoRA B
        #
        # rank → out_features
        # ----------------------------------------------------

        self.lora_B = nn.Linear(
            rank,
            self.out_features,
            bias=False,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # LoRA initialization
        #
        # A = random
        # B = zero
        #
        # Therefore initially:
        #
        # LoRA(x) = 0
        #
        # so replacing Linear with LoRA does not
        # change the pretrained model's output.
        # ----------------------------------------------------

        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=math.sqrt(5),
        )

        nn.init.zeros_(
            self.lora_B.weight
        )

    def forward(self, x):

        base_output = self.original(x)

        lora_output = self.lora_B(
            self.lora_A(
                self.dropout(x)
            )
        )

        return (
            base_output
            + self.scaling * lora_output
        )


def replace_linear_with_lora(
    module,
    rank=8,
    alpha=16,
    dropout=0.0,
    target_names=None,
):
    """
    Recursively replace selected nn.Linear
    layers with LoRALinear.

    target_names controls which layers receive LoRA.
    """

    if target_names is None:
        target_names = [
            "q_proj",
            "v_proj",
        ]

    for name, child in list(
        module.named_children()
    ):

        # ----------------------------------------------------
        # Replace target Linear layers
        # ----------------------------------------------------

        if (
            isinstance(child, nn.Linear)
            and name in target_names
        ):

            setattr(
                module,
                name,
                LoRALinear(
                    child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )

        else:

            # Recursively search
            replace_linear_with_lora(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_names=target_names,
            )


def mark_only_lora_trainable(
    model,
):

    # First freeze everything.
    for param in model.parameters():
        param.requires_grad = False

    # Then enable LoRA parameters.
    for name, param in model.named_parameters():

        if (
            "lora_A" in name
            or "lora_B" in name
        ):
            param.requires_grad = True


def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    percentage = (
        100.0 * trainable / total
    )

    return (
        total,
        trainable,
        percentage,
    )