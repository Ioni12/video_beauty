# model/architecture.py
# ---------------------------------------------------------------
# Defines the beauty scoring model.
#
# Base: MobileNetV3-Small pretrained on ImageNet.
# Head: replaced with a small regression head that outputs a
#       single float in the range [1, 5].
#
# MobileNetV3-Small is chosen deliberately — it is fast enough
# to run in real-time on a phone (the eventual deployment target)
# while still being accurate enough for this task.
# ---------------------------------------------------------------
import torch.nn as nn
from torchvision import models


def build_model() -> nn.Module:
    """Build and return the beauty scoring model.

    Loads MobileNetV3-Small with ImageNet weights, then replaces
    the classifier with a regression head:

        Linear(in_features → 256)
        Hardswish
        Dropout(0.2)
        Linear(256 → 1)

    The final output is a raw float (not clipped to [1,5]) so the
    loss function sees the full gradient signal during training.
    Clipping to [1,5] is applied at inference time only.

    Returns
    -------
    nn.Module  (not yet moved to a device — caller does .to(DEVICE))
    """
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )

    in_features = model.classifier[0].in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
    )

    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)