# model/checkpoints.py
# ---------------------------------------------------------------
# Saving and loading model checkpoints.
#
# Two kinds of saves:
#   1. Per-epoch checkpoint  — saves everything needed to resume
#      training if it crashes (model + optimizer + scheduler +
#      loss history).
#   2. Best-model snapshot  — saves only the model weights
#      whenever validation loss improves.  This is what gets
#      loaded for inference and export.
# ---------------------------------------------------------------
import os
import glob

import torch
import torch.nn as nn

from config import CHECKPOINT_DIR, DEVICE, make_output_dirs


# Fixed filename for the best model — easy to reference later
BEST_MODEL_FILENAME = "best_model.pt"


def best_model_path() -> str:
    """Return the full path to the best-model weights file."""
    return os.path.join(CHECKPOINT_DIR, BEST_MODEL_FILENAME)


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_losses: list,
    val_losses: list,
    best_val: float,
) -> None:
    """Save a full training checkpoint for a given epoch.

    The checkpoint contains everything needed to resume training
    exactly where it left off — model weights, optimizer state,
    scheduler state, and the full loss history.

    Files are named  ckpt_epoch001.pt, ckpt_epoch002.pt, …
    Old checkpoints are NOT deleted automatically; clean up
    CHECKPOINT_DIR manually if disk space is a concern.

    Parameters
    ----------
    epoch        : current epoch number (1-based)
    model        : the nn.Module being trained
    optimizer    : the optimizer
    scheduler    : the LR scheduler
    train_losses : list of per-epoch training MAE values so far
    val_losses   : list of per-epoch validation MAE values so far
    best_val     : best validation MAE seen so far
    """
    make_output_dirs()
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch{epoch:03d}.pt")
    torch.save(
        {
            "epoch":            epoch,
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "train_losses":     train_losses,
            "val_losses":       val_losses,
            "best_val":         best_val,
        },
        path,
    )


def save_best_model(model: nn.Module) -> None:
    """Save only the model weights to the best-model file.

    Called whenever a new best validation MAE is achieved.
    Overwrites the previous best silently.
    """
    make_output_dirs()
    torch.save(model.state_dict(), best_model_path())


def load_latest_checkpoint(model: nn.Module, optimizer, scheduler):
    """Resume from the most recent per-epoch checkpoint if one exists.

    Searches CHECKPOINT_DIR for files matching ckpt_epoch*.pt and
    loads the latest one (sorted lexicographically, which works
    because filenames are zero-padded to 3 digits).

    Parameters
    ----------
    model, optimizer, scheduler : objects whose state will be
        restored in-place

    Returns
    -------
    start_epoch  : epoch to resume from (0 if no checkpoint found)
    train_losses : list of training MAE values up to that epoch
    val_losses   : list of validation MAE values up to that epoch
    best_val     : best validation MAE seen so far
    """
    pattern = os.path.join(CHECKPOINT_DIR, "ckpt_epoch*.pt")
    ckpts   = sorted(glob.glob(pattern))

    if not ckpts:
        print("No checkpoint found — starting from scratch.")
        return 0, [], [], float("inf")

    ck = torch.load(ckpts[-1], map_location=DEVICE)
    model.load_state_dict(ck["model_state"])
    optimizer.load_state_dict(ck["optimizer_state"])
    scheduler.load_state_dict(ck["scheduler_state"])

    epoch = ck["epoch"]
    print(f"Resumed from checkpoint: epoch {epoch}")
    return epoch, ck["train_losses"], ck["val_losses"], ck["best_val"]


def load_best_model(model: nn.Module) -> nn.Module:
    """Load the best-model weights into *model* and return it.

    Used before evaluation, inference, and export.

    Parameters
    ----------
    model : an nn.Module with the same architecture as when the
            weights were saved (i.e. built with build_model())

    Returns
    -------
    The same *model* object with weights loaded, moved to DEVICE,
    and set to eval mode.
    """
    path = best_model_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Best model not found at '{path}'.  "
            "Train the model first (training/train.py)."
        )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Loaded best model from: {path}")
    return model