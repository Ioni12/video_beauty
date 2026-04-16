# training/train.py
# ---------------------------------------------------------------
# Training loop, single-epoch runner, and test-set evaluation.
#
# Public entry points:
#   train()    — full training run (calls everything else)
#   evaluate() — test-set MAE + Pearson r after training
# ---------------------------------------------------------------
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from config import DEVICE, EPOCHS, PATIENCE, EXPORT_DIR, make_output_dirs
from model import (
    build_model,
    count_parameters,
    save_checkpoint,
    save_best_model,
    load_latest_checkpoint,
    load_best_model,
)


# ── Optimizer / scheduler factory ───────────────────────────────
# Kept here rather than in model/ because these are training-time
# concerns only — inference and export never need them.

def build_optimizer(model: nn.Module):
    """Return (criterion, optimizer, scheduler) for *model*."""
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    return criterion, optimizer, scheduler


# ── Single epoch ─────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    training: bool = True,
) -> float:
    """Run one full pass over *loader* and return mean MAE.

    Parameters
    ----------
    model     : the nn.Module
    loader    : a DataLoader (train or val/test)
    criterion : loss function (L1Loss)
    optimizer : only used when training=True
    training  : if True, runs backward pass and optimizer step

    Returns
    -------
    Mean absolute error over the full dataset split (float)
    """
    model.train() if training else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for imgs, scores in loader:
            imgs   = imgs.to(DEVICE, non_blocking=True)
            scores = scores.to(DEVICE, non_blocking=True).unsqueeze(1)

            preds = model(imgs)
            loss  = criterion(preds, scores)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


# ── Full training run ────────────────────────────────────────────

def train(train_loader, val_loader) -> nn.Module:
    """Train the model and return the best checkpoint loaded model.

    Automatically resumes from the latest checkpoint if one exists
    in CHECKPOINT_DIR — so if Colab disconnects mid-training, just
    re-run and it picks up where it left off.

    Parameters
    ----------
    train_loader, val_loader : DataLoaders from data.make_dataloaders()

    Returns
    -------
    model with best weights loaded, on DEVICE, in eval mode
    """
    make_output_dirs()

    model                    = build_model().to(DEVICE)
    criterion, optimizer, scheduler = build_optimizer(model)

    print(f"Device            : {DEVICE}")
    print(f"Trainable params  : {count_parameters(model):,}")

    # Resume if a checkpoint exists
    start_epoch, train_losses, val_losses, best_val = load_latest_checkpoint(
        model, optimizer, scheduler
    )

    no_improve = 0

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        train_mae = run_epoch(model, train_loader, criterion, optimizer, training=True)
        val_mae   = run_epoch(model, val_loader,   criterion, optimizer, training=False)
        scheduler.step(val_mae)

        train_losses.append(train_mae)
        val_losses.append(val_mae)
        save_checkpoint(epoch, model, optimizer, scheduler,
                        train_losses, val_losses, best_val)

        if val_mae < best_val:
            best_val   = val_mae
            no_improve = 0
            save_best_model(model)
            tag = "  ← best"
        else:
            no_improve += 1
            tag = f"  (no improve {no_improve}/{PATIENCE})"

        print(
            f"Epoch {epoch:03d}/{EPOCHS}  |  "
            f"Train MAE: {train_mae:.4f}  |  "
            f"Val MAE: {val_mae:.4f}{tag}"
        )

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nBest Val MAE: {best_val:.4f}")
    return load_best_model(model)


# ── Test-set evaluation ──────────────────────────────────────────

def evaluate(model: nn.Module, test_loader) -> dict:
    """Evaluate *model* on *test_loader* and save diagnostic plots.

    Parameters
    ----------
    model       : model with best weights already loaded
    test_loader : DataLoader for the held-out test split

    Returns
    -------
    dict with keys: mae, pearson_r, pearson_p, preds, targets
    """
    make_output_dirs()
    model.eval()

    preds_all, targets_all = [], []
    with torch.no_grad():
        for imgs, scores in test_loader:
            p = model(imgs.to(DEVICE)).squeeze(1).cpu().numpy()
            preds_all.extend(p.tolist())
            targets_all.extend(scores.numpy().tolist())

    preds_all   = np.array(preds_all)
    targets_all = np.array(targets_all)
    mae         = float(np.mean(np.abs(preds_all - targets_all)))
    r, pv       = pearsonr(preds_all, targets_all)

    print(f"Test MAE: {mae:.4f}  |  Pearson r: {r:.4f}  (p={pv:.2e})")

    _plot_loss_curves_from_checkpoint()
    _plot_scatter(preds_all, targets_all, mae, r)
    
    return dict(mae=mae, pearson_r=r, pearson_p=pv,
                preds=preds_all, targets=targets_all)


# ── Plot helpers ─────────────────────────────────────────────────

def _plot_loss_curves_from_checkpoint() -> None:
    """Load the latest checkpoint and plot train/val MAE curves."""
    import glob
    from config import CHECKPOINT_DIR
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_epoch*.pt")))
    if not ckpts:
        print("No checkpoint found — skipping loss curve plot.")
        return
    ck           = torch.load(ckpts[-1], map_location="cpu")
    train_losses = ck["train_losses"]
    val_losses   = ck["val_losses"]

    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train MAE", color="royalblue", lw=2)
    ax.plot(epochs, val_losses,   label="Val MAE",   color="tomato",    lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE")
    ax.set_title("Training and Validation MAE")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()

    out = os.path.join(EXPORT_DIR, "loss_curves.png")
    fig.savefig(out, dpi=150)
    plt.show()
    print(f"Loss curve saved → {out}")


def _plot_scatter(preds, targets, mae: float, r: float) -> None:
    """Scatter plot: predicted vs ground-truth scores."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets, preds, alpha=0.4, s=15, color="steelblue")
    ax.plot([1, 5], [1, 5], "r--", lw=1.5)
    ax.set_xlabel("Ground Truth"); ax.set_ylabel("Predicted")
    ax.set_title(f"Test set  (MAE={mae:.3f}, r={r:.3f})")
    fig.tight_layout()

    out = os.path.join(EXPORT_DIR, "test_scatter.png")
    fig.savefig(out, dpi=150)
    plt.show()
    print(f"Scatter plot saved → {out}")