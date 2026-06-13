"""
train.py — Full training pipeline with two-phase fine-tuning.

Phase 1 (epochs 1 → FINETUNE_EPOCH):
    Backbone frozen, only the classifier head trains.
    High LR (1e-3), fast convergence.

Phase 2 (epoch FINETUNE_EPOCH+1 → end):
    Last backbone stage unfrozen, both head and last stage train.
    Low LR (1e-4), careful fine-tuning.

Early stopping monitors val_loss with patience=7.
Best checkpoint saved by val_accuracy.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Tuple

import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.config import (
    NUM_EPOCHS, LR_HEAD, LR_FINETUNE, FINETUNE_EPOCH,
    ES_PATIENCE, ES_DELTA, CHECKPOINT_DIR, CLASS_WEIGHTS,
)


# ── Early Stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation loss hasn't improved for `patience` epochs.

    Args:
        patience: How many epochs to wait after last improvement.
        delta: Minimum change to qualify as improvement.
    """
    def __init__(self, patience: int = ES_PATIENCE, delta: float = ES_DELTA):
        self.patience   = patience
        self.delta      = delta
        self.best_score = None
        self.counter    = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter    = 0


# ── Single epoch helpers ───────────────────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    """Run one epoch; return (avg_loss, accuracy)."""
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss   = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


# ── Main train function ────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    dataloaders: dict,
    model_name: str,
    num_epochs: int = NUM_EPOCHS,
    device: torch.device = None,
) -> dict:
    """
    Full two-phase training loop.

    Returns:
        history dict with keys:
            train_loss, val_loss, train_acc, val_acc, lr_per_epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model = model.to(device)

    # Weighted cross-entropy with label smoothing to combat overconfidence
    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    # Phase 1 optimizer — only trains the classifier head
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    early_stopping = EarlyStopping()

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr_per_epoch": [],
    }
    best_val_acc = 0.0
    ckpt_path    = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")

    print(f"\n{'='*60}")
    print(f"  Training {model_name}  |  device: {device}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {params:,}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        # ── Phase 2 switch ─────────────────────────────────────────────────────
        if epoch == FINETUNE_EPOCH + 1:
            print(f"\n  [Phase 2] Unfreezing last backbone stage at epoch {epoch}")
            model.unfreeze_last_stage()
            # Re-create optimizer with lower LR for all newly unfrozen params
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_FINETUNE,
                weight_decay=1e-4,
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=num_epochs - epoch, eta_min=1e-7
            )
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable parameters now: {params:,}\n")

        t0 = time.time()
        train_loss, train_acc = _run_epoch(
            model, dataloaders["train"], optimizer, criterion, device, train=True
        )
        val_loss, val_acc = _run_epoch(
            model, dataloaders["val"], optimizer, criterion, device, train=False
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr_per_epoch"].append(current_lr)

        phase = 1 if epoch <= FINETUNE_EPOCH else 2
        print(
            f"  Epoch {epoch:02d}/{num_epochs} [P{phase}] | "
            f"Loss {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc {train_acc:.4f}/{val_acc:.4f} | "
            f"LR {current_lr:.2e} | {elapsed:.1f}s"
        )

        # ── Checkpoint best model ──────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_acc":     val_acc,
                    "val_loss":    val_loss,
                    "model_name":  model_name,
                },
                ckpt_path,
            )
            print(f"  ✓ Checkpoint saved (val_acc={val_acc:.4f})")

        # ── Early stopping ─────────────────────────────────────────────────────
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break

    print(f"\n  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoint: {ckpt_path}\n")
    return history
