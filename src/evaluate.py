"""
evaluate.py — Full evaluation pipeline.

Computes:
  Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix,
  Inference time per image, Parameter count.

All plots saved to results/plots/.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)
from typing import Tuple

import sys
sys.path.insert(0, os.path.dirname(__file__))
from src.config import CLASS_NAMES, PLOTS_DIR


# ── Core evaluation function ───────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs inference on the test set and computes all metrics.

    Returns:
        metrics (dict): accuracy, precision, recall, f1, roc_auc,
                        inference_time_ms, total_params, trainable_params
        preds   (np.ndarray): predicted class indices
        labels  (np.ndarray): true class indices
        probs   (np.ndarray): predicted probability for class 1 (Pneumonia)
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    inference_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)

            t0      = time.time()
            logits  = model(images)
            elapsed = time.time() - t0
            inference_times.append(elapsed / images.size(0))  # per-image

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    probs  = np.array(all_probs)

    metrics = {
        "accuracy":           accuracy_score(labels, preds),
        "precision":          precision_score(labels, preds, zero_division=0),
        "recall":             recall_score(labels, preds, zero_division=0),
        "f1":                 f1_score(labels, preds, zero_division=0),
        "roc_auc":            roc_auc_score(labels, probs),
        "inference_time_ms":  float(np.mean(inference_times) * 1000),
        "total_params":       sum(p.numel() for p in model.parameters()),
        "trainable_params":   sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    return metrics, preds, labels, probs


# ── Plot helpers ───────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, model_name: str) -> None:
    """Save loss and accuracy training curves."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=14, y=1.02)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train loss", color="#2563EB")
    ax1.plot(epochs, history["val_loss"],   label="Val loss",   color="#DC2626", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train acc", color="#16A34A")
    ax2.plot(epochs, history["val_acc"],   label="Val acc",   color="#EA580C", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{model_name}_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, model_name: str) -> None:
    """Save confusion matrix plot."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(results: dict) -> None:
    """
    Plot ROC curves for all models on one axis.

    Args:
        results: {model_name: {"labels": ..., "probs": ..., "metrics": ...}}
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    colors = {"resnet50": "#2563EB", "efficientnet_b0": "#16A34A", "vit_b16": "#EA580C"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data["labels"], data["probs"])
        auc = data["metrics"]["roc_auc"]
        color = colors.get(name, "gray")
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "roc_curves_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_comparison_table(all_metrics: dict) -> None:
    """
    Bar chart comparing key metrics across all models.

    Args:
        all_metrics: {model_name: metrics_dict}
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels       = list(all_metrics.keys())
    colors       = ["#2563EB", "#16A34A", "#EA580C"]
    x            = np.arange(len(metric_names))
    width        = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (model_name, metrics) in enumerate(all_metrics.items()):
        values = [metrics[m] for m in metric_names]
        bars   = ax.bar(x + i * width, values, width, label=model_name, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper().replace("_", " ") for m in metric_names])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "model_comparison_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_metrics_table(all_metrics: dict) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"{'Model':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'ms/img':>8} {'Params(M)':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, m in all_metrics.items():
        print(
            f"{name:<20} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
            f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['roc_auc']:>6.4f} "
            f"{m['inference_time_ms']:>8.2f} {m['total_params']/1e6:>10.1f}M"
        )
    print("=" * len(header))
