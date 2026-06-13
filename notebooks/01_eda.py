"""
01_eda.py — Dataset Exploratory Data Analysis.

Run as a script or convert to notebook:
    jupyter nbconvert --to notebook --execute 01_eda.py

Produces:
    - Class distribution bar chart
    - Sample images grid (Normal vs Pneumonia)
    - Image size distribution histogram
    - Pixel intensity distribution
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import random

from src.config import DATA_DIR, CLASS_NAMES, PLOTS_DIR
from src.utils import set_seed

set_seed()
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── 1. Class distribution ──────────────────────────────────────────────────────
print("\n=== Class Distribution ===")
counts = defaultdict(dict)
for split in ("train", "val", "test"):
    for cls in CLASS_NAMES:
        path = os.path.join(DATA_DIR, split, cls)
        n = len(os.listdir(path)) if os.path.exists(path) else 0
        counts[split][cls] = n
    total = sum(counts[split].values())
    print(f"  {split:5s}: {dict(counts[split])}  | total={total}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, split in enumerate(("train", "val", "test")):
    labels = list(counts[split].keys())
    values = list(counts[split].values())
    bars   = axes[i].bar(labels, values, color=["#16A34A", "#DC2626"], alpha=0.85)
    for bar, val in zip(bars, values):
        axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(val), ha="center", fontsize=10)
    axes[i].set_title(split.capitalize())
    axes[i].set_ylabel("Image count")
    axes[i].set_ylim(0, max(values) * 1.15)

plt.suptitle("Class Distribution per Split", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: class_distribution.png")

# ── 2. Sample images grid ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for row, cls in enumerate(CLASS_NAMES):
    cls_dir = os.path.join(DATA_DIR, "train", cls)
    if not os.path.exists(cls_dir):
        continue
    files = random.sample(os.listdir(cls_dir), min(5, len(os.listdir(cls_dir))))
    for col, fname in enumerate(files):
        img = Image.open(os.path.join(cls_dir, fname)).convert("L")
        axes[row][col].imshow(img, cmap="gray")
        axes[row][col].axis("off")
        if col == 0:
            axes[row][col].set_title(cls, fontsize=11, loc="left")

plt.suptitle("Sample Images (grayscale, before RGB conversion)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sample_images.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: sample_images.png")

# ── 3. Image size distribution ────────────────────────────────────────────────
print("\n=== Image Size Distribution (train, 200 samples) ===")
widths, heights = [], []
cls_dir = os.path.join(DATA_DIR, "train", "PNEUMONIA")
if os.path.exists(cls_dir):
    files = random.sample(os.listdir(cls_dir), min(200, len(os.listdir(cls_dir))))
    for fname in files:
        img = Image.open(os.path.join(cls_dir, fname))
        w, h = img.size
        widths.append(w)
        heights.append(h)

    print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
    print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(widths,  bins=20, color="#2563EB", alpha=0.8, edgecolor="white")
    ax1.set_title("Width distribution"); ax1.set_xlabel("pixels")
    ax2.hist(heights, bins=20, color="#EA580C", alpha=0.8, edgecolor="white")
    ax2.set_title("Height distribution"); ax2.set_xlabel("pixels")
    plt.suptitle("Image Size Distribution (train/PNEUMONIA, 200 samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "image_size_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: image_size_distribution.png")

print("\n✓ EDA complete. Plots saved to", PLOTS_DIR)
