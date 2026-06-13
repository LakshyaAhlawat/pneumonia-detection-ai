"""
config.py — Central configuration for Pneumonia Detection project.
Change values here; nothing else needs to be edited.
"""

import os
import kagglehub

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
PLOTS_DIR      = os.path.join(RESULTS_DIR, "plots")

print("Checking/Downloading dataset via kagglehub...")
_kaggle_path   = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
DATA_DIR       = os.path.join(_kaggle_path, "chest_xray")

# ── Dataset ────────────────────────────────────────────────────────────────────
CLASS_NAMES    = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES    = 2
IMAGE_SIZE     = 224        # pixels (both W and H after resize)

# ImageNet statistics — required because all models are pretrained on ImageNet
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
NUM_EPOCHS     = 30
SEED           = 42
NUM_WORKERS    = 4          # DataLoader worker processes

# Phase 1: train only the classifier head (backbone frozen)
LR_HEAD        = 1e-3
WEIGHT_DECAY   = 1e-4

# Phase 2: fine-tune last backbone stage at a much lower LR
LR_FINETUNE    = 1e-4
FINETUNE_EPOCH = 10         # switch to fine-tuning after this many epochs

# ── Early Stopping ─────────────────────────────────────────────────────────────
ES_PATIENCE    = 7
ES_DELTA       = 0.001

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS = ["resnet50", "efficientnet_b0", "vit_b16"]

# ── Class weights for imbalanced dataset (~74% Pneumonia / ~26% Normal) ────────
# Inverse frequency: Normal gets higher weight
CLASS_WEIGHTS  = [2.84, 1.0]   # [Normal, Pneumonia]
