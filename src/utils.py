"""
utils.py — Shared utilities: reproducibility, device setup, logging.
"""

import os
import random
import numpy as np
import torch
import json
from datetime import datetime

from src.config import SEED, CHECKPOINT_DIR, RESULTS_DIR


def set_seed(seed: int = SEED) -> None:
    """
    Fix all random seeds for reproducibility.
    Call this at the top of every training script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA ops (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    """Returns CUDA if available, else CPU. Prints device info."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  Running on CPU (training will be slow)")
    return device


def load_checkpoint(model, model_name: str, device: torch.device):
    """
    Load best checkpoint for a model.

    Returns:
        model with loaded weights, checkpoint metadata dict.
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train the model first using run_training.py"
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded {model_name} checkpoint (val_acc={ckpt['val_acc']:.4f}, epoch={ckpt['epoch']})")
    return model, ckpt


def save_results(results: dict, filename: str = "all_results.json") -> None:
    """Save evaluation results dict to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    # Convert numpy floats to Python floats for JSON serialization
    serializable = {}
    for model_name, data in results.items():
        serializable[model_name] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in data["metrics"].items()
        }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results saved to {path}")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
