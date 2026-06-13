"""
dataset.py — Dataset class, transforms, and DataLoader factory.

Key decisions documented here:
- X-rays are grayscale but converted to RGB (3-channel) because all pretrained
  models expect 3-channel input. The single channel is replicated across R/G/B.
- Augmentation is applied ONLY to the train split — never val or test.
- ImageNet normalization is used because all models are ImageNet-pretrained.
"""

import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from collections import Counter
from typing import Optional

from src.config import (
    CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, NUM_WORKERS, DATA_DIR
)


# ── Transform factory ──────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Returns the appropriate torchvision transform pipeline for the given split.

    Train: augmentation (flip, rotate, color jitter) + normalize
    Val/Test: deterministic resize + normalize ONLY
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),          # Slightly larger than target
            transforms.RandomCrop(IMAGE_SIZE),       # Random 224×224 crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ── Dataset class ──────────────────────────────────────────────────────────────

class ChestXRayDataset(Dataset):
    """
    Loads chest X-ray images from the standard Kaggle directory layout:

        data/chest_xray/
            train/
                NORMAL/   *.jpeg
                PNEUMONIA/ *.jpeg
            val/
                ...
            test/
                ...

    Labels: NORMAL → 0, PNEUMONIA → 1
    """

    def __init__(self, root_dir: str, split: str):
        """
        Args:
            root_dir: Path to the chest_xray root directory.
            split: One of 'train', 'val', 'test'.
        """
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.transform = get_transforms(split)
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        self.samples = []   # list of (image_path, label) tuples

        for cls_name, label in self.class_to_idx.items():
            cls_dir = os.path.join(root_dir, split, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(
                    f"Class directory not found: {cls_dir}\n"
                    f"Make sure you downloaded the Kaggle dataset to {root_dir}"
                )
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir}/{split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        # X-rays are single-channel; convert to RGB for 3-channel model input
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)

    def class_counts(self) -> dict:
        """Returns {class_name: count} for analysis."""
        labels = [s[1] for s in self.samples]
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        return {idx_to_class[k]: v for k, v in Counter(labels).items()}


# ── DataLoader factory ─────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    use_weighted_sampler: bool = True,
) -> dict:
    """
    Returns a dict of DataLoaders for train, val, and test splits.

    Args:
        data_dir: Root path to chest_xray dataset. Defaults to config.DATA_DIR.
        batch_size: Batch size for all splits.
        use_weighted_sampler: If True, use WeightedRandomSampler on train set to
            counteract class imbalance (~74% PNEUMONIA).

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    if data_dir is None:
        data_dir = DATA_DIR

    loaders = {}
    for split in ("train", "val", "test"):
        dataset = ChestXRayDataset(data_dir, split)

        if split == "train" and use_weighted_sampler:
            # Compute per-sample weights: rare class gets higher weight
            counts = dataset.class_counts()
            total = sum(counts.values())
            class_weights = {
                dataset.class_to_idx[cls]: total / cnt
                for cls, cnt in counts.items()
            }
            sample_weights = torch.tensor(
                [class_weights[label] for _, label in dataset.samples],
                dtype=torch.float,
            )
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,        # shuffle=False when using sampler
                num_workers=NUM_WORKERS,
                pin_memory=True,
            )
        else:
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
            )

    return loaders
