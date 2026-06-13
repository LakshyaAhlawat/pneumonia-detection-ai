"""
run_training.py — Entry point: trains ResNet50, EfficientNet-B0, and ViT-B/16.

Usage:
    python run_training.py                    # train all models
    python run_training.py --model resnet50   # train one model
    python run_training.py --epochs 20        # override epoch count

After training, checkpoints are saved to checkpoints/<model>_best.pt
Training curves are saved to results/plots/<model>_training_curves.png
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.config import MODELS, NUM_EPOCHS, DATA_DIR
from src.dataset import get_dataloaders
from src.models import build_model
from src.train import train_model
from src.evaluate import plot_training_curves
from src.utils import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train pneumonia detection models")
    parser.add_argument(
        "--model",
        choices=MODELS + ["all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    set_seed()

    models_to_train = MODELS if args.model == "all" else [args.model]

    print(f"\nDataset: {args.data_dir}")
    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_weighted_sampler=not args.no_weighted_sampler,
    )

    # Print dataset stats
    for split, loader in dataloaders.items():
        print(f"  {split}: {len(loader.dataset)} images")

    for model_name in models_to_train:
        print(f"\n{'#'*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'#'*60}")

        model   = build_model(model_name, freeze_backbone=True)
        history = train_model(
            model       = model,
            dataloaders = dataloaders,
            model_name  = model_name,
            num_epochs  = args.epochs,
            device      = device,
        )
        plot_training_curves(history, model_name)

    print("\n✓ Training complete. Run run_evaluation.py to see results.")


if __name__ == "__main__":
    main()
