"""
run_evaluation.py — Load best checkpoints, compute all metrics, generate all plots.

Usage:
    python run_evaluation.py
    python run_evaluation.py --model resnet50   # evaluate one model only

Outputs:
    results/plots/<model>_confusion_matrix.png
    results/plots/<model>_training_curves.png   (from training)
    results/plots/roc_curves_comparison.png
    results/plots/model_comparison_metrics.png
    results/all_results.json
"""

import argparse
import sys
import os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.config import MODELS, DATA_DIR, RESULTS_DIR
from src.dataset import get_dataloaders
from src.models import build_model
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_comparison_table,
    print_metrics_table,
)
from src.utils import set_seed, get_device, load_checkpoint, save_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    set_seed()

    models_to_eval = MODELS if args.model == "all" else [args.model]

    print(f"\nLoading test data from: {args.data_dir}")
    dataloaders = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    test_loader = dataloaders["test"]
    print(f"  Test set: {len(test_loader.dataset)} images\n")

    all_results  = {}
    all_metrics  = {}

    for model_name in models_to_eval:
        print(f"{'─'*50}")
        print(f"  Evaluating: {model_name}")
        print(f"{'─'*50}")

        model = build_model(model_name, freeze_backbone=False)
        model, _ = load_checkpoint(model, model_name, device)
        model = model.to(device)

        metrics, preds, labels, probs = evaluate_model(model, test_loader, device)

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}  ← prioritize for medical task")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Inference: {metrics['inference_time_ms']:.2f} ms/image")
        print(f"  Params:    {metrics['total_params']/1e6:.1f}M\n")

        plot_confusion_matrix(labels, preds, model_name)

        all_results[model_name] = {"metrics": metrics, "labels": labels, "probs": probs}
        all_metrics[model_name] = metrics

    # ── Cross-model comparison plots ───────────────────────────────────────────
    if len(models_to_eval) > 1:
        plot_roc_curves(all_results)
        plot_comparison_table(all_metrics)
        print_metrics_table(all_metrics)

    save_results(all_results)

    # ── Save CSV comparison table ──────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "model":            name,
            "accuracy":         round(m["accuracy"], 4),
            "precision":        round(m["precision"], 4),
            "recall":           round(m["recall"], 4),
            "f1":               round(m["f1"], 4),
            "roc_auc":          round(m["roc_auc"], 4),
            "inference_ms":     round(m["inference_time_ms"], 2),
            "params_M":         round(m["total_params"] / 1e6, 1),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Comparison table saved to {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
