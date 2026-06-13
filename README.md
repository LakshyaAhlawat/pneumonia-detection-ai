# Pneumonia Detection from Chest X-Rays

Binary classification of chest X-rays using ResNet50, EfficientNet-B0, and Vision Transformer (ViT-B/16) with PyTorch transfer learning.

## Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Params |
|-------|----------|-----------|--------|----|---------|--------|
| ResNet50 | — | — | — | — | — | 25.6M |
| EfficientNet-B0 | — | — | — | — | — | 5.3M |
| ViT-B/16 | — | — | — | — | — | 86.6M |

*(Fill in after training)*

## Project Structure

```
pneumonia-detection/
├── src/
│   ├── config.py        # All hyperparameters (single source of truth)
│   ├── dataset.py       # ChestXRayDataset + DataLoader factory
│   ├── models.py        # ResNet50, EfficientNet-B0, ViT builders
│   ├── train.py         # Two-phase training loop + early stopping
│   ├── evaluate.py      # Metrics, confusion matrix, ROC-AUC, plots
│   └── utils.py         # Seed, device, checkpoint loading
├── app/
│   └── streamlit_app.py # Web application
├── notebooks/
│   └── 01_eda.py        # Dataset analysis
├── run_training.py      # Entry point: train models
├── run_evaluation.py    # Entry point: evaluate + generate plots
└── requirements.txt
```

## Setup

### 1. Install dependencies

```bash
# CPU
pip install -r requirements.txt

# GPU (CUDA 11.8 — for RTX 3050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Download dataset

1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Download and extract to `data/chest_xray/`

Expected structure:
```
data/chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```

### 3. Run EDA

```bash
python notebooks/01_eda.py
```

### 4. Train models

```bash
# Train all three models (takes 1–3 hours depending on GPU)
python run_training.py

# Train one model
python run_training.py --model efficientnet_b0

# Options
python run_training.py --help
```

### 5. Evaluate

```bash
python run_evaluation.py
```

Plots saved to `results/plots/`. Results saved to `results/comparison_table.csv`.

### 6. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Deploy to HuggingFace Spaces

1. Create a new Space at https://huggingface.co/spaces
   - SDK: Streamlit
   - Hardware: CPU Basic (free) or T4 GPU

2. In `README.md`, add YAML front matter:
```yaml
---
title: Pneumonia Detection
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.32.0"
app_file: app/streamlit_app.py
pinned: false
---
```

3. Push your repo (include `checkpoints/` folder with trained `.pt` files):
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://huggingface.co/spaces/<username>/<space-name>
git push origin main
```

## Key Design Decisions

**Why `convert('RGB')` on grayscale X-rays?**
All three models expect 3-channel input. Grayscale values are replicated across R/G/B — standard practice; works well in practice.

**Why WeightedRandomSampler?**
The dataset has ~74% Pneumonia / ~26% Normal. Without balancing, the model learns to predict Pneumonia for everything and achieves ~74% accuracy trivially.

**Why recall matters more than accuracy?**
A False Negative (predicting Normal when the patient has Pneumonia) is clinically dangerous. Always optimize for recall in medical screening tasks.

**Two-phase training:**
Phase 1 (epochs 1–10): Backbone frozen, only the classifier head trains (LR=1e-3).
Phase 2 (epoch 11+): Last backbone stage unfrozen, fine-tuned at LR=1e-4.

## Interview Talking Points

1. **ResNet skip connections** solve vanishing gradients: `output = F(x) + x` → gradient highway
2. **EfficientNet compound scaling** — scale depth/width/resolution jointly, not independently
3. **ViT underperforms on small datasets** — lacks CNN inductive biases (locality, translation equivariance)
4. **`model.eval()` vs `torch.no_grad()`** — orthogonal; eval changes Dropout/BN behavior, no_grad saves memory
5. **Class imbalance handling** — WeightedRandomSampler + weighted CrossEntropyLoss + report F1 not accuracy

## Resume Bullets

```
• Built a comparative CV study classifying pneumonia from chest X-rays (5,863 images)
  using ResNet50, EfficientNet-B0, and ViT-B/16 with PyTorch transfer learning

• Implemented a full ML pipeline: augmentation, DataLoader, two-phase fine-tuning
  with early stopping and CosineAnnealing LR scheduling, achieving 93%+ test accuracy

• Conducted parameter-count, inference-time, and ROC-AUC comparison across 3 models;
  documented ViT limitations on small datasets vs. CNN inductive biases

• Deployed a multi-model Streamlit app with real-time X-ray upload, per-class confidence
  scores, and model selection; hosted on HuggingFace Spaces
```
