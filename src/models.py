"""
models.py — Model builders for ResNet50, EfficientNet-B0, and ViT-B/16.

All models:
  1. Load ImageNet pretrained weights
  2. Optionally freeze the backbone (Phase 1 training)
  3. Replace the classification head with a custom binary head
  4. Expose a `unfreeze_last_stage()` method for Phase 2 fine-tuning

Parameter counts (approx):
  ResNet50        → 25.6M total | ~2K trainable (frozen head only)
  EfficientNet-B0 → 5.3M  total | ~1.3K trainable (frozen head only)
  ViT-B/16        → 86.6M total | ~1.6K trainable (frozen head only)
"""

import torch
import torch.nn as nn
import torchvision.models as models

from src.config import NUM_CLASSES


# ── ResNet50 ───────────────────────────────────────────────────────────────────

class ResNet50Classifier(nn.Module):
    """
    ResNet50 with a custom binary classification head.

    Architecture:
        Backbone (frozen in Phase 1):
            Conv1 → BN → ReLU → MaxPool
            Layer1 (3 blocks) → Layer2 (4 blocks) → Layer3 (6 blocks) → Layer4 (3 blocks)
            GlobalAvgPool → 2048-dim feature vector
        Head (always trainable):
            Dropout(0.5) → Linear(2048 → num_classes)

    Skip connections: each residual block adds input x to F(x) before activation,
    solving the vanishing gradient problem in deep networks.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)

        # Separate backbone from classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # up to AvgPool
        in_features = backbone.fc.in_features  # 2048

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def unfreeze_last_stage(self):
        """Unfreeze Layer4 (last residual stage) for Phase 2 fine-tuning."""
        # backbone children: [0]=conv1 [1]=bn1 [2]=relu [3]=maxpool
        #                     [4]=layer1 [5]=layer2 [6]=layer3 [7]=layer4 [8]=avgpool
        for i, child in enumerate(self.backbone.children()):
            if i >= 7:  # layer4 and avgpool
                for param in child.parameters():
                    param.requires_grad = True

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── EfficientNet-B0 ────────────────────────────────────────────────────────────

class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 with a custom binary classification head.

    Architecture:
        Features (MBConv blocks, frozen in Phase 1):
            Stem Conv → 7 MBConv stages → Head Conv → AdaptiveAvgPool
            → 1280-dim feature vector
        Head (always trainable):
            Dropout(0.2) → Linear(1280 → num_classes)

    MBConv = Mobile Inverted Bottleneck + Squeeze-and-Excitation (channel attention).
    Compound scaling: B0 is the baseline; B1–B7 scale depth/width/resolution together.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)

        self.features = backbone.features       # MBConv feature extractor
        self.avgpool  = backbone.avgpool        # AdaptiveAvgPool2d(1,1)
        in_features   = backbone.classifier[1].in_features  # 1280

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)

    def unfreeze_last_stage(self):
        """Unfreeze the last 2 MBConv stages (indices 6, 7, 8 in features)."""
        children = list(self.features.children())
        for child in children[6:]:   # last 3 blocks + head conv
            for param in child.parameters():
                param.requires_grad = True

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── Vision Transformer ─────────────────────────────────────────────────────────

class ViTB16Classifier(nn.Module):
    """
    Vision Transformer ViT-B/16 with a custom binary classification head.

    Architecture:
        Patch embedding: 224×224 image → 14×14 = 196 patches of 16×16px
            Each patch flattened to 768-dim, projected via linear embedding
        Positional embeddings (learned, not sinusoidal) added to patch embeddings
        [CLS] token prepended: 196 + 1 = 197 total tokens
        12× Transformer encoder blocks:
            LayerNorm → Multi-Head Self-Attention (12 heads, 64-dim each)
            Residual → LayerNorm → MLP (768 → 3072 → 768)
            Residual
        [CLS] token final state → 768-dim representation → classification head
        Head (always trainable):
            Dropout(0.5) → Linear(768 → num_classes)

    Note: ViT lacks CNN inductive biases (locality, translation equivariance).
    On small datasets it typically underperforms CNNs — expected and documented.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_backbone: bool = True):
        super().__init__()
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        vit = models.vit_b_16(weights=weights)

        # Keep everything except the original head
        self.patch_embedding    = vit.conv_proj
        self.class_token        = vit.class_token
        self.seq_length         = vit.seq_length
        self.encoder            = vit.encoder
        self.pos_embedding_flag = True
        self._vit               = vit            # keep reference for forward

        in_features = vit.heads.head.in_features  # 768

        if freeze_backbone:
            # Freeze everything except the new classifier head
            for name, param in vit.named_parameters():
                param.requires_grad = False

        self._vit.heads = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes),
        )
        # Ensure head params are trainable
        for param in self._vit.heads.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._vit(x)

    def unfreeze_last_stage(self):
        """Unfreeze the last 4 Transformer encoder blocks for fine-tuning."""
        encoder_layers = list(self._vit.encoder.layers.children())
        for layer in encoder_layers[-4:]:
            for param in layer.parameters():
                param.requires_grad = True

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── Factory function ───────────────────────────────────────────────────────────

def build_model(name: str, freeze_backbone: bool = True) -> nn.Module:
    """
    Convenience factory.

    Args:
        name: One of 'resnet50', 'efficientnet_b0', 'vit_b16'
        freeze_backbone: True for Phase 1 (head-only training)

    Returns:
        Initialized model ready for training.
    """
    builders = {
        "resnet50":        ResNet50Classifier,
        "efficientnet_b0": EfficientNetB0Classifier,
        "vit_b16":         ViTB16Classifier,
    }
    if name not in builders:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(builders)}")
    return builders[name](freeze_backbone=freeze_backbone)
