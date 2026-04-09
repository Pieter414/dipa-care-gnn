"""
    Loss functions for fraud detection GNN experiments.
    
    Seven loss variants designed for class-imbalanced fraud graphs:
    1. CrossEntropy (baseline)
    2. Weighted CrossEntropy (inverse class frequency)
    3. Focal Loss (hard example mining)
    4. Label Smoothing CrossEntropy
    5. Class-Balanced Loss (effective number of samples)
    6. Dice Loss (F1-oriented)
    7. CARE Composite Loss (GNN + similarity, original paper Eq.11)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedCrossEntropy(nn.Module):
    """CrossEntropy with inverse class-frequency weights."""

    def __init__(self, labels: np.ndarray):
        super().__init__()
        counts = np.bincount(labels.astype(int))
        # inverse frequency, normalized so weights sum to num_classes
        weights = len(labels) / (len(counts) * counts.astype(float) + 1e-8)
        self.register_buffer("weight", torch.FloatTensor(weights))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets.squeeze(), weight=self.weight)


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).
    Down-weights easy examples so the model focuses on hard, misclassified ones.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.squeeze()
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of the true class
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


class LabelSmoothingCE(nn.Module):
    """CrossEntropy with label smoothing to reduce overconfidence."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.squeeze()
        return F.cross_entropy(logits, targets, label_smoothing=self.smoothing)


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss (Cui et al., CVPR 2019).
    Uses effective number of samples: E_n = (1 - beta^n) / (1 - beta).
    """

    def __init__(self, labels: np.ndarray, beta: float = 0.9999):
        super().__init__()
        counts = np.bincount(labels.astype(int)).astype(float)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        # normalize so weights sum to num_classes
        weights = weights / weights.sum() * len(weights)
        self.register_buffer("weight", torch.FloatTensor(weights))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets.squeeze(), weight=self.weight)


class DiceLoss(nn.Module):
    """
    Soft Dice Loss — directly optimizes the F1 / Dice coefficient.
    Particularly useful for imbalanced binary classification.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.squeeze()
        probs = F.softmax(logits, dim=1)[:, 1]  # P(fraud)
        targets_f = targets.float()
        intersection = (probs * targets_f).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets_f.sum() + self.smooth
        )
        return 1.0 - dice


class CARECompositeLoss(nn.Module):
    """
    Original CARE-GNN composite: GNN_loss + lambda_1 * simi_loss (Eq.11).
    Only meaningful when the model produces both gnn_scores and label_scores.
    Falls back to plain CE for models that don't output label_scores.
    """

    def __init__(self, lambda_1: float = 2.0):
        super().__init__()
        self.lambda_1 = lambda_1
        self.xent = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                label_scores: torch.Tensor = None) -> torch.Tensor:
        targets = targets.squeeze()
        gnn_loss = self.xent(logits, targets)
        if label_scores is not None:
            label_loss = self.xent(label_scores, targets)
            return gnn_loss + self.lambda_1 * label_loss
        return gnn_loss


# ── Registry ────────────────────────────────────────────────────────────────

LOSS_REGISTRY = {
    "ce":             lambda labels: nn.CrossEntropyLoss(),
    "weighted_ce":    lambda labels: WeightedCrossEntropy(labels),
    "focal":          lambda labels: FocalLoss(),
    "label_smooth":   lambda labels: LabelSmoothingCE(),
    "class_balanced": lambda labels: ClassBalancedLoss(labels),
    "dice":           lambda labels: DiceLoss(),
    "care_composite": lambda labels: CARECompositeLoss(),
}


def build_loss(name: str, labels: np.ndarray) -> nn.Module:
    """Construct a loss function by name. `labels` is the full training label array."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](labels)