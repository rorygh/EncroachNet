"""Losses for 2D powerline/vegetation segmentation.

See docs/architecture.md Stage 1 for the math. Two problems are handled:
  1. Severe class imbalance (powerline pixels are ~1-5% of any image) -> focal + Dice.
  2. Thin-wire connectivity (per-pixel IoU tolerates broken wires) -> skeleton-based
     topology term (Skeleton Recall Loss, cheaper alternative to soft-clDice).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import IGNORE_INDEX, NUM_CLASSES


class FocalLoss(nn.Module):
    """Per-class-weighted focal loss (Focal Phi Loss family, see docs/sota.md Part A)."""

    def __init__(self, alpha: list[float], gamma: float = 2.0, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), target: (B, H, W) int class indices
        valid = target != self.ignore_index
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()

        target_safe = target.clamp(min=0)
        pt = p.gather(1, target_safe.unsqueeze(1)).squeeze(1)
        logpt = logp.gather(1, target_safe.unsqueeze(1)).squeeze(1)
        alpha_t = self.alpha[target_safe]

        loss = -alpha_t * (1 - pt).pow(self.gamma) * logpt
        loss = loss[valid]
        return loss.mean() if loss.numel() > 0 else logits.sum() * 0.0


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss, ignore_index-aware."""

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = (target != self.ignore_index).unsqueeze(1).float()
        probs = F.softmax(logits, dim=1)
        target_safe = target.clamp(min=0)
        target_onehot = F.one_hot(target_safe, self.num_classes).permute(0, 3, 1, 2).float()

        probs = probs * valid
        target_onehot = target_onehot * valid

        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        union = probs.sum(dims) + target_onehot.sum(dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """3x3 min-pool, used to build a differentiable soft skeleton (Shit et al., soft-clDice)."""
    p1 = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Iterative morphological soft-skeleton approximation (differentiable)."""
    skel = F.relu(x - _soft_open(x))
    body = x
    for _ in range(iterations):
        body = _soft_erode(body)
        opened = _soft_open(body)
        delta = F.relu(body - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SkeletonTopologyLoss(nn.Module):
    """Skeleton-Recall-style topology loss for a single thin-structure class
    (default: powerline). See docs/architecture.md Stage 1 and docs/sota.md Part A.
    """

    def __init__(self, class_index: int, skel_iterations: int = 10, eps: float = 1e-6):
        super().__init__()
        self.class_index = class_index
        self.skel_iterations = skel_iterations
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)[:, self.class_index : self.class_index + 1]
        target_mask = (target == self.class_index).float().unsqueeze(1)

        if target_mask.sum() == 0:
            return logits.sum() * 0.0

        pred_skel = soft_skeletonize(probs, self.skel_iterations)
        gt_skel = soft_skeletonize(target_mask, self.skel_iterations)

        t_prec = (pred_skel * target_mask).sum() / (pred_skel.sum() + self.eps)
        t_sens = (gt_skel * probs).sum() / (gt_skel.sum() + self.eps)
        cl_dice = 2 * t_prec * t_sens / (t_prec + t_sens + self.eps)
        return 1.0 - cl_dice


class CombinedSegmentationLoss(nn.Module):
    """L_2D = lambda_focal * L_focal + lambda_dice * L_dice + lambda_topo * L_topo(powerline)."""

    def __init__(self, class_names: list[str], focal_alpha: list[float], focal_gamma: float,
                 dice_weight: float, topo_weight: float, topo_class: str,
                 num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.focal = FocalLoss(focal_alpha, focal_gamma, ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index)
        self.dice_weight = dice_weight
        self.topo_weight = topo_weight
        self.topo = SkeletonTopologyLoss(class_names.index(topo_class))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        l_focal = self.focal(logits, target)
        l_dice = self.dice(logits, target)
        l_topo = self.topo(logits, target)
        total = l_focal + self.dice_weight * l_dice + self.topo_weight * l_topo
        return {"loss": total, "focal": l_focal, "dice": l_dice, "topo": l_topo}
