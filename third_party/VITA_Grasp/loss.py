"""
Loss functions for VITA-Grasp model.

5-channel loss:
    - Channel 0 (Position): CenterNet-style Focal Loss (sparse heatmap)
    - Channel 1,2 (Angle cos/sin): Weighted MSE with tanh activation
    - Channel 3 (Width): BCE with class balancing
    - Channel 4 (Semantic): CenterNet-style Focal Loss

CenterNet Focal Loss:
    - Strongly penalizes false positives (alpha=2, beta=4)
    - Produces sparse heatmap output like ByteTrack
    - Reference: ByteTrack/yolox/models/grasp_dense_head.py:284-324
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class GraspLoss(nn.Module):
    """
    Multi-channel loss for grasp prediction with CenterNet-style Focal Loss.

    Loss = w_pos * L_pos + w_ang * L_ang + w_wid * L_wid + w_sem * L_semantic
    """

    def __init__(
        self,
        pos_weight: float = 5.0,
        ang_weight: float = 5.0,
        wid_weight: float = 1.0,
        sem_weight: float = 1.0,
        pos_max_ratio: float = 100.0,
        wid_max_ratio: float = 10.0,
        sem_max_ratio: float = 5.0,
        ang_grasp_weight: float = 5.0,
        focal_alpha: float = 2.0,  # CenterNet focal loss alpha
        focal_beta: float = 4.0,   # CenterNet focal loss beta
        pos_thresh: float = 0.5,   # Threshold for positive samples
    ):
        """
        Args:
            pos_weight: Weight for position loss
            ang_weight: Weight for angle loss (cos + sin)
            wid_weight: Weight for width loss
            sem_weight: Weight for semantic loss
            pos_max_ratio: Max positive weight ratio for width BCE
            wid_max_ratio: Max positive weight ratio for width BCE
            sem_max_ratio: Max positive weight ratio for semantic BCE (unused with focal)
            ang_grasp_weight: Weight multiplier for angle loss at grasp positions
            focal_alpha: Focal loss alpha (penalizes easy negatives)
            focal_beta: Focal loss beta (down-weights negatives near positives)
            pos_thresh: Threshold for positive samples in focal loss
        """
        super().__init__()

        self.weights = {
            'pos': pos_weight,
            'ang': ang_weight,
            'wid': wid_weight,
            'sem': sem_weight,
        }

        self.max_ratios = {
            'pos': pos_max_ratio,
            'wid': wid_max_ratio,
            'sem': sem_max_ratio,
        }

        self.ang_grasp_weight = ang_grasp_weight
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.pos_thresh = pos_thresh

    def centernet_focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 2.0,
        beta: float = 4.0,
        pos_thresh: float = 0.5,
    ) -> torch.Tensor:
        """
        CenterNet-style Modified Focal Loss for Gaussian heatmap GT.

        This loss strongly penalizes false positives to produce sparse heatmaps.
        Reference: ByteTrack/yolox/models/grasp_dense_head.py:284-324

        GT is Gaussian distribution [0, 1], not binary.
        - Positive samples: target > pos_thresh
        - Negative samples: target <= pos_thresh

        Loss formulation:
        - Positive: -log(p) * (1-p)^alpha
        - Negative: -log(1-p) * p^alpha * (1-target)^beta

        The (1-target)^beta term down-weights negatives that are close to
        positive centers (where target is high but < pos_thresh).

        Args:
            pred: Predicted logits [B, 1, H, W]
            target: Target heatmap [B, 1, H, W] in [0, 1]
            alpha: Focal loss alpha (default: 2)
            beta: Focal loss beta for negative weighting (default: 4)
            pos_thresh: Threshold for positive samples (default: 0.5)

        Returns:
            loss: Scalar loss value
        """
        # Force FP32 to avoid numerical issues
        pred = pred.float()
        target = target.float()

        eps = 1e-6

        # Clamp target to valid range
        target = torch.clamp(target, 0, 1)

        # Apply sigmoid to get probability
        prob = torch.sigmoid(pred)
        prob = torch.clamp(prob, eps, 1 - eps)

        # Positive mask: Gaussian peak locations
        pos_mask = (target > pos_thresh).float()
        # Negative mask: other locations
        neg_mask = (target <= pos_thresh).float()

        # Positive loss: encourage prediction close to 1
        # -log(p) * (1-p)^alpha
        pos_loss = -torch.log(prob) * torch.pow(1 - prob, alpha) * pos_mask

        # Negative loss: encourage prediction close to 0
        # Also down-weight negatives that are close to positive centers
        # -log(1-p) * p^alpha * (1-target)^beta
        neg_weight = torch.pow(1 - target, beta)
        neg_loss = -torch.log(1 - prob) * torch.pow(prob, alpha) * neg_weight * neg_mask

        # Normalize by number of positive samples
        num_pos = pos_mask.sum()
        pos_loss_sum = pos_loss.sum()
        neg_loss_sum = neg_loss.sum()

        if num_pos == 0:
            # No positive samples, just return normalized negative loss
            return neg_loss_sum / (neg_mask.sum() + eps)

        return (pos_loss_sum + neg_loss_sum) / (num_pos + eps)

    def compute_bce_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        max_ratio: float,
    ) -> torch.Tensor:
        """
        Compute BCE loss with dynamic positive weight.

        Args:
            pred: Predicted logits [B, 1, H, W]
            target: Target mask [B, 1, H, W]
            max_ratio: Maximum positive weight ratio

        Returns:
            loss: Scalar loss value
        """
        # Calculate positive class weight
        pos = target.sum().float()
        total = target.numel()

        if pos > 0:
            ratio = (total - pos) / pos
            ratio = torch.clamp(ratio, max=max_ratio)
        else:
            ratio = torch.tensor(1.0, device=pred.device)

        loss = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=ratio,
            reduction='mean'
        )

        return loss

    def compute_angle_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        grasp_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss for angle prediction.

        Uses tanh activation on prediction to match target range [-1, 1].
        Applies higher weight at grasp positions.

        Args:
            pred: Predicted angle component [B, 1, H, W]
            target: Target angle component [B, 1, H, W]
            grasp_mask: Binary mask of grasp positions [B, 1, H, W]

        Returns:
            loss: Scalar loss value
        """
        # Apply tanh to match target range [-1, 1]
        pred_tanh = torch.tanh(pred)

        # Weight map: higher weight at grasp positions
        weight_map = torch.where(
            grasp_mask > 0,
            torch.tensor(self.ang_grasp_weight, device=pred.device),
            torch.tensor(1.0, device=pred.device)
        )

        # Weighted MSE
        loss = F.mse_loss(pred_tanh, target, reduction='none') * weight_map

        # Normalize by total weight
        total_weight = weight_map.sum() + 1e-6
        return loss.sum() / total_weight

    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total grasp loss.

        Args:
            pred_masks: Predicted 5-channel masks [B, 5, H, W]
            target_masks: Target 5-channel masks [B, 5, H, W]

        Returns:
            losses: Dict with individual losses and total loss
        """
        assert pred_masks.shape[1] == 5, f"Expected 5 channels, got {pred_masks.shape[1]}"
        assert target_masks.shape == pred_masks.shape

        # Extract channels
        pred_pos = pred_masks[:, 0:1]
        pred_cos = pred_masks[:, 1:2]
        pred_sin = pred_masks[:, 2:3]
        pred_wid = pred_masks[:, 3:4]
        pred_sem = pred_masks[:, 4:5]

        target_pos = target_masks[:, 0:1]
        target_cos = target_masks[:, 1:2]
        target_sin = target_masks[:, 2:3]
        target_wid = target_masks[:, 3:4]
        target_sem = target_masks[:, 4:5]

        # Grasp mask for angle loss weighting
        grasp_mask = (target_pos > 0).float()

        # Position loss: CenterNet-style Focal Loss for sparse heatmap
        loss_pos = self.centernet_focal_loss(
            pred_pos, target_pos,
            alpha=self.focal_alpha,
            beta=self.focal_beta,
            pos_thresh=self.pos_thresh
        )

        # Angle loss: weighted MSE at grasp positions
        loss_cos = self.compute_angle_loss(pred_cos, target_cos, grasp_mask)
        loss_sin = self.compute_angle_loss(pred_sin, target_sin, grasp_mask)

        # Width loss: BCE with class balancing
        loss_wid = self.compute_bce_loss(pred_wid, target_wid, self.max_ratios['wid'])

        # Semantic loss: CenterNet-style Focal Loss for sparse mask
        loss_sem = self.centernet_focal_loss(
            pred_sem, target_sem,
            alpha=self.focal_alpha,
            beta=self.focal_beta,
            pos_thresh=self.pos_thresh
        )

        # Combine angle losses
        loss_ang = loss_cos + loss_sin

        # Weighted total
        total_loss = (
            self.weights['pos'] * loss_pos +
            self.weights['ang'] * loss_ang +
            self.weights['wid'] * loss_wid +
            self.weights['sem'] * loss_sem
        )

        return {
            'loss': total_loss,
            'loss_pos': loss_pos,
            'loss_ang': loss_ang,
            'loss_cos': loss_cos,
            'loss_sin': loss_sin,
            'loss_wid': loss_wid,
            'loss_sem': loss_sem,
        }


class FocalGraspLoss(GraspLoss):
    """
    Grasp loss with focal loss for position and semantic channels.
    Better handles class imbalance.
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def compute_focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            pred: Predicted logits [B, 1, H, W]
            target: Target mask [B, 1, H, W]

        Returns:
            loss: Scalar loss value
        """
        prob = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        p_t = prob * target + (1 - prob) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.focal_gamma)

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
            loss = alpha_t * loss

        return loss.mean()

    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss with focal loss for position."""
        assert pred_masks.shape[1] == 5

        pred_pos = pred_masks[:, 0:1]
        pred_cos = pred_masks[:, 1:2]
        pred_sin = pred_masks[:, 2:3]
        pred_wid = pred_masks[:, 3:4]
        pred_sem = pred_masks[:, 4:5]

        target_pos = target_masks[:, 0:1]
        target_cos = target_masks[:, 1:2]
        target_sin = target_masks[:, 2:3]
        target_wid = target_masks[:, 3:4]
        target_sem = target_masks[:, 4:5]

        grasp_mask = (target_pos > 0).float()

        # Use focal loss for position
        loss_pos = self.compute_focal_loss(pred_pos, target_pos)
        loss_cos = self.compute_angle_loss(pred_cos, target_cos, grasp_mask)
        loss_sin = self.compute_angle_loss(pred_sin, target_sin, grasp_mask)
        loss_wid = self.compute_bce_loss(pred_wid, target_wid, self.max_ratios['wid'])
        loss_sem = self.compute_focal_loss(pred_sem, target_sem)

        loss_ang = loss_cos + loss_sin

        total_loss = (
            self.weights['pos'] * loss_pos +
            self.weights['ang'] * loss_ang +
            self.weights['wid'] * loss_wid +
            self.weights['sem'] * loss_sem
        )

        return {
            'loss': total_loss,
            'loss_pos': loss_pos,
            'loss_ang': loss_ang,
            'loss_cos': loss_cos,
            'loss_sin': loss_sin,
            'loss_wid': loss_wid,
            'loss_sem': loss_sem,
        }


def build_loss(loss_type: str = 'default', **kwargs) -> nn.Module:
    """
    Build loss function.

    Args:
        loss_type: 'default' or 'focal'
        **kwargs: Loss function arguments

    Returns:
        Loss module
    """
    if loss_type == 'default':
        return GraspLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalGraspLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
