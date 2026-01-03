"""
Instance Grasp Prediction Head.

Predicts grasp parameters for each ROI (detected object instance).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceGraspHead(nn.Module):
    """
    Instance-level grasp prediction head.

    Takes ROI features and predicts 5-channel grasp maps for each instance.
    """

    def __init__(
        self,
        in_channels: int = 1536,
        hidden_channels: int = 256,
        roi_size: Tuple[int, int] = (30, 40),
        output_size: Tuple[int, int] = (120, 160),
        num_output_channels: int = 5,
    ):
        """
        Initialize instance grasp head.

        Args:
            in_channels: Input feature channels (e.g., 1536 for Qwen2-VL ViT)
            hidden_channels: Hidden layer channels
            roi_size: Expected ROI input size (H, W)
            output_size: Output prediction size (H, W)
            num_output_channels: Number of output channels (default 5)
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.roi_size = roi_size
        self.output_size = output_size

        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # FPN-style decoder with upsampling
        self.decoder = nn.Sequential(
            # Stage 1: roi_size -> roi_size * 2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # Stage 2: roi_size * 2 -> roi_size * 4
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),

            # Output head
            nn.Conv2d(hidden_channels // 2, num_output_channels, 1),
        )

    def forward(
        self,
        roi_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            roi_features: ROI features [N, C, roi_H, roi_W]

        Returns:
            Dict with 5-channel predictions:
            - pred_pos: [N, 1, H, W] position heatmap
            - pred_cos: [N, 1, H, W] cos(2*theta)
            - pred_sin: [N, 1, H, W] sin(2*theta)
            - pred_width: [N, 1, H, W] normalized width
            - pred_semantic: [N, 1, H, W] semantic mask
        """
        N = roi_features.shape[0]

        if N == 0:
            H, W = self.output_size
            device = roi_features.device
            dtype = roi_features.dtype
            return {
                'pred_pos': torch.zeros((0, 1, H, W), device=device, dtype=dtype),
                'pred_cos': torch.zeros((0, 1, H, W), device=device, dtype=dtype),
                'pred_sin': torch.zeros((0, 1, H, W), device=device, dtype=dtype),
                'pred_width': torch.zeros((0, 1, H, W), device=device, dtype=dtype),
                'pred_semantic': torch.zeros((0, 1, H, W), device=device, dtype=dtype),
            }

        # Project input
        x = self.input_proj(roi_features)

        # Decode
        pred = self.decoder(x)

        # Resize to output size if needed
        if pred.shape[-2:] != self.output_size:
            pred = F.interpolate(
                pred,
                size=self.output_size,
                mode='bilinear',
                align_corners=False,
            )

        return {
            'pred_pos': pred[:, 0:1],
            'pred_cos': pred[:, 1:2],
            'pred_sin': pred[:, 2:3],
            'pred_width': pred[:, 3:4],
            'pred_semantic': pred[:, 4:5],
        }

    def decode_grasps(
        self,
        predictions: Dict[str, torch.Tensor],
        threshold: float = 0.5,
        top_k: int = 1,
    ) -> List[List[Dict]]:
        """
        Decode grasp predictions for each ROI.

        Args:
            predictions: Dict with predicted maps
            threshold: Confidence threshold
            top_k: Number of top grasps to return per ROI

        Returns:
            List of grasp lists, one per ROI
        """
        pred_pos = torch.sigmoid(predictions['pred_pos'])
        pred_cos = torch.tanh(predictions['pred_cos'])
        pred_sin = torch.tanh(predictions['pred_sin'])
        pred_width = torch.sigmoid(predictions['pred_width'])

        N = pred_pos.shape[0]
        results = []

        for i in range(N):
            pos_map = pred_pos[i, 0].cpu().numpy()

            # Find top-k positions
            if top_k == 1:
                max_idx = np.unravel_index(pos_map.argmax(), pos_map.shape)
                candidates = [max_idx]
            else:
                flat = pos_map.flatten()
                top_indices = np.argpartition(flat, -top_k)[-top_k:]
                candidates = [np.unravel_index(idx, pos_map.shape) for idx in top_indices]
                candidates = sorted(candidates, key=lambda p: pos_map[p], reverse=True)

            grasps = []
            for y, x in candidates:
                conf = pos_map[y, x]
                if conf < threshold:
                    continue

                cos_val = pred_cos[i, 0, y, x].item()
                sin_val = pred_sin[i, 0, y, x].item()
                width_val = pred_width[i, 0, y, x].item()

                # Decode angle
                angle = 0.5 * np.arctan2(sin_val, cos_val)

                grasps.append({
                    'x': int(x),
                    'y': int(y),
                    'angle': float(angle),
                    'width': float(width_val * 150),  # De-normalize
                    'confidence': float(conf),
                })

            results.append(grasps)

        return results

    def decode_best_grasp(
        self,
        predictions: Dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> List[Optional[Dict]]:
        """
        Decode the best grasp for each ROI.

        Args:
            predictions: Dict with predicted maps
            threshold: Confidence threshold

        Returns:
            List of best grasps (or None if below threshold)
        """
        grasps_list = self.decode_grasps(predictions, threshold, top_k=1)
        return [grasps[0] if grasps else None for grasps in grasps_list]


class LightweightInstanceGraspHead(nn.Module):
    """
    Lightweight instance grasp head for faster inference.

    Uses depthwise separable convolutions and fewer parameters.
    """

    def __init__(
        self,
        in_channels: int = 1536,
        hidden_channels: int = 128,
        roi_size: Tuple[int, int] = (30, 40),
        output_size: Tuple[int, int] = (120, 160),
    ):
        super().__init__()

        self.roi_size = roi_size
        self.output_size = output_size

        # Depthwise separable projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Lightweight decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, 3, padding=1, groups=256),
            nn.Conv2d(256, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 5, 1),
        )

    def forward(self, roi_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x = self.input_proj(roi_features)
        pred = self.decoder(x)

        if pred.shape[-2:] != self.output_size:
            pred = F.interpolate(pred, size=self.output_size, mode='bilinear', align_corners=False)

        return {
            'pred_pos': pred[:, 0:1],
            'pred_cos': pred[:, 1:2],
            'pred_sin': pred[:, 2:3],
            'pred_width': pred[:, 3:4],
            'pred_semantic': pred[:, 4:5],
        }


class InstanceGraspLoss(nn.Module):
    """
    Loss function for instance-level grasp prediction.
    """

    def __init__(
        self,
        heatmap_weight: float = 1.0,
        angle_weight: float = 1.0,
        width_weight: float = 1.0,
        semantic_weight: float = 0.5,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
    ):
        super().__init__()

        self.heatmap_weight = heatmap_weight
        self.angle_weight = angle_weight
        self.width_weight = width_weight
        self.semantic_weight = semantic_weight
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta

    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pos_thresh: float = 0.5,
    ) -> torch.Tensor:
        """
        Modified focal loss for Gaussian heatmap targets.

        Uses FP32 computation to avoid overflow.
        """
        # Force FP32 for numerical stability
        pred = pred.float()
        target = target.float()

        eps = 1e-6
        target = torch.clamp(target, 0, 1)
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, eps, 1 - eps)

        pos_mask = (target > pos_thresh).float()
        neg_mask = (target <= pos_thresh).float()

        # Positive loss
        pos_loss = -torch.log(pred) * torch.pow(1 - pred, self.focal_alpha) * pos_mask

        # Negative loss with penalty reduction near positives
        neg_weight = torch.pow(1 - target, self.focal_beta)
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, self.focal_alpha) * neg_weight * neg_mask

        num_pos = pos_mask.sum()
        if num_pos == 0:
            return neg_loss.sum() / (neg_mask.sum() + eps)
        return (pos_loss.sum() + neg_loss.sum()) / (num_pos + eps)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses.

        Args:
            predictions: Dict with pred_pos, pred_cos, pred_sin, pred_width, pred_semantic
            targets: Dict with gt_pos, gt_cos, gt_sin, gt_width, gt_semantic

        Returns:
            Dict with individual losses and total loss
        """
        # Position loss (focal loss)
        loss_pos = self.focal_loss(predictions['pred_pos'], targets['gt_pos'])

        # Get positive mask for angle/width losses
        pos_mask = (targets['gt_pos'] > 0.5).float()
        num_pos = pos_mask.sum() + 1e-6

        # Angle loss (L1 on cos and sin)
        pred_cos = torch.tanh(predictions['pred_cos'])
        pred_sin = torch.tanh(predictions['pred_sin'])
        loss_cos = (torch.abs(pred_cos - targets['gt_cos']) * pos_mask).sum() / num_pos
        loss_sin = (torch.abs(pred_sin - targets['gt_sin']) * pos_mask).sum() / num_pos
        loss_angle = loss_cos + loss_sin

        # Width loss (smooth L1)
        pred_width = torch.sigmoid(predictions['pred_width'])
        loss_width = F.smooth_l1_loss(
            pred_width * pos_mask,
            targets['gt_width'] * pos_mask,
            reduction='sum'
        ) / num_pos

        # Semantic loss (BCE)
        loss_semantic = F.binary_cross_entropy_with_logits(
            predictions['pred_semantic'],
            targets['gt_semantic'],
        )

        # Total loss
        total_loss = (
            self.heatmap_weight * loss_pos +
            self.angle_weight * loss_angle +
            self.width_weight * loss_width +
            self.semantic_weight * loss_semantic
        )

        return {
            'loss_total': total_loss,
            'loss_pos': loss_pos,
            'loss_angle': loss_angle,
            'loss_width': loss_width,
            'loss_semantic': loss_semantic,
        }
