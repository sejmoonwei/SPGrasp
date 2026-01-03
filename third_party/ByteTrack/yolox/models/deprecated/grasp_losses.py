#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Grasp Parameter Loss Functions for ByteTrack-Grasp

Loss components:
1. Position loss: SmoothL1 on grasp center offset (dx, dy)
2. Angle loss: L1 on cos(2*theta) and sin(2*theta) + unit circle constraint
3. Width loss: Smooth L1
4. Score loss: BCE (optional)

Strategies for handling multiple valid grasps:
1. Single-target: Compare to weighted average of top-k grasps (default)
2. Min-distance: Compare to closest valid grasp (more tolerant)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspLoss(nn.Module):
    """
    6-DOF Grasp Parameter Loss Function.

    Input format:
    - pred: (N, 6) = [grasp_dx, grasp_dy, cos2theta, sin2theta, grasp_width, grasp_score]
    - target: (N, 6) = [grasp_dx, grasp_dy, cos2theta, sin2theta, grasp_width, grasp_score]

    grasp_dx, grasp_dy: normalized offset from bbox center to grasp center
        grasp_center_x = bbox_cx + grasp_dx * bbox_w
        grasp_center_y = bbox_cy + grasp_dy * bbox_h

    Loss = position_weight * L_pos + angle_weight * L_angle + width_weight * L_width + score_weight * L_score
    """

    def __init__(
        self,
        reduction="none",
        position_weight=1.0,
        angle_weight=1.0,
        width_weight=1.0,
        score_weight=0.5,
        unit_circle_weight=0.1,
    ):
        """
        Args:
            reduction: "none", "mean", or "sum"
            position_weight: Weight for grasp center position loss
            angle_weight: Weight for angle loss
            width_weight: Weight for width loss
            score_weight: Weight for score loss
            unit_circle_weight: Weight for unit circle regularization
        """
        super().__init__()
        self.reduction = reduction
        self.position_weight = position_weight
        self.angle_weight = angle_weight
        self.width_weight = width_weight
        self.score_weight = score_weight
        self.unit_circle_weight = unit_circle_weight

        self.l1_loss = nn.L1Loss(reduction="none")
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target):
        """
        Compute grasp loss.

        Args:
            pred: (N, 6) = [grasp_dx, grasp_dy, cos2theta, sin2theta, grasp_width, grasp_score]
            target: (N, 6) = [grasp_dx, grasp_dy, cos2theta, sin2theta, grasp_width, grasp_score]

        Returns:
            loss: (N,) if reduction="none", scalar otherwise
        """
        if pred.shape[0] == 0:
            if self.reduction == "none":
                return pred.new_zeros(0)
            return pred.new_tensor(0.0)

        # Split components
        pred_dx = pred[:, 0]
        pred_dy = pred[:, 1]
        pred_cos = pred[:, 2]
        pred_sin = pred[:, 3]
        pred_width = pred[:, 4]
        pred_score = pred[:, 5]

        target_dx = target[:, 0]
        target_dy = target[:, 1]
        target_cos = target[:, 2]
        target_sin = target[:, 3]
        target_width = target[:, 4]
        target_score = target[:, 5]

        # 1. Position loss (Smooth L1 on dx, dy)
        loss_dx = self.smooth_l1(pred_dx, target_dx)
        loss_dy = self.smooth_l1(pred_dy, target_dy)
        loss_position = loss_dx + loss_dy

        # 2. Angle loss (L1 on cos/sin)
        loss_cos = self.l1_loss(pred_cos, target_cos)
        loss_sin = self.l1_loss(pred_sin, target_sin)

        # Unit circle regularization: |cos^2 + sin^2 - 1|
        # This encourages the network to output valid angle encodings
        unit_circle_reg = torch.abs(pred_cos**2 + pred_sin**2 - 1)

        loss_angle = loss_cos + loss_sin + self.unit_circle_weight * unit_circle_reg

        # 3. Width loss (Smooth L1)
        loss_width = self.smooth_l1(pred_width, target_width)

        # 4. Score loss (BCE, only for valid targets)
        # Only compute score loss where target score > 0 (valid grasp)
        valid_mask = target_score > 0
        if valid_mask.any():
            loss_score = torch.zeros_like(pred_score)
            loss_score[valid_mask] = self.bce_loss(
                pred_score[valid_mask],
                target_score[valid_mask]
            )
        else:
            loss_score = torch.zeros_like(pred_score)

        # Combine losses
        total_loss = (
            self.position_weight * loss_position +
            self.angle_weight * loss_angle +
            self.width_weight * loss_width +
            self.score_weight * loss_score
        )

        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        else:
            return total_loss


class AngleLoss(nn.Module):
    """
    Specialized loss for grasp angle using cos/sin encoding.

    Uses L2 distance between encoded angle vectors to handle angle wrapping.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_encoded, target_encoded):
        """
        Compute angle loss.

        Args:
            pred_encoded: (N, 2) = [cos2theta, sin2theta]
            target_encoded: (N, 2) = [cos2theta, sin2theta]

        Returns:
            loss: scalar or (N,) depending on reduction
        """
        if pred_encoded.shape[0] == 0:
            if self.reduction == "none":
                return pred_encoded.new_zeros(0)
            return pred_encoded.new_tensor(0.0)

        # L2 distance between encoded vectors
        diff = pred_encoded - target_encoded
        loss = torch.norm(diff, dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GraspPositionLoss(nn.Module):
    """
    Loss for grasp position offset (dx, dy) relative to bbox center.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def forward(self, pred_offset, target_offset):
        """
        Compute position offset loss.

        Args:
            pred_offset: (N, 2) = [dx, dy]
            target_offset: (N, 2) = [dx, dy]

        Returns:
            loss: scalar or (N,) depending on reduction
        """
        if pred_offset.shape[0] == 0:
            if self.reduction == "none":
                return pred_offset.new_zeros(0)
            return pred_offset.new_tensor(0.0)

        loss = self.smooth_l1(pred_offset, target_offset).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedGraspLoss(nn.Module):
    """
    Combined loss for all grasp parameters with configurable weights.
    """

    def __init__(
        self,
        angle_weight=1.0,
        width_weight=1.0,
        score_weight=0.5,
        reduction="mean",
    ):
        super().__init__()
        self.angle_weight = angle_weight
        self.width_weight = width_weight
        self.score_weight = score_weight
        self.reduction = reduction

        self.angle_loss = AngleLoss(reduction="none")
        self.width_loss = nn.SmoothL1Loss(reduction="none")
        self.score_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, target, valid_mask=None):
        """
        Compute combined grasp loss.

        Args:
            pred: (N, 4) = [cos2theta, sin2theta, width, score]
            target: (N, 4) = [cos2theta, sin2theta, width, score]
            valid_mask: (N,) boolean mask for valid samples

        Returns:
            loss_dict: Dictionary with individual and total losses
        """
        if pred.shape[0] == 0:
            return {
                'loss_angle': pred.new_tensor(0.0),
                'loss_width': pred.new_tensor(0.0),
                'loss_score': pred.new_tensor(0.0),
                'loss_grasp': pred.new_tensor(0.0),
            }

        # Angle loss
        pred_angle = pred[:, :2]
        target_angle = target[:, :2]
        loss_angle = self.angle_loss(pred_angle, target_angle)

        # Width loss
        loss_width = self.width_loss(pred[:, 2], target[:, 2])

        # Score loss (only where target score > 0)
        score_mask = target[:, 3] > 0
        if score_mask.any():
            loss_score = torch.zeros_like(pred[:, 3])
            loss_score[score_mask] = self.score_loss(
                pred[score_mask, 3],
                target[score_mask, 3]
            )
        else:
            loss_score = torch.zeros_like(pred[:, 3])

        # Apply valid mask if provided
        if valid_mask is not None:
            loss_angle = loss_angle[valid_mask]
            loss_width = loss_width[valid_mask]
            loss_score = loss_score[valid_mask]

        # Reduce
        if self.reduction == "mean":
            loss_angle = loss_angle.mean() if loss_angle.numel() > 0 else pred.new_tensor(0.0)
            loss_width = loss_width.mean() if loss_width.numel() > 0 else pred.new_tensor(0.0)
            loss_score = loss_score.mean() if loss_score.numel() > 0 else pred.new_tensor(0.0)
        elif self.reduction == "sum":
            loss_angle = loss_angle.sum()
            loss_width = loss_width.sum()
            loss_score = loss_score.sum()

        # Total loss
        loss_grasp = (
            self.angle_weight * loss_angle +
            self.width_weight * loss_width +
            self.score_weight * loss_score
        )

        return {
            'loss_angle': loss_angle,
            'loss_width': loss_width,
            'loss_score': loss_score,
            'loss_grasp': loss_grasp,
        }


class MinDistanceGraspLoss(nn.Module):
    """
    Grasp loss that computes against the closest valid grasp among multiple GT grasps.

    This is more tolerant than single-target loss since objects often have multiple
    valid grasp positions. The prediction is considered correct if it matches ANY
    valid grasp for the object.

    Usage:
        For each prediction, provide all valid GT grasps for that object.
        Loss is computed against the closest matching grasp.
    """

    def __init__(
        self,
        position_weight=1.0,
        angle_weight=1.0,
        width_weight=1.0,
        reduction="mean",
    ):
        super().__init__()
        self.position_weight = position_weight
        self.angle_weight = angle_weight
        self.width_weight = width_weight
        self.reduction = reduction

    def forward(self, pred, all_targets, target_counts):
        """
        Compute grasp loss against closest matching GT grasp.

        Args:
            pred: (N, 6) predictions [grasp_dx, grasp_dy, cos2theta, sin2theta, width, score]
            all_targets: (M, 6) all GT grasps concatenated
            target_counts: (N,) number of GT grasps for each prediction

        Returns:
            loss: scalar or (N,) depending on reduction
        """
        if pred.shape[0] == 0:
            if self.reduction == "none":
                return pred.new_zeros(0)
            return pred.new_tensor(0.0)

        losses = []
        target_idx = 0

        for i in range(pred.shape[0]):
            n_targets = target_counts[i].item()
            if n_targets == 0:
                # No valid grasps for this object, use zero loss
                losses.append(pred.new_tensor(0.0))
                continue

            # Get all GT grasps for this prediction
            gt_grasps = all_targets[target_idx:target_idx + n_targets]  # (K, 6)
            target_idx += n_targets

            # Compute distance to each GT grasp
            pred_i = pred[i:i+1].expand(n_targets, -1)  # (K, 6)

            # Position distance
            pos_dist = ((pred_i[:, 0] - gt_grasps[:, 0])**2 +
                       (pred_i[:, 1] - gt_grasps[:, 1])**2)

            # Angle distance (L2 on cos/sin encoding)
            angle_dist = ((pred_i[:, 2] - gt_grasps[:, 2])**2 +
                         (pred_i[:, 3] - gt_grasps[:, 3])**2)

            # Width distance
            width_dist = (pred_i[:, 4] - gt_grasps[:, 4])**2

            # Combined distance for matching
            total_dist = (self.position_weight * pos_dist +
                         self.angle_weight * angle_dist +
                         self.width_weight * width_dist)

            # Find closest GT grasp
            min_idx = torch.argmin(total_dist)
            closest_gt = gt_grasps[min_idx]

            # Compute actual loss against closest GT
            loss_pos = F.smooth_l1_loss(pred[i, :2], closest_gt[:2], reduction='sum')
            loss_angle = F.l1_loss(pred[i, 2:4], closest_gt[2:4], reduction='sum')
            loss_width = F.smooth_l1_loss(pred[i, 4], closest_gt[4], reduction='sum')

            loss_i = (self.position_weight * loss_pos +
                     self.angle_weight * loss_angle +
                     self.width_weight * loss_width)
            losses.append(loss_i)

        losses = torch.stack(losses)

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        return losses


class MultiGraspToleranceLoss(nn.Module):
    """
    完整的多抓取容忍损失函数。

    核心思想：
    1. 位置损失: 预测中心落在任一GT抓取的高斯范围内都算正确
    2. 角度/宽度损失: 对比与预测位置最近的GT抓取的参数

    这解决了anchor-based预测中"抓取中心错误时如何计算其他损失"的问题。

    工作流程:
    ┌─────────────────────────────────────────────────────────┐
    │  预测: (pred_dx, pred_dy, pred_cos, pred_sin, pred_w)   │
    │  GT抓取集合: {G1, G2, G3, ...} 每个Gi有完整6-DOF参数    │
    │                                                         │
    │  Step 1: 计算预测位置到每个GT位置的距离                  │
    │          dist_i = ||(pred_dx, pred_dy) - (gt_dx_i, gt_dy_i)||
    │                                                         │
    │  Step 2: 位置损失 (高斯容忍)                            │
    │          response_i = exp(-dist_i² / 2σ²)               │
    │          L_pos = 1 - max(response_i)                    │
    │                                                         │
    │  Step 3: 找到最近的GT抓取                               │
    │          nearest_idx = argmin(dist_i)                   │
    │          G_nearest = GT抓取集合[nearest_idx]            │
    │                                                         │
    │  Step 4: 角度/宽度损失 (对比最近GT)                     │
    │          L_angle = L1(pred_cos/sin, G_nearest.cos/sin)  │
    │          L_width = SmoothL1(pred_w, G_nearest.w)        │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        sigma=0.15,  # 高斯sigma (归一化bbox坐标)
        position_weight=1.0,
        angle_weight=1.0,
        width_weight=1.0,
        score_weight=0.5,
        reduction="mean",
    ):
        super().__init__()
        self.sigma = sigma
        self.position_weight = position_weight
        self.angle_weight = angle_weight
        self.width_weight = width_weight
        self.score_weight = score_weight
        self.reduction = reduction

    def forward(self, pred, all_gt_grasps, gt_counts):
        """
        计算多抓取容忍损失。

        Args:
            pred: (N, 6) 预测 [grasp_dx, grasp_dy, cos2θ, sin2θ, width, score]
            all_gt_grasps: (M, 6) 所有GT抓取拼接
            gt_counts: (N,) 每个预测对应的GT抓取数量

        Returns:
            loss_dict: 包含各分量损失和总损失
        """
        if pred.shape[0] == 0:
            return {
                'loss_pos': pred.new_tensor(0.0),
                'loss_angle': pred.new_tensor(0.0),
                'loss_width': pred.new_tensor(0.0),
                'loss_score': pred.new_tensor(0.0),
                'loss_grasp': pred.new_tensor(0.0),
            }

        N = pred.shape[0]
        loss_pos_list = []
        loss_angle_list = []
        loss_width_list = []
        loss_score_list = []

        gt_idx = 0
        for i in range(N):
            n_gt = int(gt_counts[i].item())

            if n_gt == 0:
                # 无有效GT，损失为0
                loss_pos_list.append(pred.new_tensor(0.0))
                loss_angle_list.append(pred.new_tensor(0.0))
                loss_width_list.append(pred.new_tensor(0.0))
                loss_score_list.append(pred.new_tensor(0.0))
                continue

            # 获取该预测对应的所有GT抓取
            gt_grasps = all_gt_grasps[gt_idx:gt_idx + n_gt]  # (K, 6)
            gt_idx += n_gt

            pred_i = pred[i]  # (6,)

            # ========== Step 1: 计算位置距离 ==========
            pred_pos = pred_i[:2]  # (2,)
            gt_pos = gt_grasps[:, :2]  # (K, 2)
            sq_dist = ((pred_pos.unsqueeze(0) - gt_pos) ** 2).sum(dim=1)  # (K,)
            dist = torch.sqrt(sq_dist + 1e-8)  # (K,)

            # ========== Step 2: 位置损失 (高斯容忍) ==========
            gaussian_response = torch.exp(-sq_dist / (2 * self.sigma ** 2))  # (K,)
            max_response = gaussian_response.max()
            loss_pos = 1.0 - max_response

            # ========== Step 3: 找到最近的GT ==========
            nearest_idx = torch.argmin(dist)
            nearest_gt = gt_grasps[nearest_idx]  # (6,)

            # ========== Step 4: 角度/宽度损失 (对比最近GT) ==========
            # 角度损失
            loss_cos = torch.abs(pred_i[2] - nearest_gt[2])
            loss_sin = torch.abs(pred_i[3] - nearest_gt[3])
            loss_angle = loss_cos + loss_sin

            # 宽度损失
            loss_width = F.smooth_l1_loss(pred_i[4], nearest_gt[4], reduction='sum')

            # 置信度损失 (仅当GT score > 0)
            if nearest_gt[5] > 0:
                loss_score = F.binary_cross_entropy_with_logits(
                    pred_i[5], nearest_gt[5], reduction='sum'
                )
            else:
                loss_score = pred.new_tensor(0.0)

            loss_pos_list.append(loss_pos)
            loss_angle_list.append(loss_angle)
            loss_width_list.append(loss_width)
            loss_score_list.append(loss_score)

        # 堆叠损失
        loss_pos = torch.stack(loss_pos_list)
        loss_angle = torch.stack(loss_angle_list)
        loss_width = torch.stack(loss_width_list)
        loss_score = torch.stack(loss_score_list)

        # 规约
        if self.reduction == "mean":
            loss_pos = loss_pos.mean()
            loss_angle = loss_angle.mean()
            loss_width = loss_width.mean()
            loss_score = loss_score.mean()
        elif self.reduction == "sum":
            loss_pos = loss_pos.sum()
            loss_angle = loss_angle.sum()
            loss_width = loss_width.sum()
            loss_score = loss_score.sum()

        # 总损失
        loss_grasp = (
            self.position_weight * loss_pos +
            self.angle_weight * loss_angle +
            self.width_weight * loss_width +
            self.score_weight * loss_score
        )

        return {
            'loss_pos': loss_pos,
            'loss_angle': loss_angle,
            'loss_width': loss_width,
            'loss_score': loss_score,
            'loss_grasp': loss_grasp,
        }


def decode_grasp_angle(cos2theta, sin2theta):
    """
    Decode grasp angle from cos/sin encoding.

    Args:
        cos2theta: cos(2*theta)
        sin2theta: sin(2*theta)

    Returns:
        theta: grasp angle in radians [-pi/2, pi/2]
    """
    return 0.5 * torch.atan2(sin2theta, cos2theta)


def encode_grasp_angle(theta):
    """
    Encode grasp angle to cos/sin.

    Args:
        theta: grasp angle in radians

    Returns:
        cos2theta, sin2theta
    """
    return torch.cos(2 * theta), torch.sin(2 * theta)
