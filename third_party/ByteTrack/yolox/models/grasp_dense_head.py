#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GraspDenseHead: 密集抓取预测头

混合架构:
1. YOLOX: 目标检测 + 跟踪 (稀疏，每个物体1个bbox)
2. GraspDenseHead: 密集抓取预测 (每个像素位置预测抓取参数)

输出格式 (每个像素):
- grasp_heatmap: [1] 该位置作为抓取中心的置信度
- grasp_params:  [4] (cos2theta, sin2theta, width, quality)

使用流程:
1. YOLOX检测物体bbox
2. GraspDenseHead预测全图密集抓取
3. 对每个bbox，从对应区域采样最佳抓取点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .network_blocks import BaseConv, DWConv


class GraspDenseHead(nn.Module):
    """
    密集抓取预测头

    输入: FPN特征 [P3, P4, P5]
    输出:
        - grasp_heatmap: (B, 1, H/4, W/4) 抓取中心热图
        - grasp_cos:     (B, 1, H/4, W/4) cos(2θ)
        - grasp_sin:     (B, 1, H/4, W/4) sin(2θ)
        - grasp_width:   (B, 1, H/4, W/4) 抓取宽度
        - grasp_quality: (B, 1, H/4, W/4) 抓取质量
    """

    def __init__(
        self,
        in_channels=[256, 512, 1024],
        hidden_dim=256,
        output_stride=4,  # 输出相对于输入的下采样率
        act="silu",
    ):
        super().__init__()

        self.output_stride = output_stride

        # 特征融合 (FPN → 单一特征图)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, hidden_dim, 1, 1, 0)
            )
            self.fpn_convs.append(
                BaseConv(hidden_dim, hidden_dim, 3, 1, act=act)
            )

        # 上采样融合
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 抓取预测头
        self.grasp_stem = nn.Sequential(
            BaseConv(hidden_dim, hidden_dim, 3, 1, act=act),
            BaseConv(hidden_dim, hidden_dim, 3, 1, act=act),
        )

        # 分支预测
        self.heatmap_head = nn.Sequential(
            BaseConv(hidden_dim, hidden_dim // 2, 3, 1, act=act),
            nn.Conv2d(hidden_dim // 2, 1, 1, 1, 0),
        )

        self.angle_head = nn.Sequential(
            BaseConv(hidden_dim, hidden_dim // 2, 3, 1, act=act),
            nn.Conv2d(hidden_dim // 2, 2, 1, 1, 0),  # cos, sin
        )

        self.width_head = nn.Sequential(
            BaseConv(hidden_dim, hidden_dim // 2, 3, 1, act=act),
            nn.Conv2d(hidden_dim // 2, 1, 1, 1, 0),
        )

        self.quality_head = nn.Sequential(
            BaseConv(hidden_dim, hidden_dim // 2, 3, 1, act=act),
            nn.Conv2d(hidden_dim // 2, 1, 1, 1, 0),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Heatmap头使用特殊初始化 (focal loss)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.heatmap_head[-1].bias, bias_value)

    def forward(self, features):
        """
        前向传播

        Args:
            features: List of FPN features [P3, P4, P5]
                     P3: (B, C, H/8, W/8)
                     P4: (B, C, H/16, W/16)
                     P5: (B, C, H/32, W/32)

        Returns:
            grasp_output: dict with dense predictions
        """
        # 特征金字塔融合 (自顶向下)
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # P5 → P4 → P3 融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + self.upsample(laterals[i])

        # 使用最高分辨率特征 (P3, stride=8)
        feat = self.fpn_convs[0](laterals[0])

        # 上采样到 stride=4
        feat = self.upsample(feat)

        # 抓取特征
        grasp_feat = self.grasp_stem(feat)

        # 分支预测
        heatmap = self.heatmap_head(grasp_feat)  # (B, 1, H/4, W/4)
        angle = self.angle_head(grasp_feat)       # (B, 2, H/4, W/4)
        width = self.width_head(grasp_feat)       # (B, 1, H/4, W/4)
        quality = self.quality_head(grasp_feat)   # (B, 1, H/4, W/4)

        return {
            'heatmap': heatmap,         # logits, 需要sigmoid
            'cos2theta': angle[:, 0:1], # tanh后 [-1, 1]
            'sin2theta': angle[:, 1:2], # tanh后 [-1, 1]
            'width': width,             # sigmoid后 [0, 1]
            'quality': quality,         # sigmoid后 [0, 1]
        }

    def decode(self, outputs, threshold=0.5):
        """
        解码密集预测为抓取点列表

        Args:
            outputs: forward()的输出
            threshold: 热图阈值

        Returns:
            grasps: List of (x, y, theta, width, score) per image
        """
        heatmap = torch.sigmoid(outputs['heatmap'])
        cos2theta = torch.tanh(outputs['cos2theta'])
        sin2theta = torch.tanh(outputs['sin2theta'])
        width = torch.sigmoid(outputs['width'])
        quality = torch.sigmoid(outputs['quality'])

        batch_size = heatmap.shape[0]
        results = []

        for b in range(batch_size):
            # 找到高置信度位置
            heat = heatmap[b, 0]  # (H, W)
            mask = heat > threshold

            if not mask.any():
                results.append([])
                continue

            # 提取抓取点
            ys, xs = torch.where(mask)
            scores = heat[mask]
            cos_vals = cos2theta[b, 0][mask]
            sin_vals = sin2theta[b, 0][mask]
            width_vals = width[b, 0][mask]
            quality_vals = quality[b, 0][mask]

            # 转换坐标到原图
            xs = xs.float() * self.output_stride
            ys = ys.float() * self.output_stride

            # 解码角度
            thetas = 0.5 * torch.atan2(sin_vals, cos_vals)

            # 组装结果
            grasps = torch.stack([
                xs, ys, thetas, width_vals * 100, scores * quality_vals
            ], dim=1)

            results.append(grasps)

        return results


class GraspDenseLoss(nn.Module):
    """
    密集抓取预测损失

    使用高斯热图作为GT，而非稀疏点
    """

    def __init__(
        self,
        heatmap_weight=1.0,
        angle_weight=1.0,
        width_weight=1.0,
        quality_weight=0.5,
        gaussian_sigma=2.0,  # 高斯核大小
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.angle_weight = angle_weight
        self.width_weight = width_weight
        self.quality_weight = quality_weight
        self.gaussian_sigma = gaussian_sigma

    def generate_heatmap(self, grasps, output_size, img_size):
        """
        生成高斯热图GT

        Args:
            grasps: List of (cx, cy, cos, sin, width, score) per grasp
            output_size: (H, W) of output heatmap
            img_size: (H, W) of input image

        Returns:
            heatmap: (H, W) Gaussian heatmap
            angle_map: (2, H, W) cos/sin at each position
            width_map: (H, W) width at each position
            quality_map: (H, W) quality at each position
        """
        H, W = output_size
        img_H, img_W = img_size
        scale_y = H / img_H
        scale_x = W / img_W

        heatmap = torch.zeros(H, W)
        angle_map = torch.zeros(2, H, W)
        width_map = torch.zeros(H, W)
        quality_map = torch.zeros(H, W)

        if len(grasps) == 0:
            return heatmap, angle_map, width_map, quality_map

        # 生成网格
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        xx = xx.float()
        yy = yy.float()

        for grasp in grasps:
            cx, cy, cos2t, sin2t, w, score = grasp

            # 转换到输出坐标
            cx_out = cx * scale_x
            cy_out = cy * scale_y

            # 高斯响应
            dist_sq = (xx - cx_out) ** 2 + (yy - cy_out) ** 2
            gaussian = torch.exp(-dist_sq / (2 * self.gaussian_sigma ** 2))

            # 取最大值 (处理重叠)
            heatmap = torch.max(heatmap, gaussian * score)

            # 对于其他参数，使用最近抓取的值
            mask = gaussian > 0.5
            angle_map[0][mask] = cos2t
            angle_map[1][mask] = sin2t
            width_map[mask] = w
            quality_map[mask] = score

        return heatmap, angle_map, width_map, quality_map

    def focal_loss(self, pred, target, alpha=2, beta=4, pos_thresh=0.5):
        """
        Modified Focal Loss for Gaussian heatmap GT (from CenterNet)

        GT是高斯分布[0, 1]，不是binary。
        - 正样本: target > pos_thresh 的位置
        - 负样本: target <= pos_thresh 的位置

        Note: 强制使用FP32计算避免FP16溢出
        """
        # 强制转换为FP32计算，避免FP16溢出
        pred = pred.float()
        target = target.float()

        eps = 1e-6  # FP32可以使用更小的eps

        # 确保target在有效范围内
        target = torch.clamp(target, 0, 1)

        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, eps, 1 - eps)

        # 正样本: 高斯峰值位置
        pos_mask = (target > pos_thresh).float()
        # 负样本: 其他位置
        neg_mask = (target <= pos_thresh).float()

        # 正样本损失: 鼓励预测接近1
        pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask

        # 负样本损失: 鼓励预测接近0，同时降低靠近正样本区域的权重
        neg_weight = torch.pow(1 - target, beta)
        neg_loss = -torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weight * neg_mask

        num_pos = pos_mask.sum()
        pos_loss_sum = pos_loss.sum()
        neg_loss_sum = neg_loss.sum()

        if num_pos == 0:
            return neg_loss_sum / (neg_mask.sum() + eps)
        return (pos_loss_sum + neg_loss_sum) / (num_pos + eps)

    def forward(self, outputs, targets):
        """
        计算密集抓取损失

        Args:
            outputs: dict from GraspDenseHead.forward()
            targets: dict with GT heatmaps and param maps

        Returns:
            loss_dict: dict with individual losses
        """
        # Heatmap loss (Focal Loss)
        loss_heatmap = self.focal_loss(
            outputs['heatmap'],
            targets['heatmap']
        )

        # 只在有效抓取位置计算其他损失
        mask = targets['heatmap'] > 0.5

        if mask.sum() > 0:
            # Angle loss
            pred_cos = torch.tanh(outputs['cos2theta'])
            pred_sin = torch.tanh(outputs['sin2theta'])
            loss_angle = (
                F.l1_loss(pred_cos[mask], targets['cos2theta'][mask]) +
                F.l1_loss(pred_sin[mask], targets['sin2theta'][mask])
            )

            # Width loss
            pred_width = torch.sigmoid(outputs['width'])
            loss_width = F.smooth_l1_loss(
                pred_width[mask], targets['width'][mask]
            )

            # Quality loss - 使用 bce_with_logits 支持 FP16 autocast
            # 因为我们已经对 outputs['quality'] 应用了sigmoid，这里需要logits形式
            # 但为了保持一致性，我们改用直接计算logits的方式
            loss_quality = F.binary_cross_entropy_with_logits(
                outputs['quality'][mask], targets['quality'][mask]
            )
        else:
            loss_angle = outputs['cos2theta'].new_tensor(0.0)
            loss_width = outputs['width'].new_tensor(0.0)
            loss_quality = outputs['quality'].new_tensor(0.0)

        # Total loss
        loss_total = (
            self.heatmap_weight * loss_heatmap +
            self.angle_weight * loss_angle +
            self.width_weight * loss_width +
            self.quality_weight * loss_quality
        )

        return {
            'loss_heatmap': loss_heatmap,
            'loss_angle': loss_angle,
            'loss_width': loss_width,
            'loss_quality': loss_quality,
            'loss_grasp_dense': loss_total,
        }


def sample_grasps_in_bbox(dense_outputs, bbox, top_k=5, stride=4):
    """
    从密集预测中采样bbox区域内的抓取点

    Args:
        dense_outputs: GraspDenseHead的输出
        bbox: [x1, y1, x2, y2] 检测框
        top_k: 返回top-k个抓取点
        stride: 密集预测的步长

    Returns:
        grasps: List of (x, y, theta, width, score)
    """
    heatmap = torch.sigmoid(dense_outputs['heatmap'][0, 0])  # (H, W)
    cos2theta = torch.tanh(dense_outputs['cos2theta'][0, 0])
    sin2theta = torch.tanh(dense_outputs['sin2theta'][0, 0])
    width = torch.sigmoid(dense_outputs['width'][0, 0])
    quality = torch.sigmoid(dense_outputs['quality'][0, 0])

    x1, y1, x2, y2 = bbox

    # 转换到特征图坐标
    fx1, fy1 = int(x1 / stride), int(y1 / stride)
    fx2, fy2 = int(x2 / stride) + 1, int(y2 / stride) + 1

    # 裁剪区域
    H, W = heatmap.shape
    fx1, fy1 = max(0, fx1), max(0, fy1)
    fx2, fy2 = min(W, fx2), min(H, fy2)

    if fx2 <= fx1 or fy2 <= fy1:
        return []

    # 提取区域内的热图
    region_heat = heatmap[fy1:fy2, fx1:fx2]
    region_cos = cos2theta[fy1:fy2, fx1:fx2]
    region_sin = sin2theta[fy1:fy2, fx1:fx2]
    region_width = width[fy1:fy2, fx1:fx2]
    region_quality = quality[fy1:fy2, fx1:fx2]

    # 找top-k位置
    flat_heat = region_heat.flatten()
    k = min(top_k, flat_heat.numel())
    topk_vals, topk_inds = torch.topk(flat_heat, k)

    grasps = []
    for i in range(k):
        if topk_vals[i] < 0.1:  # 阈值
            continue

        idx = topk_inds[i]
        ry, rx = idx // region_heat.shape[1], idx % region_heat.shape[1]

        # 转换回原图坐标
        x = (fx1 + rx) * stride + stride / 2
        y = (fy1 + ry) * stride + stride / 2

        # 边界检查：确保抓取中心在bbox内
        if x < x1 or x > x2 or y < y1 or y > y2:
            continue

        cos_val = region_cos[ry, rx]
        sin_val = region_sin[ry, rx]
        theta = 0.5 * torch.atan2(sin_val, cos_val)

        w = region_width[ry, rx] * 100  # 转换为像素
        score = topk_vals[i] * region_quality[ry, rx]

        grasps.append({
            'x': x.item(),
            'y': y.item(),
            'theta': theta.item(),
            'width': w.item(),
            'score': score.item(),
        })

    return grasps
