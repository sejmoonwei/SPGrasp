#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX-Grasp: 集成检测跟踪 + 密集抓取预测的完整模型

架构:
┌─────────────────────────────────────────────────────────────────────┐
│                        YOLOX-Grasp                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  输入图像 ──────────────────────────────────────────────────────    │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Backbone (CSPDarknet)                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Neck (PAFPN) → FPN Features                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│      │                                                              │
│      ├────────────────────┬────────────────────┐                   │
│      │                    │                    │                    │
│      ▼                    ▼                    ▼                    │
│  ┌──────────┐      ┌──────────────┐    ┌──────────────────────┐   │
│  │ YOLOXHead│      │ GraspDense   │    │   (可选) 其他任务头   │   │
│  │          │      │    Head      │    │   深度/分割等         │   │
│  │ • bbox   │      │              │    │                      │   │
│  │ • obj    │      │ • heatmap    │    │                      │   │
│  │ • cls    │      │ • angle      │    │                      │   │
│  │          │      │ • width      │    │                      │   │
│  └──────────┘      └──────────────┘    └──────────────────────┘   │
│      │                    │                                         │
│      ▼                    ▼                                         │
│  检测结果              密集抓取预测                                  │
│  [bbox, conf]          [heatmap, angle, width]                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

使用流程:
1. 模型输出检测框 + 密集抓取热图
2. ByteTrack进行多目标跟踪
3. 对每个跟踪目标，从其bbox区域采样抓取点
"""

import torch
import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .grasp_dense_head import GraspDenseHead, GraspDenseLoss


class YOLOXGrasp(nn.Module):
    """
    YOLOX + Dense Grasp 集成模型

    将目标检测和密集抓取预测统一到一个模型中，共享backbone和neck。

    注意: backbone 参数实际上是 YOLOPAFPN，它内部包含 CSPDarknet backbone + FPN neck。
    这与原始 YOLOX 的命名约定一致。
    """

    def __init__(
        self,
        backbone,
        head,
        grasp_head=None,
        grasp_loss_fn=None,
        grasp_loss_weight=1.0,
    ):
        """
        Args:
            backbone: YOLOPAFPN (包含 CSPDarknet + FPN neck)
            head: YOLOXHead for detection
            grasp_head: GraspDenseHead for dense grasp (optional)
            grasp_loss_fn: GraspDenseLoss instance (optional, created if None)
            grasp_loss_weight: 抓取损失的总权重
        """
        super().__init__()
        self.backbone = backbone  # 实际上是 YOLOPAFPN
        self.head = head
        self.grasp_head = grasp_head
        self.grasp_loss_weight = grasp_loss_weight

        # 损失函数：使用传入的或创建默认的
        if grasp_loss_fn is not None:
            self.grasp_loss_fn = grasp_loss_fn
        else:
            self.grasp_loss_fn = GraspDenseLoss(
                heatmap_weight=1.0,
                angle_weight=1.0,
                width_weight=1.0,
                quality_weight=0.5,
                gaussian_sigma=2.0,
            )

    def set_grasp_loss_fn(self, grasp_loss_fn, grasp_loss_weight=None):
        """设置抓取损失函数（用于从Exp配置中设置）"""
        self.grasp_loss_fn = grasp_loss_fn
        if grasp_loss_weight is not None:
            self.grasp_loss_weight = grasp_loss_weight

    def forward(self, x, targets=None, grasp_targets=None):
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, H, W)
            targets: 检测GT (训练时)
            grasp_targets: 抓取GT (训练时)

        Returns:
            训练模式: (det_loss, grasp_loss, loss_dict)
            推理模式: (det_outputs, grasp_outputs)
        """
        # Backbone (YOLOPAFPN: CSPDarknet + FPN)
        # 输出是 (pan_out2, pan_out1, pan_out0) 三个尺度的特征
        fpn_outs = self.backbone(x)

        # Detection head
        if self.training:
            det_outputs = self.head(fpn_outs, targets, x)
        else:
            det_outputs = self.head(fpn_outs)

        # Grasp head (if available)
        grasp_outputs = None
        if self.grasp_head is not None:
            grasp_outputs = self.grasp_head(fpn_outs)

        if self.training:
            # det_outputs is (loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg)
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = det_outputs

            # Grasp loss
            if self.grasp_head is not None and grasp_targets is not None:
                # Debug: 检查输入形状和值
                if torch.isnan(grasp_outputs['heatmap']).any():
                    print(f"[DEBUG] grasp_outputs heatmap has NaN!")
                if torch.isnan(grasp_targets['heatmap']).any():
                    print(f"[DEBUG] grasp_targets heatmap has NaN!")

                # 确保类型匹配
                grasp_targets = {k: v.to(grasp_outputs['heatmap'].dtype)
                                for k, v in grasp_targets.items()}

                grasp_loss_dict = self.grasp_loss_fn(grasp_outputs, grasp_targets)
                grasp_loss = grasp_loss_dict['loss_grasp_dense'] * self.grasp_loss_weight

                if torch.isnan(grasp_loss):
                    print(f"[DEBUG] grasp_loss is NaN!")
                    print(f"[DEBUG] loss_heatmap: {grasp_loss_dict['loss_heatmap']}")
                    print(f"[DEBUG] loss_angle: {grasp_loss_dict['loss_angle']}")
                    print(f"[DEBUG] loss_width: {grasp_loss_dict['loss_width']}")
                    print(f"[DEBUG] loss_quality: {grasp_loss_dict['loss_quality']}")
            else:
                grasp_loss = x.new_tensor(0.0)

            # Total loss
            total_loss = loss + grasp_loss

            # Return format matching original YOLOX for compatibility with trainer
            return {
                'total_loss': total_loss,
                'iou_loss': iou_loss,
                'l1_loss': l1_loss,
                'conf_loss': conf_loss,
                'cls_loss': cls_loss,
                'grasp_loss': grasp_loss,
                'num_fg': num_fg,
            }
        else:
            return {
                'det_outputs': det_outputs,
                'grasp_outputs': grasp_outputs,
            }


class YOLOXGraspPredictor:
    """
    YOLOX-Grasp 推理器

    封装模型推理 + 后处理 + 抓取采样
    """

    def __init__(
        self,
        model,
        conf_thresh=0.3,
        nms_thresh=0.45,
        grasp_thresh=0.5,
        top_k_grasps=5,
        device='cuda',
    ):
        """
        Args:
            model: YOLOXGrasp模型
            conf_thresh: 检测置信度阈值
            nms_thresh: NMS阈值
            grasp_thresh: 抓取热图阈值
            top_k_grasps: 每个物体采样的抓取数
            device: 运行设备
        """
        self.model = model
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.grasp_thresh = grasp_thresh
        self.top_k_grasps = top_k_grasps
        self.device = device

        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def predict(self, image):
        """
        推理单张图像

        Args:
            image: numpy array (H, W, 3) BGR

        Returns:
            results: List of dict, each containing:
                - bbox: [x1, y1, x2, y2]
                - score: detection confidence
                - grasps: List of (x, y, theta, width, score)
        """
        import cv2
        import numpy as np
        from yolox.utils import postprocess
        from .grasp_dense_head import sample_grasps_in_bbox

        # Preprocess
        img_h, img_w = image.shape[:2]
        # Assume model expects 640x640, adjust as needed
        test_size = (640, 640)
        ratio = min(test_size[0] / img_h, test_size[1] / img_w)
        resized = cv2.resize(image, (int(img_w * ratio), int(img_h * ratio)))

        padded = np.zeros((test_size[0], test_size[1], 3), dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized

        img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        # Forward
        outputs = self.model(img_tensor)
        det_outputs = outputs['det_outputs']
        grasp_outputs = outputs['grasp_outputs']

        # Postprocess detections
        det_results = postprocess(
            det_outputs, 1, self.conf_thresh, self.nms_thresh
        )[0]

        if det_results is None:
            return []

        # Scale back to original image
        det_results[:, :4] /= ratio

        results = []
        for det in det_results:
            x1, y1, x2, y2 = det[:4].tolist()
            score = det[4].item()

            # Sample grasps from dense prediction
            # Scale bbox to feature map coordinates
            bbox_scaled = [x1 * ratio, y1 * ratio, x2 * ratio, y2 * ratio]
            grasps = sample_grasps_in_bbox(
                grasp_outputs, bbox_scaled,
                top_k=self.top_k_grasps,
                stride=4
            )

            # Scale grasp coordinates back
            for g in grasps:
                g['x'] /= ratio
                g['y'] /= ratio

            results.append({
                'bbox': [x1, y1, x2, y2],
                'score': score,
                'grasps': grasps,
            })

        return results


def build_yolox_grasp(
    num_classes=1,
    depth=0.33,
    width=0.50,
    with_grasp_head=True,
):
    """
    构建 YOLOX-Grasp 模型

    Args:
        num_classes: 检测类别数
        depth: 模型深度系数
        width: 模型宽度系数
        with_grasp_head: 是否包含密集抓取头

    Returns:
        YOLOXGrasp model
    """
    # Channels
    in_channels = [256, 512, 1024]

    # Backbone (YOLOPAFPN 包含 CSPDarknet + FPN neck)
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, depthwise=False, act="silu")

    # Detection head
    head = YOLOXHead(num_classes, width, in_channels=in_channels, depthwise=False, act="silu")

    # Grasp head
    grasp_head = None
    if with_grasp_head:
        # Adjust in_channels based on width
        grasp_in_channels = [int(c * width) for c in in_channels]
        grasp_head = GraspDenseHead(
            in_channels=grasp_in_channels,
            hidden_dim=int(256 * width),
            output_stride=4,
            act="silu",
        )

    return YOLOXGrasp(backbone, head, grasp_head)
