"""
ROI Feature Extractor for Instance-Level Grasp Prediction.

Extracts ROI features from ViT features or 2D feature maps for
per-instance grasp prediction.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIExtractor(nn.Module):
    """
    ROI feature extractor for instance-level prediction.

    Supports extraction from:
    1. 2D feature maps using RoIAlign
    2. ViT sequence features via reshape + interpolation
    """

    def __init__(
        self,
        context_ratio: float = 1.5,
        output_size: Tuple[int, int] = (30, 40),
        align_corners: bool = False,
    ):
        """
        Initialize ROI extractor.

        Args:
            context_ratio: Ratio to expand bounding boxes for context
            output_size: Output ROI size (H, W)
            align_corners: Align corners in interpolation
        """
        super().__init__()
        self.context_ratio = context_ratio
        self.output_size = output_size
        self.align_corners = align_corners

    def expand_boxes(
        self,
        boxes: torch.Tensor,
        image_size: Tuple[int, int],
        make_square: bool = True,
    ) -> torch.Tensor:
        """
        Expand bounding boxes to include context.

        Args:
            boxes: [N, 4] as (x1, y1, x2, y2)
            image_size: (H, W) of original image
            make_square: If True, expand to square

        Returns:
            expanded_boxes: [N, 4]
        """
        if len(boxes) == 0:
            return boxes

        x1, y1, x2, y2 = boxes.unbind(-1)
        H, W = image_size

        # Calculate center and size
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Expand
        if make_square:
            size = torch.max(w, h) * self.context_ratio
            new_w = size
            new_h = size
        else:
            new_w = w * self.context_ratio
            new_h = h * self.context_ratio

        # Calculate new box coordinates
        new_x1 = (cx - new_w / 2).clamp(min=0)
        new_y1 = (cy - new_h / 2).clamp(min=0)
        new_x2 = (cx + new_w / 2).clamp(max=W)
        new_y2 = (cy + new_h / 2).clamp(max=H)

        return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)

    def extract_from_2d_features(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Extract ROI features from 2D feature map using RoIAlign.

        Args:
            features: [B, C, fH, fW] feature map
            boxes: [N, 4] as (x1, y1, x2, y2) in pixel coordinates
            batch_ids: [N] batch index for each box (default: all 0)
            image_size: Original image size (H, W) for scaling

        Returns:
            roi_features: [N, C, output_H, output_W]
        """
        try:
            from torchvision.ops import roi_align
        except ImportError:
            return self._extract_from_2d_manual(features, boxes, batch_ids, image_size)

        if len(boxes) == 0:
            return torch.zeros(
                (0, features.shape[1], *self.output_size),
                device=features.device,
                dtype=features.dtype,
            )

        B, C, fH, fW = features.shape

        # Calculate spatial scale
        if image_size is not None:
            H, W = image_size
            spatial_scale_x = fW / W
            spatial_scale_y = fH / H

            # Check for non-uniform scaling (potential issue)
            scale_diff = abs(spatial_scale_x - spatial_scale_y)
            if scale_diff > 0.01:  # More than 1% difference
                import warnings
                warnings.warn(
                    f"Non-uniform spatial scale detected: "
                    f"scale_x={spatial_scale_x:.4f}, scale_y={spatial_scale_y:.4f}. "
                    f"This may cause coordinate misalignment. "
                    f"Consider using square input or matching aspect ratios."
                )

            # Use geometric mean for better balance with non-square images
            # This minimizes the maximum error in both dimensions
            spatial_scale = (spatial_scale_x * spatial_scale_y) ** 0.5
        else:
            spatial_scale = 1.0

        # Prepare batch indices
        if batch_ids is None:
            batch_ids = torch.zeros(len(boxes), device=boxes.device)

        # Prepare ROIs: [batch_idx, x1, y1, x2, y2]
        rois = torch.cat([batch_ids.unsqueeze(1), boxes], dim=1)

        # Apply RoIAlign
        roi_features = roi_align(
            features,
            rois,
            output_size=self.output_size,
            spatial_scale=spatial_scale,
            aligned=True,
        )

        return roi_features

    def _extract_from_2d_manual(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Manual ROI extraction without torchvision.

        Uses grid_sample for bilinear interpolation.
        """
        if len(boxes) == 0:
            return torch.zeros(
                (0, features.shape[1], *self.output_size),
                device=features.device,
                dtype=features.dtype,
            )

        B, C, fH, fW = features.shape
        N = len(boxes)

        if batch_ids is None:
            batch_ids = torch.zeros(N, dtype=torch.long, device=boxes.device)

        # Calculate scale
        if image_size is not None:
            H, W = image_size
        else:
            H, W = fH, fW

        roi_features = []

        for i in range(N):
            batch_idx = int(batch_ids[i])
            x1, y1, x2, y2 = boxes[i]

            # Normalize to [-1, 1] for grid_sample
            # Map box to feature map coordinates
            fx1 = (x1 / W) * 2 - 1
            fx2 = (x2 / W) * 2 - 1
            fy1 = (y1 / H) * 2 - 1
            fy2 = (y2 / H) * 2 - 1

            # Create sampling grid
            out_h, out_w = self.output_size
            theta = torch.tensor([
                [(fx2 - fx1) / 2, 0, (fx1 + fx2) / 2],
                [0, (fy2 - fy1) / 2, (fy1 + fy2) / 2],
            ], device=boxes.device, dtype=boxes.dtype).unsqueeze(0)

            grid = F.affine_grid(theta, (1, C, out_h, out_w), align_corners=self.align_corners)
            roi = F.grid_sample(
                features[batch_idx:batch_idx+1],
                grid,
                mode='bilinear',
                align_corners=self.align_corners,
            )
            roi_features.append(roi)

        return torch.cat(roi_features, dim=0)

    def extract_from_vit_sequence(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
        image_size: Tuple[int, int] = (480, 640),
        patch_size: int = 14,
    ) -> torch.Tensor:
        """
        Extract ROI features from ViT sequence features.

        Args:
            features: [B, N, D] ViT sequence features
            boxes: [M, 4] pixel coordinates (x1, y1, x2, y2)
            batch_ids: [M] batch indices
            image_size: Original image size (H, W)
            patch_size: ViT patch size

        Returns:
            roi_features: [M, D, output_H, output_W]
        """
        B, N, D = features.shape
        H, W = image_size

        # Calculate feature map dimensions
        fH = H // patch_size
        fW = W // patch_size
        expected_N = fH * fW

        # Calculate the actual image area covered by ViT patches
        actual_H = fH * patch_size
        actual_W = fW * patch_size

        # Warn about edge pixels not covered by ViT
        if actual_H < H or actual_W < W:
            import warnings
            warnings.warn(
                f"ViT patches do not cover entire image. "
                f"Image: {H}x{W}, ViT covers: {actual_H}x{actual_W}. "
                f"Edge pixels ({H - actual_H} bottom, {W - actual_W} right) are not represented.",
                stacklevel=2
            )

        # Clip boxes to the area covered by ViT features
        if len(boxes) > 0:
            boxes = boxes.clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(max=actual_W)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(max=actual_H)

        # Handle potential padding tokens
        if N > expected_N:
            features = features[:, :expected_N, :]
        elif N < expected_N:
            # Pad if necessary
            padding = torch.zeros(B, expected_N - N, D, device=features.device, dtype=features.dtype)
            features = torch.cat([features, padding], dim=1)

        # Reshape to 2D: [B, N, D] -> [B, D, fH, fW]
        features_2d = features.permute(0, 2, 1).reshape(B, D, fH, fW)

        # Use 2D extraction with the actual covered image size
        return self.extract_from_2d_features(
            features_2d, boxes, batch_ids, (actual_H, actual_W)
        )

    def forward(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
        is_vit_sequence: bool = False,
        patch_size: int = 14,
    ) -> torch.Tensor:
        """
        Extract ROI features.

        Args:
            features: Input features [B, C, H, W] or [B, N, D]
            boxes: Bounding boxes [M, 4]
            batch_ids: Batch indices [M]
            image_size: Original image size
            is_vit_sequence: If True, treat features as ViT sequence
            patch_size: ViT patch size (if is_vit_sequence)

        Returns:
            ROI features [M, C/D, output_H, output_W]
        """
        if is_vit_sequence:
            if image_size is None:
                raise ValueError("image_size required for ViT sequence features")
            return self.extract_from_vit_sequence(
                features, boxes, batch_ids, image_size, patch_size
            )
        else:
            return self.extract_from_2d_features(
                features, boxes, batch_ids, image_size
            )


class MultiScaleROIExtractor(nn.Module):
    """
    Multi-scale ROI feature extractor.

    Extracts features from multiple FPN levels and combines them.
    """

    def __init__(
        self,
        context_ratio: float = 1.5,
        output_size: Tuple[int, int] = (30, 40),
        feature_channels: List[int] = [256, 512, 1024],
        out_channels: int = 256,
    ):
        super().__init__()

        self.roi_extractor = ROIExtractor(
            context_ratio=context_ratio,
            output_size=output_size,
        )

        # Channel projection for each level
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in feature_channels
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(feature_channels), out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        boxes: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Extract and fuse multi-scale ROI features.

        Args:
            features: Dict with 'P3', 'P4', 'P5' features
            boxes: [N, 4] bounding boxes
            batch_ids: [N] batch indices
            image_size: Original image size

        Returns:
            Fused ROI features [N, out_channels, H, W]
        """
        if len(boxes) == 0:
            out_ch = self.projections[0].out_channels
            return torch.zeros(
                (0, out_ch, *self.roi_extractor.output_size),
                device=boxes.device,
            )

        roi_features_list = []

        for i, (key, proj) in enumerate(zip(['P3', 'P4', 'P5'], self.projections)):
            if key not in features:
                continue

            feat = features[key]
            roi_feat = self.roi_extractor.extract_from_2d_features(
                feat, boxes, batch_ids, image_size
            )
            roi_feat = proj(roi_feat)
            roi_features_list.append(roi_feat)

        if len(roi_features_list) == 0:
            raise ValueError("No valid features found")

        # Concatenate and fuse
        roi_features = torch.cat(roi_features_list, dim=1)
        roi_features = self.fusion(roi_features)

        return roi_features
