"""
YOLOX Bridge for ByteTrack Integration.

Provides a standardized interface to ByteTrack's YOLOX detector for
instance-level grasp prediction.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# ByteTrack path
BYTETRACK_PATH = Path(__file__).parent.parent.parent.parent / "ByteTrack"


class YOLOXBridge:
    """
    ByteTrack YOLOX detector bridge.

    Wraps ByteTrack's YOLOX model to provide a clean interface for
    object detection in the grasp prediction pipeline.
    """

    def __init__(
        self,
        exp_file: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        device: str = "cuda",
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        num_classes: int = 1,
    ):
        """
        Initialize YOLOX bridge.

        Args:
            exp_file: Path to ByteTrack experiment config file
            ckpt_path: Path to model checkpoint
            device: Device to run inference on
            conf_thresh: Confidence threshold for detection
            nms_thresh: NMS IoU threshold
            num_classes: Number of object classes
        """
        self.device = device
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.num_classes = num_classes
        self.available = False

        self.model = None
        self.exp = None
        self.input_size = (640, 640)

        if exp_file is not None:
            self._load_model(exp_file, ckpt_path)

    def _load_model(self, exp_file: str, ckpt_path: Optional[str] = None):
        """Load YOLOX model from ByteTrack."""
        try:
            # Add ByteTrack to path
            bytetrack_path = str(BYTETRACK_PATH)
            if bytetrack_path not in sys.path:
                sys.path.insert(0, bytetrack_path)

            from yolox.exp import get_exp
            from yolox.utils import postprocess

            self.postprocess = postprocess

            # Load experiment config
            self.exp = get_exp(exp_file, None)
            self.input_size = self.exp.input_size

            # Build model
            self.model = self.exp.get_model()

            # Load checkpoint
            if ckpt_path:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                if "model" in ckpt:
                    self.model.load_state_dict(ckpt["model"])
                else:
                    self.model.load_state_dict(ckpt)

            self.model.to(self.device)
            self.model.eval()
            self.available = True

            print(f"YOLOX loaded from {exp_file}")
            if ckpt_path:
                print(f"  Checkpoint: {ckpt_path}")

        except Exception as e:
            print(f"Failed to load YOLOX: {e}")
            print(f"  ByteTrack path: {BYTETRACK_PATH}")
            self.available = False

    @torch.no_grad()
    def detect(
        self,
        image: torch.Tensor,
        return_features: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Detect objects in image.

        Args:
            image: Input image [B, C, H, W] or [C, H, W], normalized to [0, 1]
            return_features: If True, also return FPN features

        Returns:
            boxes: Detection boxes [N, 4] as (x1, y1, x2, y2)
            scores: Confidence scores [N]
            class_ids: Class IDs [N]
            features (optional): Dict of FPN features {'P3', 'P4', 'P5'}
        """
        if not self.available:
            empty_boxes = torch.zeros((0, 4), device=self.device)
            empty_scores = torch.zeros((0,), device=self.device)
            empty_cls = torch.zeros((0,), dtype=torch.long, device=self.device)
            if return_features:
                return empty_boxes, empty_scores, empty_cls, {}
            return empty_boxes, empty_scores, empty_cls

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        B, C, H, W = image.shape

        # Preprocess: scale to [0, 255] and resize
        image_preprocessed = self._preprocess(image)

        # Forward pass - avoid double backbone computation
        if return_features:
            # Extract FPN features once and reuse
            fpn_outs = self.model.backbone(image_preprocessed)
            fpn_features = {
                'P3': fpn_outs[0],  # stride 8
                'P4': fpn_outs[1],  # stride 16
                'P5': fpn_outs[2],  # stride 32
            }
            # Run detection head
            outputs = self.model.head(fpn_outs)
        else:
            outputs = self.model(image_preprocessed)
            fpn_features = None

        # Handle dict output from YOLOXGrasp (with grasp head)
        if isinstance(outputs, dict):
            det_outputs = outputs.get('det_outputs', outputs)
        else:
            det_outputs = outputs

        # Postprocess
        det_outputs = self.postprocess(
            det_outputs,
            self.num_classes,
            self.conf_thresh,
            self.nms_thresh,
        )

        # Parse outputs
        boxes, scores, class_ids = self._parse_outputs(det_outputs[0], (H, W))

        if return_features:
            return boxes, scores, class_ids, fpn_features
        return boxes, scores, class_ids

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for YOLOX.

        Applies the same preprocessing as ByteTrack:
        1. Resize with aspect ratio preservation and padding
        2. Apply ImageNet normalization (mean/std)

        Args:
            image: Input image [B, C, H, W] normalized to [0, 1]

        Returns:
            Preprocessed image ready for YOLOX inference
        """
        B, C, H, W = image.shape
        target_H, target_W = self.input_size

        # Calculate scale to preserve aspect ratio (like ByteTrack)
        r = min(target_H / H, target_W / W)
        new_H = int(H * r)
        new_W = int(W * r)

        # Resize preserving aspect ratio
        if (H, W) != (new_H, new_W):
            image_resized = torch.nn.functional.interpolate(
                image,
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False,
            )
        else:
            image_resized = image

        # Pad to target size with 114/255 (ByteTrack convention)
        pad_value = 114.0 / 255.0  # Normalized pad value
        padded = torch.full(
            (B, C, target_H, target_W),
            pad_value,
            device=image.device,
            dtype=image.dtype,
        )
        padded[:, :, :new_H, :new_W] = image_resized

        # Apply ImageNet normalization (same as ByteTrack training)
        # Note: ByteTrack uses BGR order and specific mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        normalized = (padded - mean) / std

        return normalized.to(self.device)

    def _get_fpn_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract FPN features from backbone."""
        # Access backbone directly
        fpn_outs = self.model.backbone(image)

        return {
            'P3': fpn_outs[0],  # stride 8
            'P4': fpn_outs[1],  # stride 16
            'P5': fpn_outs[2],  # stride 32
        }

    def _parse_outputs(
        self,
        output: Optional[torch.Tensor],
        original_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse YOLOX output tensor.

        Args:
            output: YOLOX output [N, 7] or None
            original_size: Original image size (H, W)

        Returns:
            boxes: [N, 4] as (x1, y1, x2, y2)
            scores: [N]
            class_ids: [N]
        """
        if output is None or len(output) == 0:
            return (
                torch.zeros((0, 4), device=self.device),
                torch.zeros((0,), device=self.device),
                torch.zeros((0,), dtype=torch.long, device=self.device),
            )

        # YOLOX output format: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
        boxes = output[:, :4].clone()
        scores = output[:, 4] * output[:, 5]  # obj_conf * cls_conf
        class_ids = output[:, 6].long()

        # Scale boxes back to original size
        # Account for aspect-ratio-preserving resize + padding
        orig_H, orig_W = original_size
        target_H, target_W = self.input_size

        # Same ratio calculation as in _preprocess
        r = min(target_H / orig_H, target_W / orig_W)

        # Boxes are in input_size coordinates, scale back to original
        # Only the actual resized area (not padding) contains valid detections
        boxes[:, [0, 2]] /= r
        boxes[:, [1, 3]] /= r

        # Clip to original image bounds
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=orig_W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=orig_H)

        return boxes, scores, class_ids

    def get_fpn_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get FPN features without detection.

        Args:
            image: Input image [B, C, H, W]

        Returns:
            Dict with 'P3', 'P4', 'P5' features
        """
        if not self.available:
            return {}

        image_preprocessed = self._preprocess(image)
        return self._get_fpn_features(image_preprocessed)


class YOLOXGraspBridge(YOLOXBridge):
    """
    Extended YOLOX bridge that includes grasp dense head output.

    For models trained with GraspDenseHead (ByteTrack-Grasp).
    """

    @torch.no_grad()
    def detect_with_grasp(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Detect objects and get dense grasp predictions.

        Returns:
            boxes: Detection boxes [N, 4]
            scores: Confidence scores [N]
            class_ids: Class IDs [N]
            grasp_outputs: Dict with grasp predictions
        """
        if not self.available:
            empty = torch.zeros((0, 4), device=self.device)
            return empty, torch.zeros((0,)), torch.zeros((0,), dtype=torch.long), {}

        if image.dim() == 3:
            image = image.unsqueeze(0)

        B, C, H, W = image.shape
        image_preprocessed = self._preprocess(image)

        # Forward with grasp head
        outputs = self.model(image_preprocessed)

        # Check if model has grasp head
        if hasattr(outputs, 'keys'):
            # Dict output from YOLOXGrasp
            det_outputs = outputs.get('det_outputs')
            grasp_outputs = outputs.get('grasp_outputs', {})
        else:
            # Regular YOLOX output
            det_outputs = outputs
            grasp_outputs = {}

        # Postprocess detections
        det_outputs = self.postprocess(
            det_outputs if not isinstance(det_outputs, dict) else det_outputs,
            self.num_classes,
            self.conf_thresh,
            self.nms_thresh,
        )

        boxes, scores, class_ids = self._parse_outputs(det_outputs[0], (H, W))

        return boxes, scores, class_ids, grasp_outputs


def build_yolox_bridge(
    exp_file: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    device: str = "cuda",
    with_grasp: bool = False,
    **kwargs
) -> Union[YOLOXBridge, YOLOXGraspBridge]:
    """
    Build YOLOX bridge.

    Args:
        exp_file: Path to experiment config
        ckpt_path: Path to checkpoint
        device: Device string
        with_grasp: If True, use YOLOXGraspBridge

    Returns:
        YOLOX bridge instance
    """
    if with_grasp:
        return YOLOXGraspBridge(exp_file, ckpt_path, device, **kwargs)
    return YOLOXBridge(exp_file, ckpt_path, device, **kwargs)
