"""
YOLOX + Qwen2VL-Grasp Instance-Level Grasp Prediction Pipeline.

Combines YOLOX object detection with Qwen2VL visual features for
per-instance grasp prediction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .yolox_bridge import YOLOXBridge, build_yolox_bridge
from .roi_extractor import ROIExtractor
from .instance_grasp_head import InstanceGraspHead, LightweightInstanceGraspHead


class YOLOXInstanceGrasp(nn.Module):
    """
    YOLOX + Qwen2VL-Grasp Instance-Level Grasp Prediction.

    Pipeline (Mode A - ROI-Crop):
        1. YOLOX detects objects in the image
        2. Qwen2VL ViT extracts visual features
        3. ROI features extracted for each detection
        4. Instance grasp head predicts grasp for each ROI
        5. Coordinates mapped back to original image

    Args:
        qwen_model: Qwen2VLGrasp model (provides ViT encoder)
        yolox_exp: Path to YOLOX experiment config
        yolox_ckpt: Path to YOLOX checkpoint
        mode: Pipeline mode ("roi_crop" or "shared")
        device: Device to run on
    """

    def __init__(
        self,
        qwen_model: nn.Module,
        yolox_exp: Optional[str] = None,
        yolox_ckpt: Optional[str] = None,
        mode: str = "roi_crop",
        device: str = "cuda",
        context_ratio: float = 1.5,
        roi_output_size: Tuple[int, int] = (30, 40),
        grasp_output_size: Tuple[int, int] = (120, 160),
        lightweight: bool = False,
    ):
        super().__init__()

        self.mode = mode
        self.device = device

        # Qwen2VL-Grasp model (use its ViT encoder)
        self.qwen_model = qwen_model

        # Get ViT hidden dimension
        if hasattr(qwen_model, 'vit_hidden_dim'):
            self.vit_hidden_dim = qwen_model.vit_hidden_dim
        else:
            # Default for Qwen2-VL-2B
            self.vit_hidden_dim = 1536

        # YOLOX detector
        if yolox_exp is not None:
            self.detector = build_yolox_bridge(
                exp_file=yolox_exp,
                ckpt_path=yolox_ckpt,
                device=device,
            )
        else:
            self.detector = None

        # ROI extractor
        self.roi_extractor = ROIExtractor(
            context_ratio=context_ratio,
            output_size=roi_output_size,
        )

        # Instance grasp head
        GraspHeadClass = LightweightInstanceGraspHead if lightweight else InstanceGraspHead
        self.instance_head = GraspHeadClass(
            in_channels=self.vit_hidden_dim,
            roi_size=roi_output_size,
            output_size=grasp_output_size,
        )

        # Store sizes for coordinate mapping
        self.roi_output_size = roi_output_size
        self.grasp_output_size = grasp_output_size

    def get_vit_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract ViT features from images.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            ViT features [B, N, D]
        """
        # Access ViT through qwen_model
        if hasattr(self.qwen_model, 'vit'):
            with torch.no_grad():
                return self.qwen_model.vit(images)
        elif hasattr(self.qwen_model, 'visual'):
            with torch.no_grad():
                return self.qwen_model.visual(images)
        else:
            raise AttributeError("Cannot find ViT encoder in qwen_model")

    @torch.no_grad()
    def detect(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect objects using YOLOX.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            boxes: Detection boxes [N, 4]
            scores: Confidence scores [N]
            class_ids: Class IDs [N]
        """
        if self.detector is None:
            raise RuntimeError("No detector available. Provide yolox_exp or external boxes.")
        return self.detector.detect(images)

    def map_grasp_to_original(
        self,
        grasp: Dict,
        expanded_box: torch.Tensor,
    ) -> Dict:
        """
        Map grasp coordinates from ROI space to original image space.

        Args:
            grasp: Grasp dict with x, y, angle, width, confidence
            expanded_box: Expanded ROI box [x1, y1, x2, y2]

        Returns:
            Grasp dict with original image coordinates
        """
        x1, y1, x2, y2 = expanded_box.tolist()

        roi_h, roi_w = self.grasp_output_size
        scale_x = (x2 - x1) / roi_w
        scale_y = (y2 - y1) / roi_h

        return {
            'x': grasp['x'] * scale_x + x1,
            'y': grasp['y'] * scale_y + y1,
            'angle': grasp['angle'],
            'width': grasp['width'] * scale_x,
            'confidence': grasp['confidence'],
        }

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Predict instance-level grasps.

        Args:
            images: Input images [B, C, H, W] or [C, H, W]
            boxes: Optional external detection boxes [N, 4]
            labels: Optional instance labels
            threshold: Grasp confidence threshold

        Returns:
            List of instance grasp results
        """
        # Ensure batch dimension
        if images.dim() == 3:
            images = images.unsqueeze(0)

        B, C, H, W = images.shape
        images = images.to(self.device)

        # Step 1: Detect objects (if no external boxes)
        if boxes is None:
            boxes, scores, class_ids = self.detect(images)
            if len(boxes) == 0:
                return []
        else:
            boxes = boxes.to(self.device)
            scores = torch.ones(len(boxes), device=self.device)

        # Step 2: Extract ViT features
        vit_features = self.get_vit_features(images)

        # Step 3: Expand boxes for context
        expanded_boxes = self.roi_extractor.expand_boxes(boxes, (H, W))

        # Step 4: Extract ROI features
        batch_ids = torch.zeros(len(boxes), device=self.device)
        roi_features = self.roi_extractor.extract_from_vit_sequence(
            vit_features,
            expanded_boxes,
            batch_ids,
            image_size=(H, W),
            patch_size=14,  # Qwen2-VL patch size
        )

        # Step 5: Predict grasps for each ROI
        predictions = self.instance_head(roi_features)

        # Step 6: Decode best grasp for each ROI
        grasps = self.instance_head.decode_best_grasp(predictions, threshold)

        # Step 7: Map back to original coordinates and format results
        results = []
        for i, (box, grasp) in enumerate(zip(boxes, grasps)):
            result = {
                'instance_id': i,
                'box': box.cpu().tolist(),
                'label': labels[i] if labels else f"object_{i}",
            }

            if grasp is not None:
                # Map to original coordinates
                mapped_grasp = self.map_grasp_to_original(grasp, expanded_boxes[i])
                result.update({
                    'x': mapped_grasp['x'],
                    'y': mapped_grasp['y'],
                    'angle': mapped_grasp['angle'],
                    'width': mapped_grasp['width'],
                    'confidence': mapped_grasp['confidence'],
                })
            else:
                result['x'] = None
                result['y'] = None
                result['angle'] = None
                result['width'] = None
                result['confidence'] = 0.0

            results.append(result)

        return results

    def predict_with_dense_output(
        self,
        images: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Dict], List[Dict[str, torch.Tensor]]]:
        """
        Predict with full dense output maps.

        Returns both decoded grasps and raw prediction maps for visualization.

        Args:
            images: Input images
            boxes: Optional external boxes

        Returns:
            results: List of instance results
            dense_outputs: List of prediction dicts per instance
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)

        B, C, H, W = images.shape
        images = images.to(self.device)

        if boxes is None:
            boxes, scores, _ = self.detect(images)
            if len(boxes) == 0:
                return [], []
        else:
            boxes = boxes.to(self.device)

        vit_features = self.get_vit_features(images)
        expanded_boxes = self.roi_extractor.expand_boxes(boxes, (H, W))
        batch_ids = torch.zeros(len(boxes), device=self.device)

        roi_features = self.roi_extractor.extract_from_vit_sequence(
            vit_features,
            expanded_boxes,
            batch_ids,
            image_size=(H, W),
            patch_size=14,
        )

        predictions = self.instance_head(roi_features)
        grasps = self.instance_head.decode_best_grasp(predictions, threshold=0.5)

        # Format results
        results = []
        dense_outputs = []

        for i, (box, grasp) in enumerate(zip(boxes, grasps)):
            result = {
                'instance_id': i,
                'box': box.cpu().tolist(),
            }

            if grasp is not None:
                mapped = self.map_grasp_to_original(grasp, expanded_boxes[i])
                result.update(mapped)

            results.append(result)

            # Extract dense output for this instance
            dense_outputs.append({
                'pred_pos': torch.sigmoid(predictions['pred_pos'][i:i+1]),
                'pred_cos': torch.tanh(predictions['pred_cos'][i:i+1]),
                'pred_sin': torch.tanh(predictions['pred_sin'][i:i+1]),
                'pred_width': torch.sigmoid(predictions['pred_width'][i:i+1]),
                'pred_semantic': torch.sigmoid(predictions['pred_semantic'][i:i+1]),
                'expanded_box': expanded_boxes[i].cpu(),
            })

        return results, dense_outputs

    def forward(
        self,
        images: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict]]:
        """
        Forward pass.

        In training mode: returns loss dict
        In eval mode: returns predictions

        Args:
            images: Input images [B, C, H, W]
            boxes: Detection boxes [N, 4]
            targets: Training targets dict

        Returns:
            Training: Dict with losses
            Eval: List of instance predictions
        """
        if self.training:
            return self._forward_train(images, boxes, targets)
        else:
            return self.predict(images, boxes)

    def _forward_train(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass."""
        B, C, H, W = images.shape

        # Get features (may require grad for fine-tuning)
        if hasattr(self.qwen_model, 'vit'):
            vit_features = self.qwen_model.vit(images)
        else:
            vit_features = self.qwen_model.visual(images)

        # Expand boxes
        expanded_boxes = self.roi_extractor.expand_boxes(boxes, (H, W))

        # Extract ROI features
        batch_ids = targets.get('batch_ids', torch.zeros(len(boxes), device=images.device))
        roi_features = self.roi_extractor.extract_from_vit_sequence(
            vit_features,
            expanded_boxes,
            batch_ids,
            image_size=(H, W),
            patch_size=14,
        )

        # Predict
        predictions = self.instance_head(roi_features)

        # Compute loss (loss function should be attached externally)
        return predictions


def build_yolox_instance_grasp(
    qwen_checkpoint: str,
    yolox_exp: Optional[str] = None,
    yolox_ckpt: Optional[str] = None,
    device: str = "cuda",
    lightweight: bool = False,
) -> YOLOXInstanceGrasp:
    """
    Build YOLOX Instance Grasp pipeline.

    Args:
        qwen_checkpoint: Path to Qwen2VL-Grasp checkpoint
        yolox_exp: Path to YOLOX experiment config
        yolox_ckpt: Path to YOLOX checkpoint
        device: Device string
        lightweight: Use lightweight grasp head

    Returns:
        YOLOXInstanceGrasp model
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from qwen2vl_grasp import Qwen2VLGrasp

    # Load Qwen2VL-Grasp checkpoint
    checkpoint = torch.load(qwen_checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})

    qwen_model = Qwen2VLGrasp(
        pretrained_model=config.get('pretrained_model', 'Qwen/Qwen2-VL-2B-Instruct'),
        freeze_vit=True,
        decoder_channels=config.get('decoder_channels', 256),
        output_size=tuple(config.get('output_size', [480, 640])),
    )
    qwen_model.load_state_dict(checkpoint['model_state_dict'])
    qwen_model.to(device)
    qwen_model.eval()

    # Build pipeline
    pipeline = YOLOXInstanceGrasp(
        qwen_model=qwen_model,
        yolox_exp=yolox_exp,
        yolox_ckpt=yolox_ckpt,
        device=device,
        lightweight=lightweight,
    )
    pipeline.to(device)

    return pipeline
