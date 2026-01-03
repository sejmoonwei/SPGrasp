"""
Instance-level Grasp Prediction for Qwen2VL-Grasp.

Three approaches for instance selection:
1. Detection + ROI: Use detector (YOLOX/GroundingDINO) to get object boxes, then predict grasp per ROI
2. Prompt-based: Add prompt encoder for point/box prompts (SAM-style)
3. Language-guided: Use GroundingDINO for open-vocabulary object detection

This file implements Approach 1 (Detection + ROI) as the primary method.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .qwen2vl_grasp import Qwen2VLGrasp


class InstanceGraspPredictor(nn.Module):
    """
    Instance-level grasp prediction using Detection + ROI approach.

    Pipeline:
        1. Detect objects using external detector (YOLOX/GroundingDINO)
        2. Crop ROI regions from image
        3. Predict grasp for each ROI
        4. Map predictions back to original image coordinates
    """

    def __init__(
        self,
        grasp_model: Qwen2VLGrasp,
        roi_size: Tuple[int, int] = (224, 224),
        context_ratio: float = 1.5,  # Expand ROI by this ratio for context
    ):
        """
        Args:
            grasp_model: Qwen2VLGrasp model for grasp prediction
            roi_size: Size to resize each ROI for grasp prediction
            context_ratio: Ratio to expand bounding box for context
        """
        super().__init__()
        self.grasp_model = grasp_model
        self.roi_size = roi_size
        self.context_ratio = context_ratio

    def expand_box(
        self,
        box: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Expand bounding box for context while keeping within image bounds.

        Args:
            box: [x1, y1, x2, y2]
            image_size: (H, W)

        Returns:
            Expanded box [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Expand
        new_w = w * self.context_ratio
        new_h = h * self.context_ratio

        # Make square (use larger dimension)
        size = max(new_w, new_h)

        # New box
        new_x1 = max(0, cx - size / 2)
        new_y1 = max(0, cy - size / 2)
        new_x2 = min(image_size[1], cx + size / 2)
        new_y2 = min(image_size[0], cy + size / 2)

        return torch.tensor([new_x1, new_y1, new_x2, new_y2], device=box.device)

    def crop_roi(
        self,
        image: torch.Tensor,
        box: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Crop ROI from image and resize to roi_size.

        Args:
            image: Image tensor [C, H, W]
            box: Bounding box [x1, y1, x2, y2]

        Returns:
            roi: Cropped and resized ROI [C, roi_H, roi_W]
            info: Dict with crop info for coordinate mapping
        """
        _, H, W = image.shape
        x1, y1, x2, y2 = box.int().tolist()

        # Ensure valid bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # Crop
        roi = image[:, y1:y2, x1:x2]

        # Resize
        roi = F.interpolate(
            roi.unsqueeze(0),
            size=self.roi_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Store mapping info
        info = {
            'original_box': [x1, y1, x2, y2],
            'roi_size': self.roi_size,
            'scale_x': (x2 - x1) / self.roi_size[1],
            'scale_y': (y2 - y1) / self.roi_size[0],
            'offset_x': x1,
            'offset_y': y1,
        }

        return roi, info

    def map_grasp_to_original(
        self,
        grasp: Dict,
        info: Dict,
    ) -> Dict:
        """
        Map grasp coordinates from ROI space to original image space.

        Args:
            grasp: Grasp dict with x, y, angle, width
            info: Crop info from crop_roi

        Returns:
            Grasp dict with coordinates in original image space
        """
        return {
            'x': grasp['x'] * info['scale_x'] + info['offset_x'],
            'y': grasp['y'] * info['scale_y'] + info['offset_y'],
            'angle': grasp['angle'],
            'width': grasp['width'] * info['scale_x'],  # Scale width too
            'confidence': grasp['confidence'],
            'roi_info': info,
        }

    @torch.no_grad()
    def predict_instance_grasps(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Predict grasps for each detected instance.

        Args:
            image: Image tensor [C, H, W] or [B, C, H, W]
            boxes: Detection boxes [N, 4] as [x1, y1, x2, y2]
            labels: Optional instance labels
            threshold: Confidence threshold

        Returns:
            List of grasp predictions, one per instance
        """
        if image.dim() == 4:
            image = image.squeeze(0)

        H, W = image.shape[-2:]
        results = []

        for i, box in enumerate(boxes):
            # Expand box for context
            expanded_box = self.expand_box(box, (H, W))

            # Crop ROI
            roi, crop_info = self.crop_roi(image, expanded_box)

            # Predict grasp
            roi_batch = roi.unsqueeze(0)
            outputs = self.grasp_model.predict_grasp(roi_batch, threshold=threshold)

            # Get best grasp
            grasp_poses = outputs.get('grasp_poses', [None])[0]

            if grasp_poses is not None:
                # Map to original coordinates
                grasp = self.map_grasp_to_original(grasp_poses, crop_info)
                grasp['instance_id'] = i
                grasp['label'] = labels[i] if labels else f"object_{i}"
                grasp['box'] = box.tolist()
                results.append(grasp)
            else:
                results.append({
                    'instance_id': i,
                    'label': labels[i] if labels else f"object_{i}",
                    'box': box.tolist(),
                    'grasp': None,
                })

        return results

    def forward(
        self,
        images: torch.Tensor,
        boxes_list: List[torch.Tensor],
        labels_list: Optional[List[List[str]]] = None,
        threshold: float = 0.5,
    ) -> List[List[Dict]]:
        """
        Batch forward for instance grasp prediction.

        Args:
            images: Batch of images [B, C, H, W]
            boxes_list: List of detection boxes for each image
            labels_list: Optional list of labels for each image
            threshold: Confidence threshold

        Returns:
            List of grasp results for each image
        """
        batch_results = []

        for i, (image, boxes) in enumerate(zip(images, boxes_list)):
            labels = labels_list[i] if labels_list else None
            results = self.predict_instance_grasps(image, boxes, labels, threshold)
            batch_results.append(results)

        return batch_results


class GroundingDINOWrapper:
    """
    Wrapper for GroundingDINO to get language-guided object detection.

    Usage:
        detector = GroundingDINOWrapper()
        boxes, labels = detector.detect(image, "red cup. blue bottle.")
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.model.to(device)
            self.model.eval()
            self.available = True
        except Exception as e:
            print(f"GroundingDINO not available: {e}")
            print("Install with: pip install transformers>=4.36.0")
            self.available = False

    @torch.no_grad()
    def detect(
        self,
        image: Union[torch.Tensor, np.ndarray, "PIL.Image.Image"],
        text_prompt: str,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Detect objects matching text prompt.

        Args:
            image: Input image
            text_prompt: Text describing objects, e.g., "red cup. blue bottle."

        Returns:
            boxes: Detection boxes [N, 4] as [x1, y1, x2, y2]
            labels: List of matched labels
        """
        if not self.available:
            return torch.zeros((0, 4)), []

        from PIL import Image

        # Convert to PIL if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Process
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Post-process
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],  # (H, W)
        )[0]

        boxes = results["boxes"]  # [N, 4] in xyxy format
        labels = results["labels"]  # List of strings

        return boxes, labels


class YOLOXWrapper:
    """
    Wrapper for YOLOX detector.
    Uses the ByteTrack YOLOX implementation.
    """

    def __init__(
        self,
        exp_file: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        device: str = "cuda",
        conf_thresh: float = 0.3,
        nms_thresh: float = 0.45,
    ):
        self.device = device
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        try:
            import sys
            sys.path.insert(0, "/data/myp/sam2/sam2grasp/third_party/ByteTrack")
            from yolox.exp import get_exp
            from yolox.utils import postprocess

            if exp_file:
                self.exp = get_exp(exp_file, None)
                self.model = self.exp.get_model()
                if ckpt_path:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    self.model.load_state_dict(ckpt["model"])
                self.model.to(device)
                self.model.eval()
                self.available = True
            else:
                self.available = False
                print("YOLOX: No exp_file provided")

        except Exception as e:
            print(f"YOLOX not available: {e}")
            self.available = False

    @torch.no_grad()
    def detect(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect objects in image.

        Args:
            image: Input image tensor [C, H, W] or [B, C, H, W]

        Returns:
            boxes: Detection boxes [N, 4]
            scores: Detection scores [N]
        """
        if not self.available:
            return torch.zeros((0, 4)), torch.zeros((0,))

        if image.dim() == 3:
            image = image.unsqueeze(0)

        outputs = self.model(image.to(self.device))

        # Post-process (simplified)
        # Full implementation would use yolox postprocess
        boxes = outputs[0][:, :4] if outputs[0] is not None else torch.zeros((0, 4))
        scores = outputs[0][:, 4] if outputs[0] is not None else torch.zeros((0,))

        return boxes, scores


class LanguageGuidedGraspPredictor(nn.Module):
    """
    Language-guided instance grasp prediction.

    Combines GroundingDINO for detection with Qwen2VL-Grasp for grasp prediction.

    Usage:
        predictor = LanguageGuidedGraspPredictor(grasp_model)
        results = predictor.predict(image, "pick up the red cup")
    """

    def __init__(
        self,
        grasp_model: Qwen2VLGrasp,
        grounding_dino_model: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
    ):
        super().__init__()

        self.instance_predictor = InstanceGraspPredictor(grasp_model)
        self.detector = GroundingDINOWrapper(
            model_id=grounding_dino_model,
            device=device,
        )
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        text_prompt: str,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Predict grasps for objects matching text description.

        Args:
            image: Input image [C, H, W] or [B, C, H, W]
            text_prompt: Text describing target objects
            threshold: Grasp confidence threshold

        Returns:
            List of grasp predictions for matched objects
        """
        if image.dim() == 4:
            image = image.squeeze(0)

        # Detect objects matching text
        boxes, labels = self.detector.detect(image, text_prompt)

        if len(boxes) == 0:
            return []

        # Predict grasps for each detected instance
        results = self.instance_predictor.predict_instance_grasps(
            image, boxes, labels, threshold
        )

        return results


def build_instance_grasp_predictor(
    grasp_model: Qwen2VLGrasp,
    method: str = "grounding_dino",
    **kwargs
) -> nn.Module:
    """
    Build instance-level grasp predictor.

    Args:
        grasp_model: Qwen2VLGrasp model
        method: "grounding_dino" or "yolox"

    Returns:
        Instance grasp predictor
    """
    if method == "grounding_dino":
        return LanguageGuidedGraspPredictor(grasp_model, **kwargs)
    elif method == "yolox":
        return InstanceGraspPredictor(grasp_model, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
