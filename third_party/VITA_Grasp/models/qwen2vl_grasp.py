"""
Qwen2VL-Grasp: Qwen2-VL ViT encoder + Grasp Decoder for 4-DOF grasp prediction.

Architecture:
    Image (H, W, 3) -> Qwen2-VL ViT (~675M) -> Visual Features -> Grasp Decoder -> 5-channel output

Output channels:
    - Channel 0: Grasp position heatmap (0-1)
    - Channel 1: Angle cos(2θ) (-1 to 1)
    - Channel 2: Angle sin(2θ) (-1 to 1)
    - Channel 3: Grasp width (0-1)
    - Channel 4: Semantic mask (0-1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import warnings

from .grasp_decoder import GraspDecoder


class Qwen2VLGrasp(nn.Module):
    """
    Qwen2-VL ViT + Grasp Decoder for single-frame grasp prediction.
    No LLM module, no temporal module.
    """

    def __init__(
        self,
        pretrained_model: str = "Qwen/Qwen2-VL-2B-Instruct",
        freeze_vit: bool = True,
        vit_lr_scale: float = 0.1,
        decoder_channels: int = 256,
        num_decoder_layers: int = 4,
        output_size: Tuple[int, int] = (480, 640),
        use_flash_attn: bool = True,
    ):
        """
        Args:
            pretrained_model: Qwen2-VL model path or HuggingFace model ID
            freeze_vit: Whether to freeze ViT parameters
            vit_lr_scale: Learning rate scale for ViT (used when not frozen)
            decoder_channels: Hidden channels in grasp decoder
            num_decoder_layers: Number of upsampling layers in decoder
            output_size: Output mask size (H, W)
            use_flash_attn: Whether to use flash attention
        """
        super().__init__()

        self.pretrained_model = pretrained_model
        self.freeze_vit = freeze_vit
        self.vit_lr_scale = vit_lr_scale
        self.output_size = output_size

        # Load Qwen2-VL ViT encoder
        self.vit, self.vit_hidden_dim = self._load_qwen2vl_vit(
            pretrained_model, use_flash_attn
        )

        # Freeze ViT if specified
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()

        # Grasp Decoder: visual features -> 5-channel grasp mask
        self.grasp_decoder = GraspDecoder(
            in_channels=self.vit_hidden_dim,
            hidden_channels=decoder_channels,
            out_channels=5,  # pos, cos, sin, width, semantic
            num_layers=num_decoder_layers,
            output_size=output_size,
        )

        # Image processor (will be set during model loading)
        self.image_processor = None

    def _load_qwen2vl_vit(
        self, model_path: str, use_flash_attn: bool
    ) -> Tuple[nn.Module, int]:
        """
        Load Qwen2-VL visual encoder only (no LLM).

        Returns:
            vit: Visual encoder module
            hidden_dim: ViT hidden dimension
        """
        try:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        except ImportError:
            raise ImportError(
                "Please install transformers>=4.37.0: pip install transformers>=4.37.0"
            )

        print(f"Loading Qwen2-VL ViT from {model_path}...")

        # Load full model first, then extract ViT
        attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        try:
            full_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",  # Load to CPU first
                attn_implementation=attn_impl,
            )
        except Exception as e:
            warnings.warn(f"Flash attention not available, using eager: {e}")
            full_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )

        # Extract visual encoder
        vit = full_model.visual
        hidden_dim = full_model.config.vision_config.embed_dim

        # Store processor for image preprocessing
        self.image_processor = Qwen2VLProcessor.from_pretrained(model_path)

        # Delete LLM parts to save memory
        del full_model.model
        del full_model.lm_head
        del full_model
        torch.cuda.empty_cache()

        print(f"ViT loaded. Hidden dim: {hidden_dim}")
        return vit, hidden_dim

    def extract_visual_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract visual features from Qwen2-VL ViT.

        Args:
            pixel_values: Preprocessed image tensor [B, C, H, W] or [B, N, C, H, W]
            image_grid_thw: Grid size tensor for dynamic resolution

        Returns:
            features: Visual features [B, num_patches, hidden_dim]
        """
        if self.freeze_vit:
            self.vit.eval()
            with torch.no_grad():
                features = self.vit(pixel_values, grid_thw=image_grid_thw)
        else:
            features = self.vit(pixel_values, grid_thw=image_grid_thw)

        return features

    def forward(
        self,
        images: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for grasp prediction.

        Args:
            images: Input images [B, C, H, W] (preprocessed)
            image_grid_thw: Optional grid size for Qwen2-VL

        Returns:
            dict with:
                - pred_masks: 5-channel prediction [B, 5, H, W]
                - pred_pos: Position heatmap [B, 1, H, W]
                - pred_cos: Angle cos [B, 1, H, W]
                - pred_sin: Angle sin [B, 1, H, W]
                - pred_width: Width map [B, 1, H, W]
                - pred_semantic: Semantic mask [B, 1, H, W]
        """
        # Extract visual features from ViT
        visual_features = self.extract_visual_features(images, image_grid_thw)

        # Decode to grasp prediction
        pred_masks = self.grasp_decoder(visual_features)  # [B, 5, H, W]

        # Split channels
        return {
            'pred_masks': pred_masks,
            'pred_pos': pred_masks[:, 0:1],
            'pred_cos': pred_masks[:, 1:2],
            'pred_sin': pred_masks[:, 2:3],
            'pred_width': pred_masks[:, 3:4],
            'pred_semantic': pred_masks[:, 4:5],
        }

    def predict_grasp(
        self,
        images: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict grasp poses from images.

        Args:
            images: Input images [B, C, H, W]
            image_grid_thw: Optional grid size
            threshold: Threshold for position heatmap

        Returns:
            dict with grasp predictions and decoded 4-DOF poses
        """
        outputs = self.forward(images, image_grid_thw)

        # Apply activations
        pred_pos = torch.sigmoid(outputs['pred_pos'])
        pred_cos = torch.tanh(outputs['pred_cos'])
        pred_sin = torch.tanh(outputs['pred_sin'])
        pred_width = torch.sigmoid(outputs['pred_width'])
        pred_semantic = torch.sigmoid(outputs['pred_semantic'])

        # Decode angle: θ = 0.5 * atan2(sin, cos)
        pred_angle = 0.5 * torch.atan2(pred_sin, pred_cos)

        # Find peak positions
        batch_size = images.shape[0]
        grasp_poses = []

        for b in range(batch_size):
            pos_map = pred_pos[b, 0]

            # Find max position
            max_val = pos_map.max()
            if max_val > threshold:
                max_idx = (pos_map == max_val).nonzero()[0]
                y, x = max_idx[0].item(), max_idx[1].item()

                angle = pred_angle[b, 0, y, x].item()
                width = pred_width[b, 0, y, x].item()

                grasp_poses.append({
                    'x': x,
                    'y': y,
                    'angle': angle,
                    'width': width,
                    'confidence': max_val.item(),
                })
            else:
                grasp_poses.append(None)

        outputs.update({
            'pred_pos_sigmoid': pred_pos,
            'pred_cos_tanh': pred_cos,
            'pred_sin_tanh': pred_sin,
            'pred_width_sigmoid': pred_width,
            'pred_semantic_sigmoid': pred_semantic,
            'pred_angle': pred_angle,
            'grasp_poses': grasp_poses,
        })

        return outputs

    def get_param_groups(self, base_lr: float) -> list:
        """
        Get parameter groups with different learning rates.

        Args:
            base_lr: Base learning rate for decoder

        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []

        # Decoder parameters with base learning rate
        decoder_params = list(self.grasp_decoder.parameters())
        param_groups.append({
            'params': decoder_params,
            'lr': base_lr,
            'name': 'decoder',
        })

        # ViT parameters with scaled learning rate (if not frozen)
        if not self.freeze_vit:
            vit_params = list(self.vit.parameters())
            param_groups.append({
                'params': vit_params,
                'lr': base_lr * self.vit_lr_scale,
                'name': 'vit',
            })

        return param_groups

    @property
    def device(self):
        return next(self.parameters()).device

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def build_qwen2vl_grasp(
    pretrained_model: str = "Qwen/Qwen2-VL-2B-Instruct",
    freeze_vit: bool = True,
    **kwargs
) -> Qwen2VLGrasp:
    """
    Build Qwen2VL-Grasp model.

    Args:
        pretrained_model: Qwen2-VL model path
        freeze_vit: Whether to freeze ViT
        **kwargs: Additional arguments for Qwen2VLGrasp

    Returns:
        Qwen2VLGrasp model
    """
    model = Qwen2VLGrasp(
        pretrained_model=pretrained_model,
        freeze_vit=freeze_vit,
        **kwargs
    )

    total_params = model.num_parameters()
    trainable_params = model.num_parameters(trainable_only=True)

    print(f"Model built:")
    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")

    return model
