"""
InternViT-Grasp: InternViT encoder + Grasp Decoder for 4-DOF grasp prediction.

Architecture (ByteTrack-style aligned):
    Input: Square image (640x640) with aspect ratio preserved + padding
    -> Resize to 448x448 for ViT (square-to-square, uniform scaling!)
    -> InternViT-300M (~300M) -> Visual Features
    -> Grasp Decoder -> Output at stride=4 (160x160, like ByteTrack)

Key insight for alignment:
    - Dataset outputs 640x640 input + 160x160 GT (stride=4, like ByteTrack)
    - ViT requires 448x448, but 640->448 is UNIFORM scaling (both are squares!)
    - Decoder outputs at stride=4 (160x160) to match GT efficiently
    - This is 16x faster than full resolution (160x160 vs 640x640)

Output channels (at stride=4, e.g., 160x160):
    - Channel 0: Grasp position heatmap (0-1)
    - Channel 1: Angle cos(2*theta) (-1 to 1)
    - Channel 2: Angle sin(2*theta) (-1 to 1)
    - Channel 3: Grasp width (0-1)
    - Channel 4: Semantic mask (0-1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .grasp_decoder import GraspDecoder


class InternViTGrasp(nn.Module):
    """
    InternViT + Grasp Decoder for single-frame grasp prediction.

    Uses ByteTrack-style input processing:
    - Input: 640x640 square image (aspect-ratio preserved + padding)
    - ViT: 448x448 (uniform downscale from 640)
    - Output: 160x160 at stride=4 (same as GT, like ByteTrack)
    """

    def __init__(
        self,
        pretrained_model: str = "OpenGVLab/InternViT-300M-448px",
        freeze_vit: bool = True,
        vit_lr_scale: float = 0.1,
        decoder_channels: int = 256,
        num_decoder_layers: int = 4,
        input_size: Tuple[int, int] = (640, 640),  # ByteTrack-style square input
        output_stride: int = 4,  # Output stride (like ByteTrack), output_size = input_size / stride
        vit_size: Tuple[int, int] = (448, 448),    # InternViT fixed size
    ):
        """
        Args:
            pretrained_model: InternViT model path or HuggingFace model ID
            freeze_vit: Whether to freeze ViT parameters
            vit_lr_scale: Learning rate scale for ViT (used when not frozen)
            decoder_channels: Hidden channels in grasp decoder
            num_decoder_layers: Number of upsampling layers in decoder
            input_size: Input image size (H, W), should be square
            output_stride: Output stride for grasp prediction (default: 4, like ByteTrack)
                          Output size = input_size / output_stride
            vit_size: ViT input size (H, W), fixed to 448x448 for InternViT-300M
        """
        super().__init__()

        self.pretrained_model = pretrained_model
        self.freeze_vit = freeze_vit
        self.vit_lr_scale = vit_lr_scale
        self.input_size = input_size    # Model input size (640x640)
        self.output_stride = output_stride
        self.output_size = (input_size[0] // output_stride, input_size[1] // output_stride)
        self.vit_size = vit_size        # ViT internal size (448x448)

        # Load InternViT encoder
        self.vit, self.vit_hidden_dim = self._load_internvit(pretrained_model)

        # Freeze ViT if specified
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()

        # Cache ViT dtype to avoid repeated queries during forward pass
        self._vit_dtype = next(self.vit.parameters()).dtype

        # Grasp Decoder: visual features -> 5-channel grasp mask
        # Output size = input_size / output_stride (e.g., 160x160 for stride=4)
        self.grasp_decoder = GraspDecoder(
            in_channels=self.vit_hidden_dim,
            hidden_channels=decoder_channels,
            out_channels=5,  # pos, cos, sin, width, semantic
            num_layers=num_decoder_layers,
            output_size=self.output_size,  # Output at stride=4 resolution
        )

        # CLIP normalization constants
        self.register_buffer('norm_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('norm_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def _load_internvit(self, model_path: str) -> Tuple[nn.Module, int]:
        """
        Load InternViT visual encoder from HuggingFace.

        Returns:
            vit: Visual encoder module
            hidden_dim: ViT hidden dimension
        """
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "Please install transformers>=4.37.0: pip install transformers>=4.37.0"
            )

        print(f"Loading InternViT from {model_path}...")

        # Try to use Flash Attention 2 for better performance
        attn_implementation = "flash_attention_2"
        try:
            vit = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
            print(f"InternViT loaded with Flash Attention 2 (optimized)")
        except Exception as e:
            print(f"Flash Attention 2 unavailable: {e}")
            print("Falling back to eager attention implementation")
            vit = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="eager",
            )

        hidden_dim = vit.config.hidden_size  # 1024 for InternViT-300M

        print(f"InternViT loaded. Hidden dim: {hidden_dim}")
        print(f"  Input size: {self.input_size} -> ViT size: {self.vit_size} -> Output size: {self.output_size} (stride={self.output_stride})")
        return vit, hidden_dim

    def preprocess_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for InternViT.

        Resizes from input_size (640x640) to vit_size (448x448).
        Since both are squares, this is UNIFORM scaling - no aspect ratio distortion!

        Args:
            images: Input images [B, C, H, W] in [0, 1] range, size=input_size

        Returns:
            Preprocessed images [B, C, 448, 448] normalized for CLIP
        """
        # Resize to ViT size (640x640 -> 448x448, uniform scaling)
        if images.shape[-2:] != self.vit_size:
            images = F.interpolate(
                images,
                size=self.vit_size,
                mode='bilinear',
                align_corners=False
            )

        # CLIP normalization
        images = (images - self.norm_mean) / self.norm_std

        return images

    def extract_visual_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from InternViT.

        Args:
            pixel_values: Preprocessed image tensor [B, C, 448, 448]

        Returns:
            features: Visual features [B, num_patches, hidden_dim]
        """
        # Convert to ViT dtype using cached dtype (avoid repeated param query)
        pixel_values = pixel_values.to(dtype=self._vit_dtype)

        # ViT forward pass (eval mode is already set in __init__ for frozen ViT)
        if self.freeze_vit:
            with torch.no_grad():
                outputs = self.vit(pixel_values)
        else:
            outputs = self.vit(pixel_values)

        # Shape: [B, num_patches + 1, hidden_dim] (includes CLS token)
        features = outputs.last_hidden_state

        # Remove CLS token
        features = features[:, 1:, :]  # [B, num_patches, hidden_dim]

        # Convert back to float32 for decoder
        features = features.float()

        return features

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for grasp prediction.

        Args:
            images: Input images [B, C, H, W] in [0, 1] range
                    Expected size: input_size (640x640)

        Returns:
            dict with (all at output_size, e.g., 160x160 for stride=4):
                - pred_masks: 5-channel prediction [B, 5, H, W]
                - pred_pos: Position heatmap [B, 1, H, W]
                - pred_cos: Angle cos [B, 1, H, W]
                - pred_sin: Angle sin [B, 1, H, W]
                - pred_width: Width map [B, 1, H, W]
                - pred_semantic: Semantic mask [B, 1, H, W]
        """
        # Preprocess: resize to 448x448 + normalize
        preprocessed = self.preprocess_image(images)

        # Extract features from ViT
        visual_features = self.extract_visual_features(preprocessed)

        # Decode to grasp prediction (output at input_size)
        pred_masks = self.grasp_decoder(visual_features)

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
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict grasp poses from images.

        Args:
            images: Input images [B, C, H, W]
            threshold: Threshold for position heatmap

        Returns:
            dict with grasp predictions and decoded 4-DOF poses
        """
        outputs = self.forward(images)

        # Apply activations
        pred_pos = torch.sigmoid(outputs['pred_pos'])
        pred_cos = torch.tanh(outputs['pred_cos'])
        pred_sin = torch.tanh(outputs['pred_sin'])
        pred_width = torch.sigmoid(outputs['pred_width'])
        pred_semantic = torch.sigmoid(outputs['pred_semantic'])

        # Decode angle: theta = 0.5 * atan2(sin, cos)
        pred_angle = 0.5 * torch.atan2(pred_sin, pred_cos)

        # Find peak positions
        batch_size = images.shape[0]
        grasp_poses = []

        for b in range(batch_size):
            pos_map = pred_pos[b, 0]

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


def build_internvit_grasp(
    pretrained_model: str = "OpenGVLab/InternViT-300M-448px",
    freeze_vit: bool = True,
    input_size: Tuple[int, int] = (640, 640),
    output_stride: int = 4,
    **kwargs
) -> InternViTGrasp:
    """
    Build InternViT-Grasp model.

    Args:
        pretrained_model: InternViT model path
        freeze_vit: Whether to freeze ViT
        input_size: Model input size (should be square)
        output_stride: Output stride for grasp prediction (default: 4, like ByteTrack)
        **kwargs: Additional arguments for InternViTGrasp

    Returns:
        InternViTGrasp model
    """
    model = InternViTGrasp(
        pretrained_model=pretrained_model,
        freeze_vit=freeze_vit,
        input_size=input_size,
        output_stride=output_stride,
        **kwargs
    )

    total_params = model.num_parameters()
    trainable_params = model.num_parameters(trainable_only=True)
    output_size = (input_size[0] // output_stride, input_size[1] // output_stride)

    print(f"Model built:")
    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.1f}M")
    print(f"  Input size: {input_size}")
    print(f"  Output size: {output_size} (stride={output_stride})")
    print(f"  ViT internal size: (448, 448)")

    return model
