"""
Grasp Decoder: Decode ViT features to 5-channel grasp prediction.

Output channels:
    - Channel 0: Grasp position heatmap
    - Channel 1: Angle cos(2θ)
    - Channel 2: Angle sin(2θ)
    - Channel 3: Grasp width
    - Channel 4: Semantic mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        activation: str = 'relu',
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'none':
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class UpsampleBlock(nn.Module):
    """Upsampling block with convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        use_deconv: bool = False,
    ):
        super().__init__()

        if use_deconv:
            self.upsample = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4,
                stride=scale_factor,
                padding=1,
                bias=False
            )
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.upsample(x)))


class GraspDecoder(nn.Module):
    """
    Grasp Decoder: Transform ViT patch features to dense 5-channel grasp prediction.

    Takes flattened patch features from ViT and outputs:
        - 5-channel grasp mask at specified resolution

    Architecture:
        Patch features [B, N, D] -> Reshape [B, D, h, w] -> Upsample layers -> [B, 5, H, W]
    """

    def __init__(
        self,
        in_channels: int = 1536,  # Qwen2-VL ViT hidden dim
        hidden_channels: int = 256,
        out_channels: int = 5,
        num_layers: int = 4,
        output_size: Tuple[int, int] = (480, 640),
        patch_size: int = 14,  # Qwen2-VL patch size
    ):
        """
        Args:
            in_channels: ViT hidden dimension
            hidden_channels: Hidden channels in decoder
            out_channels: Output channels (5 for grasp)
            num_layers: Number of upsampling layers
            output_size: Output spatial size (H, W)
            patch_size: ViT patch size
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.output_size = output_size
        self.patch_size = patch_size

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels * 4),
            nn.LayerNorm(hidden_channels * 4),
            nn.GELU(),
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )

        # Upsampling layers
        self.upsample_layers = nn.ModuleList()

        current_channels = hidden_channels
        for i in range(num_layers):
            out_ch = max(hidden_channels // (2 ** (i + 1)), 64)
            self.upsample_layers.append(
                nn.Sequential(
                    UpsampleBlock(current_channels, out_ch, scale_factor=2),
                    ConvBlock(out_ch, out_ch),
                )
            )
            current_channels = out_ch

        # Unified output head for all channels (more efficient than separate heads)
        # Combines all 5 output channels into a single convolution path
        self.output_head = nn.Sequential(
            ConvBlock(current_channels, current_channels),
            nn.Conv2d(current_channels, out_channels, kernel_size=1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize decoder weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reshape_features(
        self,
        features: torch.Tensor,
        target_h: Optional[int] = None,
        target_w: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Reshape flattened patch features to spatial feature map.

        Args:
            features: Patch features [B, N, D] or [B, D, h, w]
            target_h: Target height (optional)
            target_w: Target width (optional)

        Returns:
            Spatial features [B, D, h, w]
        """
        if features.dim() == 4:
            return features

        B, N, D = features.shape

        # Infer spatial dimensions
        if target_h is not None and target_w is not None:
            h, w = target_h, target_w
        else:
            # Assume square or calculate from output size ratio
            h_out, w_out = self.output_size
            ratio = w_out / h_out

            # Find h, w such that h * w = N and w/h ≈ ratio
            h = int(math.sqrt(N / ratio))
            w = N // h

            # Handle edge cases
            if h * w != N:
                # Try to find valid factorization
                for h_try in range(int(math.sqrt(N)), 0, -1):
                    if N % h_try == 0:
                        h = h_try
                        w = N // h_try
                        break

        if h * w != N:
            # Fallback: use sqrt and pad if necessary
            h = w = int(math.sqrt(N))
            if h * w < N:
                h = w = int(math.ceil(math.sqrt(N)))

            # Pad features if needed
            if h * w > N:
                pad_size = h * w - N
                features = F.pad(features, (0, 0, 0, pad_size), mode='constant', value=0)

        # Reshape: [B, N, D] -> [B, D, h, w]
        features = features[:, :h*w, :].reshape(B, h, w, D).permute(0, 3, 1, 2)

        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode ViT features to 5-channel grasp prediction.

        Args:
            features: ViT features [B, N, D] or [B, D, h, w]

        Returns:
            pred_masks: 5-channel prediction [B, 5, H, W]
        """
        B = features.shape[0]

        # Project features if they are flattened
        if features.dim() == 3:
            features = self.input_proj(features)  # [B, N, hidden_channels]

        # Reshape to spatial
        x = self.reshape_features(features)  # [B, C, h, w]

        # Upsample
        for layer in self.upsample_layers:
            x = layer(x)

        # Final resize to output size
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(
                x,
                size=self.output_size,
                mode='bilinear',
                align_corners=False
            )

        # Apply unified output head (single pass, more efficient)
        pred_masks = self.output_head(x)  # [B, 5, H, W]

        return pred_masks


class LightweightGraspDecoder(nn.Module):
    """
    Lightweight version of GraspDecoder for faster inference.
    Uses fewer parameters with depthwise separable convolutions.
    """

    def __init__(
        self,
        in_channels: int = 1536,
        hidden_channels: int = 128,
        out_channels: int = 5,
        output_size: Tuple[int, int] = (480, 640),
    ):
        super().__init__()

        self.output_size = output_size

        # Simpler input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Single upsampling path
        self.decoder = nn.Sequential(
            # Upsample x4
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels, bias=False),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),

            # Upsample x4
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 2, 3, padding=1, groups=hidden_channels // 2, bias=False),
            nn.Conv2d(hidden_channels // 2, hidden_channels // 4, 1, bias=False),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),

            # Output
            nn.Conv2d(hidden_channels // 4, out_channels, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]

        if features.dim() == 3:
            N, D = features.shape[1], features.shape[2]
            h = w = int(math.sqrt(N))
            features = self.input_proj(features)
            features = features[:, :h*w, :].reshape(B, h, w, -1).permute(0, 3, 1, 2)

        x = self.decoder(features)

        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)

        return x
