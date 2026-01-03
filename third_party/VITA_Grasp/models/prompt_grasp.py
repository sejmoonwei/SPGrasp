"""
Prompt-based Instance Grasp Prediction for Qwen2VL-Grasp.

Adds a prompt encoder to enable point/box prompts for instance selection,
similar to SAM/SPGrasp approach.

This is Approach 2: Adding prompt encoder to the base Qwen2VL-Grasp model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from .grasp_decoder import GraspDecoder


class PositionEmbeddingRandom(nn.Module):
    """
    Random positional encoding for prompt coordinates.
    Similar to SAM's implementation.
    """

    def __init__(self, embed_dim: int = 128, scale: float = 1.0):
        super().__init__()
        self.positional_encoding_gaussian_matrix = nn.Parameter(
            scale * torch.randn((2, embed_dim)),
            requires_grad=False,
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates to positional embedding."""
        # coords: [B, N, 2] in range [0, 1]
        coords = 2 * coords - 1  # [-1, 1]
        coords = coords @ self.positional_encoding_gaussian_matrix  # [B, N, embed_dim]
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)  # [B, N, 2*embed_dim]

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for given size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device

        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1).reshape(-1, 2))
        return pe.reshape(h, w, -1).permute(2, 0, 1)  # [C, H, W]

    def forward_with_coords(
        self,
        coords: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Encode specific coordinates."""
        # coords: [B, N, 2] in pixel coordinates
        coords = coords.clone()
        coords[..., 0] = coords[..., 0] / image_size[1]  # normalize x
        coords[..., 1] = coords[..., 1] / image_size[0]  # normalize y
        return self._pe_encoding(coords)


class PromptEncoder(nn.Module):
    """
    Prompt encoder for point and box prompts.

    Encodes user prompts (points, boxes) into embeddings that can be
    used to condition the grasp decoder for instance-level prediction.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (30, 40),
        input_image_size: Tuple[int, int] = (480, 640),
        mask_in_chans: int = 16,
    ):
        """
        Args:
            embed_dim: Embedding dimension for prompts
            image_embedding_size: Size of image features (H, W)
            input_image_size: Size of input images (H, W)
            mask_in_chans: Channels for mask encoding
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size

        # Positional encoding
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Point embeddings: [positive, negative, box_corner1, box_corner2]
        self.num_point_embeddings = 4
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        ])
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # Mask encoding (for mask prompts)
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            nn.LayerNorm([mask_in_chans // 4, self.mask_input_size[0] // 2, self.mask_input_size[1] // 2]),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            nn.LayerNorm([mask_in_chans, self.mask_input_size[0] // 4, self.mask_input_size[1] // 4]),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """Get dense positional encoding for image features."""
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool = True,
    ) -> torch.Tensor:
        """
        Embed point prompts.

        Args:
            points: Point coordinates [B, N, 2]
            labels: Point labels [B, N], 1=positive, 0=negative, -1=padding

        Returns:
            Point embeddings [B, N, embed_dim]
        """
        points = points + 0.5  # Shift to center of pixel

        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        # Get positional encoding
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        # Add point type embedding
        point_embedding = torch.where(
            (labels == -1).unsqueeze(-1),
            self.not_a_point_embed.weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 0).unsqueeze(-1),
            point_embedding + self.point_embeddings[0].weight,  # negative
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 1).unsqueeze(-1),
            point_embedding + self.point_embeddings[1].weight,  # positive
            point_embedding,
        )

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Embed box prompts.

        Args:
            boxes: Box coordinates [B, N, 4] as [x1, y1, x2, y2]

        Returns:
            Box embeddings [B, N*2, embed_dim]
        """
        boxes = boxes + 0.5  # Shift to center of pixel

        # Get corners
        corners = torch.stack([
            boxes[..., :2],  # top-left
            boxes[..., 2:],  # bottom-right
        ], dim=2)  # [B, N, 2, 2]

        B, N = boxes.shape[:2]
        corners = corners.reshape(B, N * 2, 2)

        # Get positional encoding
        corner_embedding = self.pe_layer.forward_with_coords(corners, self.input_image_size)

        # Add corner type embeddings
        corner_embedding[:, 0::2] += self.point_embeddings[2].weight  # corner 1
        corner_embedding[:, 1::2] += self.point_embeddings[3].weight  # corner 2

        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Embed mask prompts.

        Args:
            masks: Mask tensor [B, 1, H, W]

        Returns:
            Mask embeddings [B, embed_dim, h, w]
        """
        return self.mask_downscaling(masks)

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts.

        Args:
            points: (coords [B, N, 2], labels [B, N])
            boxes: Box coordinates [B, N, 4]
            masks: Mask tensor [B, 1, H, W]

        Returns:
            sparse_embeddings: Point/box embeddings [B, N, embed_dim]
            dense_embeddings: Mask embeddings [B, embed_dim, h, w]
        """
        sparse_embeddings = torch.empty((1, 0, self.embed_dim), device=self._get_device())

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device


class PromptGraspDecoder(nn.Module):
    """
    Grasp decoder that incorporates prompt embeddings for instance selection.

    Architecture:
        Visual features + Prompt embeddings -> Cross-attention -> Grasp prediction
    """

    def __init__(
        self,
        in_channels: int = 1536,
        embed_dim: int = 256,
        hidden_channels: int = 256,
        num_heads: int = 8,
        output_size: Tuple[int, int] = (480, 640),
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.output_size = output_size

        # Project visual features to embed_dim
        self.visual_proj = nn.Linear(in_channels, embed_dim)

        # Cross-attention: prompts attend to visual features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Self-attention for visual features
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, embed_dim),
        )

        # Upsampling decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels // 2, 5, 1),  # 5 output channels
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        sparse_embeddings: torch.Tensor,
        dense_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with prompt conditioning.

        Args:
            visual_features: ViT features [B, N, D]
            sparse_embeddings: Point/box embeddings [B, M, embed_dim]
            dense_embeddings: Mask embeddings [B, embed_dim, h, w]

        Returns:
            pred_masks: 5-channel prediction [B, 5, H, W]
        """
        B = visual_features.shape[0]

        # Project visual features
        visual_features = self.visual_proj(visual_features)  # [B, N, embed_dim]

        # Cross-attention: prompts query visual features
        if sparse_embeddings.shape[1] > 0:
            attn_out, _ = self.cross_attn(
                sparse_embeddings,  # query
                visual_features,    # key
                visual_features,    # value
            )
            # Combine with visual features
            visual_features = visual_features + attn_out.mean(dim=1, keepdim=True)

        # Self-attention on visual features
        visual_features, _ = self.self_attn(
            visual_features, visual_features, visual_features
        )

        # Reshape to spatial
        N = visual_features.shape[1]
        h = w = int(math.sqrt(N))
        visual_features = visual_features[:, :h*w, :].reshape(B, h, w, -1).permute(0, 3, 1, 2)

        # Add dense embeddings (mask prompt)
        if dense_embeddings.shape[-2:] != visual_features.shape[-2:]:
            dense_embeddings = F.interpolate(
                dense_embeddings, size=visual_features.shape[-2:], mode='bilinear'
            )
        visual_features = visual_features + dense_embeddings

        # Decode to grasp prediction
        pred_masks = self.decoder(visual_features)

        # Final resize
        if pred_masks.shape[-2:] != self.output_size:
            pred_masks = F.interpolate(
                pred_masks, size=self.output_size, mode='bilinear', align_corners=False
            )

        return pred_masks


class Qwen2VLGraspWithPrompt(nn.Module):
    """
    Qwen2VL-Grasp with prompt encoder for instance-level prediction.

    Similar to SPGrasp, accepts point/box prompts to select target instance.
    """

    def __init__(
        self,
        pretrained_model: str = "Qwen/Qwen2-VL-2B-Instruct",
        freeze_vit: bool = True,
        embed_dim: int = 256,
        output_size: Tuple[int, int] = (480, 640),
    ):
        super().__init__()

        self.output_size = output_size
        self.embed_dim = embed_dim

        # Load Qwen2-VL ViT
        self.vit, self.vit_hidden_dim = self._load_vit(pretrained_model)

        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()

        # Prompt encoder
        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            input_image_size=output_size,
        )

        # Prompt-conditioned decoder
        self.decoder = PromptGraspDecoder(
            in_channels=self.vit_hidden_dim,
            embed_dim=embed_dim,
            output_size=output_size,
        )

    def _load_vit(self, model_path: str):
        """Load Qwen2-VL ViT encoder."""
        from transformers import Qwen2VLForConditionalGeneration

        full_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        vit = full_model.visual
        hidden_dim = full_model.config.vision_config.embed_dim

        del full_model.model
        del full_model.lm_head
        torch.cuda.empty_cache()

        return vit, hidden_dim

    def forward(
        self,
        images: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward with prompt conditioning.

        Args:
            images: Input images [B, C, H, W]
            point_coords: Point coordinates [B, N, 2]
            point_labels: Point labels [B, N]
            boxes: Box prompts [B, N, 4]
            masks: Mask prompts [B, 1, H, W]

        Returns:
            Dict with pred_masks and per-channel predictions
        """
        # Extract visual features
        with torch.no_grad() if not self.training else torch.enable_grad():
            visual_features = self.vit(images)

        # Encode prompts
        points = (point_coords, point_labels) if point_coords is not None else None
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points, boxes, masks)

        # Decode with prompt conditioning
        pred_masks = self.decoder(visual_features, sparse_embeddings, dense_embeddings)

        return {
            'pred_masks': pred_masks,
            'pred_pos': pred_masks[:, 0:1],
            'pred_cos': pred_masks[:, 1:2],
            'pred_sin': pred_masks[:, 2:3],
            'pred_width': pred_masks[:, 3:4],
            'pred_semantic': pred_masks[:, 4:5],
        }
