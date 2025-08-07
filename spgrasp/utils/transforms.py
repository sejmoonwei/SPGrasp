# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, ToTensor
from torchvision.transforms.functional import InterpolationMode, resize, pad


class _ResizePad(nn.Module):
    """Custom transform to scale with aspect ratio and pad to a square."""

    def __init__(self, size, interpolation=InterpolationMode.NEAREST):
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, img):
        # Get original dimensions (C, H, W)
        orig_h, orig_w = img.shape[-2], img.shape[-1]

        # Calculate scaling factor
        scale = self.size / max(orig_h, orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        # Scale while maintaining aspect ratio
        img = resize(
            img,
            size=[new_h, new_w],
            interpolation=self.interpolation,
            antialias=False
        )

        # Calculate padding parameters (left, top, right, bottom)
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        padding = [
            pad_w // 2,  # left
            pad_h // 2,  # top
            pad_w // 2,  # right
            pad_h // 2,
        ]

        # Pad with zeros to make it a square
        return pad(img, padding, 0)


class SAM2Transforms(nn.Module):
    def __init__(
            self,
            resolution,
            mask_threshold,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
            interpolation=InterpolationMode.NEAREST  # Added interpolation mode parameter
    ):
        super().__init__()
        self.dataset_mode = "ocid"  # Default mode
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()

        # Use custom scale-and-pad layer
        self.transforms = nn.Sequential(
            _ResizePad(resolution, interpolation),
            Normalize(self.mean, self.std)
        )

    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)

    def forward_batch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch

    def transform_coords(
        self, coords: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 model.
        """
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        coords = coords * self.resolution  # unnormalize coords
        return coords

    def transform_boxes(
        self, boxes: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a tensor of shape Bx4. The coordinates can be in absolute image or normalized coordinates,
        if the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
        """
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes
    @staticmethod
    def process_channels_separately(input_masks):
        """
        Applies activation functions to each channel separately.
        Args:
            input_masks: An input tensor of shape [1, 5, 128, 128].

        Returns:
            A processed tensor of the same shape.
        """
        # Create a mapping of activation functions (modify as needed)
        activation_map = {
            0: torch.sigmoid,
            1: torch.tanh,
            2: torch.tanh,  # Changed activation for grasp width channel to ReLU
            3: torch.relu,  # Note: This resolves the conflict for channel 3 in the original requirement
            4: torch.sigmoid
        }

        processed = torch.empty_like(input_masks)

        # Process each channel individually
        for channel in range(input_masks.size(1)):
            func = activation_map.get(channel, lambda x: x)  # Default: no processing
            processed[:, channel] = func(input_masks[:, channel])

        return processed

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        from spgrasp.utils.misc import get_connected_components

        masks = masks.float()
        #for Jacquard:

        #for OCID 480,640
        masks = self.process_channels_separately(masks)
        input_masks = masks #1，5，128，128


        mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image
        try:
            if self.max_hole_area > 0:
                # Holes are those connected components in background with area <= self.fill_hole_area
                # (background regions are those with mask scores <= self.mask_threshold)
                labels, areas = get_connected_components(
                    mask_flat <= self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with a small positive mask score (10.0) to change them to foreground.
                masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

            if self.max_sprinkle_area > 0:
                labels, areas = get_connected_components(
                    mask_flat > self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with negative mask score (-10.0) to change them to background.
                masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        except Exception as e:
            # Skip the post-processing step if the CUDA kernel fails
            warnings.warn(
                f"{e}\n\nSkipping the post-processing step due to the error above. You can "
                "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
                "functionality may be limited (which doesn't affect the results in most cases; see "
                "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
                category=UserWarning,
                stacklevel=2,
            )
            masks = input_masks

        masks = self.postprocess_mask(masks, orig_hw)

        return masks

    # The correct post-processing flow should include the following steps:
    def postprocess_mask(self, masks, original_size):
        """
        Processing flow:
        1. Upsample to the padded size.
        2. Crop according to the dataset mode.
        3. Scale to the original image size.
        """
        if self.dataset_mode == "ocid":
            # OCID specific logic
            # Step 1: Upsample to padded size (OCID specific)
            upsampled = F.interpolate(masks, (640, 640), mode="nearest")
    
            # Step 2: Crop the valid region (assuming vertical padding during training)
            # Calculate padding: original height 480 -> padded to 640, requires (640-480)//2 = 80 padding on top and bottom
            h_start = 80
            h_end = 80 + 480
            cropped = upsampled[:, :, h_start:h_end, :]  # Crop the height dimension
    
            # Step 3: Scale to original size
            final_mask = F.interpolate(cropped, original_size, mode="nearest")

        elif self.dataset_mode == "jacquard":
            # Jacquard specific logic (and a good general default)
            # Jacquard images are 1024x1024, and SAM input is 1024x1024.
            # No complex cropping is needed, just direct upsampling.
            final_mask = F.interpolate(masks, size=original_size, mode="bilinear", align_corners=False)
        
        else:
            # Default fallback (same as jacquard)
            warnings.warn(f"Unknown dataset_mode '{self.dataset_mode}'. Falling back to default upsampling.")
            final_mask = F.interpolate(masks, size=original_size, mode="bilinear", align_corners=False)


        return final_mask