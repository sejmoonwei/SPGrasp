"""
GraspNet Dataset for VITA-Grasp (ByteTrack-style preprocessing).

Key design principles (aligned with ByteTrack):
1. Square input: All images resized to square (e.g., 640x640) with aspect ratio preserved
2. Uniform scaling: scale_x = scale_y = r, preserving grasp angles correctly
3. Padding: Use 114 as padding value (YOLOX convention)
4. GT alignment: Grasp GT generated using the SAME scale factor as image preprocessing

Data flow:
    Original image (720x1280)
    -> Aspect-ratio-preserving resize to square (640x640)
    -> r = min(640/720, 640/1280) = 0.5
    -> Actual size: 360x640, padded to 640x640
    -> Grasp coordinates scaled by same r
    -> Image and GT perfectly aligned!
"""

import os
import random
import logging
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preproc(image: np.ndarray, input_size: Tuple[int, int], pad_value: int = 114):
    """
    ByteTrack-style image preprocessing.

    Resize image with aspect ratio preserved, then pad to target size.

    Args:
        image: Input image [H, W, C] (BGR or RGB)
        input_size: Target size (H, W), typically square like (640, 640)
        pad_value: Padding value (default: 114, YOLOX convention)

    Returns:
        padded_img: Preprocessed image [H, W, C]
        r: Scaling factor (uniform, same for x and y)
    """
    img = np.array(image)
    orig_h, orig_w = img.shape[:2]
    target_h, target_w = input_size

    # Calculate uniform scale to preserve aspect ratio
    r = min(target_h / orig_h, target_w / orig_w)

    # Resize
    new_h, new_w = int(orig_h * r), int(orig_w * r)
    resized_img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # Create padded image
    if len(img.shape) == 3:
        padded_img = np.full((target_h, target_w, img.shape[2]), pad_value, dtype=np.uint8)
    else:
        padded_img = np.full((target_h, target_w), pad_value, dtype=np.uint8)

    # Place resized image in top-left corner
    padded_img[:new_h, :new_w] = resized_img

    return padded_img, r


def generate_grasp_gt(
    rect_data: np.ndarray,
    label_mask: np.ndarray,
    orig_size: Tuple[int, int],
    input_size: Tuple[int, int],
    output_size: Tuple[int, int],
    scale: float,
    gaussian_sigma: float = 2.0,
    max_width: float = 100.0,
) -> np.ndarray:
    """
    Generate 5-channel grasp GT using ByteTrack-style uniform scaling.

    CRITICAL: Uses stride=4 output resolution (like ByteTrack) for efficiency.
    Coordinates are scaled from original -> input_size -> output_size.

    Data flow (e.g., 720x1280 -> 640x640 -> 160x160):
        1. Original coords (720x1280)
        2. scale by r=0.5 to input coords (360x640 content in 640x640)
        3. scale by output/input to GT coords (90x160 content in 160x160)

    Args:
        rect_data: Grasp rectangles [N, 7] = [cx, cy, open_x, open_y, height, score, obj_id]
        label_mask: Instance segmentation mask [orig_H, orig_W]
        orig_size: Original image size (H, W)
        input_size: Model input size (H, W), e.g., (640, 640)
        output_size: Output GT size (H, W), e.g., (160, 160) for stride=4
        scale: Uniform scaling factor from orig to input (SAME as used in preproc!)
        gaussian_sigma: Gaussian kernel sigma for heatmap
        max_width: Maximum grasp width for normalization

    Returns:
        grasp_gt: 5-channel annotation [5, out_H, out_W]
            - Channel 0: Grasp center heatmap
            - Channel 1: cos(2*theta)
            - Channel 2: sin(2*theta)
            - Channel 3: Normalized width
            - Channel 4: Semantic mask
    """
    orig_h, orig_w = orig_size
    input_h, input_w = input_size
    out_h, out_w = output_size

    # Total scale from original to output
    # scale: orig -> input, then output/input for input -> output
    total_scale = scale * (out_h / input_h)  # Assuming square (out_h/input_h == out_w/input_w)

    # Content size in output coordinates
    content_h = int(orig_h * total_scale)
    content_w = int(orig_w * total_scale)

    # Clamp to output size
    content_h = min(content_h, out_h)
    content_w = min(content_w, out_w)

    # Initialize GT maps
    grasp_gt = np.zeros((5, out_h, out_w), dtype=np.float32)

    # Channel 4: Semantic mask - resize directly to output size content area
    if label_mask is not None:
        label_resized = cv2.resize(
            label_mask.astype(np.float32),
            (content_w, content_h),
            interpolation=cv2.INTER_NEAREST
        )
        grasp_gt[4, :content_h, :content_w] = (label_resized > 0).astype(np.float32)

    if rect_data is None or len(rect_data) == 0:
        return grasp_gt

    # Generate grid for content area only (MUCH smaller: e.g., 90x160 vs 360x640)
    yy, xx = np.meshgrid(np.arange(content_h), np.arange(content_w), indexing='ij')
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    # Temporary maps for content area
    heatmap = np.zeros((content_h, content_w), dtype=np.float32)
    cos2theta_map = np.zeros((content_h, content_w), dtype=np.float32)
    sin2theta_map = np.zeros((content_h, content_w), dtype=np.float32)
    width_map = np.zeros((content_h, content_w), dtype=np.float32)

    for rect in rect_data:
        if len(rect) < 7:
            continue

        cx, cy, open_x, open_y, height, score, obj_id = rect[:7]

        if score < 0.1:  # Filter low-quality grasps
            continue

        # Compute grasp parameters in ORIGINAL coordinates
        dx = open_x - cx
        dy = open_y - cy
        theta = np.arctan2(dy, dx)
        width = 2 * np.sqrt(dx**2 + dy**2)

        # Scale coordinates using total_scale (orig -> output directly)
        cx_scaled = cx * total_scale
        cy_scaled = cy * total_scale
        # Width scales with input scale (for max_width normalization consistency)
        width_scaled = width * scale

        # Boundary check
        if not (0 <= cx_scaled < content_w and 0 <= cy_scaled < content_h):
            continue

        # Generate Gaussian heatmap
        dist_sq = (xx - cx_scaled) ** 2 + (yy - cy_scaled) ** 2
        gaussian = np.exp(-dist_sq / (2 * gaussian_sigma ** 2))
        response = gaussian * score

        # CRITICAL: Only update angle/width where this grasp has HIGHER response
        # This ensures angle/width correspond to the best grasp at each location
        # (Angle and width should NOT be interpolated!)
        update_mask = response > heatmap

        # Channel 0: Position heatmap (max for overlapping)
        heatmap = np.maximum(heatmap, response)

        # Angle parameters (theta is preserved since scaling is uniform!)
        # Only update where this grasp is the current best
        cos2theta = np.cos(2 * theta)
        sin2theta = np.sin(2 * theta)
        cos2theta_map[update_mask] = cos2theta
        sin2theta_map[update_mask] = sin2theta

        # Normalized width - also only update where this grasp is best
        norm_width = min(width_scaled / max_width, 1.0)
        width_map[update_mask] = norm_width

    # Copy content to output (padding area remains zero)
    grasp_gt[0, :content_h, :content_w] = heatmap
    grasp_gt[1, :content_h, :content_w] = cos2theta_map
    grasp_gt[2, :content_h, :content_w] = sin2theta_map
    grasp_gt[3, :content_h, :content_w] = width_map

    return grasp_gt


class GraspNetDataset(Dataset):
    """
    GraspNet-1Billion Dataset with ByteTrack-style preprocessing.

    Key features:
    - Square input (default 640x640) with aspect ratio preserved
    - Uniform scaling ensures grasp angles are correct
    - Perfect alignment between image and GT

    Directory structure:
        scenes/
            scene_0000/
                kinect/
                    rgb/          # 0000.png - 0255.png
                    label/        # 0000.png - 0255.png (instance segmentation)
                    rect/         # 0000.npy - 0255.npy (grasp rectangles)
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = 'train',
        camera_type: str = 'kinect',
        num_train_scenes: int = 100,
        num_val_scenes: int = 30,
        input_size: Tuple[int, int] = (640, 640),  # Square input like ByteTrack
        output_stride: int = 4,  # Output stride for GT (like ByteTrack)
        max_width: float = 100.0,
        gaussian_sigma: float = 2.0,
        augment: bool = True,
        pad_value: int = 114,
    ):
        """
        Args:
            dataset_root: Path to GraspNet scenes directory
            split: 'train' or 'val'
            camera_type: 'kinect' or 'realsense'
            num_train_scenes: Number of scenes for training
            num_val_scenes: Number of scenes for validation
            input_size: Model input size (H, W), should be square
            output_stride: GT output stride (default: 4, like ByteTrack)
                           GT size = input_size / output_stride
            max_width: Maximum grasp width for normalization
            gaussian_sigma: Gaussian sigma for heatmap generation
            augment: Whether to apply data augmentation
            pad_value: Padding value (default: 114, YOLOX convention)
        """
        super().__init__()

        self.dataset_root = dataset_root
        self.split = split
        self.camera_type = camera_type
        self.input_size = input_size
        self.output_stride = output_stride
        self.output_size = (input_size[0] // output_stride, input_size[1] // output_stride)
        self.max_width = max_width
        self.gaussian_sigma = gaussian_sigma
        self.augment = augment and split == 'train'
        self.pad_value = pad_value

        # Determine scene range
        if split == 'train':
            scene_start, scene_end = 0, num_train_scenes
        else:
            scene_start, scene_end = num_train_scenes, num_train_scenes + num_val_scenes

        # Collect samples
        self.samples = []
        self._collect_samples(scene_start, scene_end)

        logger.info(f"GraspNetDataset [{split}]: {len(self.samples)} samples, "
                   f"input_size={input_size}, output_size={self.output_size} (stride={output_stride})")

    def _collect_samples(self, scene_start: int, scene_end: int):
        """Collect all valid frame samples from scenes."""
        for scene_idx in range(scene_start, scene_end):
            scene_name = f"scene_{scene_idx:04d}"
            scene_path = os.path.join(self.dataset_root, scene_name)

            if not os.path.isdir(scene_path):
                continue

            rgb_dir = os.path.join(scene_path, self.camera_type, 'rgb')
            label_dir = os.path.join(scene_path, self.camera_type, 'label')
            rect_dir = os.path.join(scene_path, self.camera_type, 'rect')

            if not all(os.path.isdir(d) for d in [rgb_dir, label_dir, rect_dir]):
                continue

            for frame_idx in range(256):
                frame_name = f"{frame_idx:04d}"
                rgb_path = os.path.join(rgb_dir, f"{frame_name}.png")
                label_path = os.path.join(label_dir, f"{frame_name}.png")
                rect_path = os.path.join(rect_dir, f"{frame_name}.npy")

                if all(os.path.exists(p) for p in [rgb_path, label_path, rect_path]):
                    self.samples.append({
                        'scene': scene_name,
                        'frame_idx': frame_idx,
                        'rgb_path': rgb_path,
                        'label_path': label_path,
                        'rect_path': rect_path,
                    })

    def _load_grasp_rects(self, rect_path: str) -> np.ndarray:
        """Load grasp rectangles from .npy file."""
        try:
            data = np.load(rect_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.size > 0:
                return data
        except Exception as e:
            logger.warning(f"Failed to load rect: {rect_path}, error: {e}")
        return np.array([])

    def _augment(
        self,
        image: np.ndarray,
        grasp_gt: np.ndarray,
        content_size_input: Tuple[int, int],
        content_size_output: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation.

        Note: Only augments the content area, not the padding.
        Image and GT have different resolutions (e.g., 640x640 vs 160x160).

        Args:
            image: Input image [H, W, C] at input_size resolution
            grasp_gt: Grasp GT [5, H, W] at output_size resolution
            content_size_input: Actual content size (h, w) in input coordinates
            content_size_output: Actual content size (h, w) in output coordinates
        """
        content_h_in, content_w_in = content_size_input
        content_h_out, content_w_out = content_size_output

        # Random horizontal flip
        if random.random() > 0.5:
            # Flip image (full resolution)
            image = np.fliplr(image).copy()

            # Flip GT (lower resolution) - need to handle content vs padding carefully
            grasp_gt = np.flip(grasp_gt, axis=2).copy()

            # After flip, content is on the RIGHT side, need to shift to LEFT
            # Create new GT with content on left
            new_gt = np.zeros_like(grasp_gt)
            out_h, out_w = grasp_gt.shape[1], grasp_gt.shape[2]

            # The flipped content is now at [out_w - content_w_out : out_w]
            # Move it back to [0 : content_w_out]
            new_gt[:, :content_h_out, :content_w_out] = grasp_gt[:, :content_h_out, (out_w - content_w_out):out_w]

            # Flip angle: cos(2θ) stays same, sin(2θ) flips sign
            new_gt[2] = -new_gt[2]

            grasp_gt = new_gt

        # Random color jitter (only affects image, not GT)
        if random.random() > 0.5:
            # Brightness
            beta = random.uniform(-32, 32)
            image = np.clip(image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        if random.random() > 0.5:
            # Contrast
            alpha = random.uniform(0.5, 1.5)
            image = np.clip(image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        return image, grasp_gt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load RGB image (BGR -> RGB)
        image = cv2.imread(sample['rgb_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # ByteTrack-style preprocessing: aspect-ratio-preserving resize + padding
        image, r = preproc(image, self.input_size, self.pad_value)

        # Calculate content size in input coordinates (before padding)
        content_h_in = int(orig_h * r)
        content_w_in = int(orig_w * r)

        # Calculate content size in output coordinates (stride=4)
        total_scale = r / self.output_stride
        content_h_out = int(orig_h * total_scale)
        content_w_out = int(orig_w * total_scale)
        # Clamp to output size
        content_h_out = min(content_h_out, self.output_size[0])
        content_w_out = min(content_w_out, self.output_size[1])

        # Load instance segmentation mask
        label_mask = cv2.imread(sample['label_path'], cv2.IMREAD_UNCHANGED)
        if label_mask is None:
            label_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # Load grasp rectangles
        rects = self._load_grasp_rects(sample['rect_path'])

        # Generate grasp GT at output resolution (stride=4)
        grasp_gt = generate_grasp_gt(
            rect_data=rects,
            label_mask=label_mask,
            orig_size=(orig_h, orig_w),
            input_size=self.input_size,
            output_size=self.output_size,  # e.g., (160, 160) for stride=4
            scale=r,  # CRITICAL: Same scale as image preprocessing!
            gaussian_sigma=self.gaussian_sigma,
            max_width=self.max_width,
        )

        # Data augmentation (handles different resolutions)
        if self.augment:
            image, grasp_gt = self._augment(
                image, grasp_gt,
                (content_h_in, content_w_in),
                (content_h_out, content_w_out)
            )

        # Convert to tensor
        # Image: [H, W, C] -> [C, H, W], normalize to [0, 1]
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        grasp_gt = torch.from_numpy(grasp_gt).float()

        return {
            'image': image,
            'grasp_mask': grasp_gt,
            'scene': sample['scene'],
            'frame_idx': sample['frame_idx'],
            'scale': r,  # Return scale for reference
        }


def build_graspnet_dataloader(
    dataset_root: str,
    split: str = 'train',
    batch_size: int = 4,
    num_workers: int = 8,  # Number of data loading workers (increased for better throughput)
    input_size: Tuple[int, int] = (640, 640),
    output_stride: int = 4,  # GT output stride (like ByteTrack)
    is_distributed: bool = False,
    prefetch_factor: int = 2,  # Prefetch batches for smoother training
    persistent_workers: bool = False,  # Disabled for DDP compatibility
    **kwargs
) -> DataLoader:
    """
    Build GraspNet dataloader with optimized settings for faster training.

    Args:
        dataset_root: Path to GraspNet scenes directory
        split: 'train' or 'val'
        batch_size: Batch size (per GPU in distributed mode)
        num_workers: Number of data loading workers (default: 8)
        input_size: Model input size (H, W)
        output_stride: GT output stride (default: 4, like ByteTrack)
        is_distributed: Whether to use distributed training
        prefetch_factor: Number of batches to prefetch per worker (default: 4)
        persistent_workers: Keep workers alive between epochs (default: True)

    Returns:
        DataLoader (and DistributedSampler if distributed)
    """
    dataset = GraspNetDataset(
        dataset_root=dataset_root,
        split=split,
        input_size=input_size,
        output_stride=output_stride,  # Use stride=4 for fast GT generation
        augment=(split == 'train'),
        **kwargs
    )

    # Create sampler for distributed training
    sampler = None
    if is_distributed:
        import torch.distributed as dist
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=(split == 'train'),
            seed=42,  # Fixed seed for reproducibility
        )

    # Only use persistent_workers and prefetch_factor when num_workers > 0
    use_persistent = persistent_workers and num_workers > 0

    # IMPORTANT: In DDP, drop_last=True for BOTH train and val to prevent sync deadlock
    # Different GPUs may have different batch counts if drop_last=False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),  # Don't shuffle if using sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_distributed or (split == 'train'),  # Drop last in DDP to prevent deadlock
        persistent_workers=use_persistent,  # Keep workers alive for faster epoch transitions
        prefetch_factor=prefetch_factor if num_workers > 0 else None,  # Prefetch batches
    )

    return loader
