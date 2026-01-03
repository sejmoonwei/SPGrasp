#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GraspNet-1Billion Dataset for ByteTrack-Grasp

Data structure:
- RGB images: 720x1280
- Instance mask (label): uint8, 0=background, others=object_id
- Grasp annotations (rect): (N, 7) = [center_x, center_y, open_x, open_y, height, score, object_id]

Output format:
- targets: (N, 10) = [cls, cx, cy, w, h, track_id, cos2theta, sin2theta, grasp_width, grasp_score]
"""

import cv2
import numpy as np
import os
import torch
from loguru import logger

from .datasets_wrapper import Dataset


def generate_grasp_dense_gt(rect_data, img_size, output_size, input_dim, gaussian_sigma=2.0):
    """
    从稀疏抓取标注生成密集GT热图

    Args:
        rect_data: (N, 7) = [center_x, center_y, open_x, open_y, height, score, object_id]
        img_size: (H, W) 原图尺寸
        output_size: (H, W) 输出热图尺寸
        input_dim: (H, W) 模型输入尺寸 (用于计算正确的缩放比例)
        gaussian_sigma: 高斯核标准差

    Returns:
        grasp_gt: dict with:
            - heatmap: (1, H, W) 抓取中心热图
            - cos2theta: (1, H, W) cos(2θ)
            - sin2theta: (1, H, W) sin(2θ)
            - width: (1, H, W) 归一化宽度
            - quality: (1, H, W) 抓取质量
    """
    img_H, img_W = img_size
    out_H, out_W = output_size
    input_H, input_W = input_dim

    # 计算与 preproc 一致的缩放比例
    # preproc: r = min(input_H/img_H, input_W/img_W), 保持宽高比
    # 然后从 input_dim 到 output_size: output/input
    # 总缩放 = r * (output/input)
    r = min(input_H / img_H, input_W / img_W)
    scale = r * (out_H / input_H)  # 假设 out_H/input_H == out_W/input_W (正方形)
    scale_x = scale
    scale_y = scale

    # 初始化GT maps
    heatmap = np.zeros((out_H, out_W), dtype=np.float32)
    cos2theta_map = np.zeros((out_H, out_W), dtype=np.float32)
    sin2theta_map = np.zeros((out_H, out_W), dtype=np.float32)
    width_map = np.zeros((out_H, out_W), dtype=np.float32)
    quality_map = np.zeros((out_H, out_W), dtype=np.float32)

    if rect_data is None or len(rect_data) == 0:
        return {
            'heatmap': heatmap[np.newaxis, :, :],
            'cos2theta': cos2theta_map[np.newaxis, :, :],
            'sin2theta': sin2theta_map[np.newaxis, :, :],
            'width': width_map[np.newaxis, :, :],
            'quality': quality_map[np.newaxis, :, :],
        }

    # 生成网格
    yy, xx = np.meshgrid(np.arange(out_H), np.arange(out_W), indexing='ij')
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    for rect in rect_data:
        cx, cy, open_x, open_y, height, score, obj_id = rect

        if score < 0.1:  # 过滤低质量抓取
            continue

        # 计算抓取参数
        dx = open_x - cx
        dy = open_y - cy
        theta = np.arctan2(dy, dx)
        cos2theta = np.cos(2 * theta)
        sin2theta = np.sin(2 * theta)
        width = 2 * np.sqrt(dx**2 + dy**2)
        normalized_width = min(width / 100.0, 1.0)

        # 转换到输出坐标
        cx_out = cx * scale_x
        cy_out = cy * scale_y

        # 高斯响应
        dist_sq = (xx - cx_out) ** 2 + (yy - cy_out) ** 2
        gaussian = np.exp(-dist_sq / (2 * gaussian_sigma ** 2))

        # 取最大值 (处理重叠)
        heatmap = np.maximum(heatmap, gaussian * score)

        # 对于其他参数，在高斯区域内设置
        mask = gaussian > 0.5
        cos2theta_map[mask] = cos2theta
        sin2theta_map[mask] = sin2theta
        width_map[mask] = normalized_width
        quality_map[mask] = score

    return {
        'heatmap': heatmap[np.newaxis, :, :],
        'cos2theta': cos2theta_map[np.newaxis, :, :],
        'sin2theta': sin2theta_map[np.newaxis, :, :],
        'width': width_map[np.newaxis, :, :],
        'quality': quality_map[np.newaxis, :, :],
    }


class GraspNetDataset(Dataset):
    """
    GraspNet-1Billion Dataset Loader for object detection with grasp parameters.

    Each detected object carries 4-DOF grasp parameters:
    - cos(2*theta): angle cosine component
    - sin(2*theta): angle sine component
    - width: grasp width (normalized)
    - score: grasp confidence
    """

    def __init__(
        self,
        data_dir="/data/myp/grasp_dataset/scenes",
        scene_range=(0, 100),      # Training: 0-99, Val: 100-129
        camera="kinect",           # "kinect" or "realsense"
        img_size=(720, 1280),
        preproc=None,
        cache=False,
    ):
        """
        Args:
            data_dir: Path to GraspNet scenes directory
            scene_range: Tuple of (start_scene, end_scene), exclusive end
            camera: Camera type, "kinect" or "realsense"
            img_size: Image size (height, width)
            preproc: Preprocessing transform
            cache: Whether to cache images in memory
        """
        super().__init__(img_size)
        self.data_dir = data_dir
        self.camera = camera
        self.img_size = img_size
        self.preproc = preproc
        self.cache = cache

        # Build data index: [(scene_id, frame_id), ...]
        self.data_index = []
        self._build_index(scene_range)

        # Single class: graspable object
        self.num_classes = 1
        self.class_ids = [0]

        # Cache for images and annotations
        self.imgs = None
        self.annotations = None
        if self.cache:
            self._cache_data()

        logger.info(f"GraspNetDataset initialized: {len(self.data_index)} frames "
                   f"from scenes {scene_range[0]}-{scene_range[1]-1}")

    def _build_index(self, scene_range):
        """Build data index from scene directories."""
        for scene_id in range(scene_range[0], scene_range[1]):
            scene_dir = os.path.join(
                self.data_dir, f"scene_{scene_id:04d}", self.camera
            )
            if not os.path.exists(scene_dir):
                continue

            rgb_dir = os.path.join(scene_dir, "rgb")
            if not os.path.exists(rgb_dir):
                continue

            # Each scene has 256 frames (0000-0255)
            for frame_id in range(256):
                rgb_path = os.path.join(rgb_dir, f"{frame_id:04d}.png")
                if os.path.exists(rgb_path):
                    self.data_index.append((scene_id, frame_id))

    def _cache_data(self):
        """Cache all images and annotations in memory."""
        logger.info("Caching GraspNet data...")
        self.imgs = {}
        self.annotations = {}
        for idx in range(len(self.data_index)):
            scene_id, frame_id = self.data_index[idx]
            img, res, _, _ = self.pull_item(idx)
            self.imgs[idx] = img
            self.annotations[idx] = res
        logger.info(f"Cached {len(self.imgs)} images")

    def __len__(self):
        return len(self.data_index)

    def _get_paths(self, scene_id, frame_id):
        """Get file paths for a single frame."""
        base = os.path.join(self.data_dir, f"scene_{scene_id:04d}", self.camera)
        return {
            'rgb': os.path.join(base, "rgb", f"{frame_id:04d}.png"),
            'label': os.path.join(base, "label", f"{frame_id:04d}.png"),
            'rect': os.path.join(base, "rect", f"{frame_id:04d}.npy"),
        }

    def _load_frame(self, scene_id, frame_id):
        """Load data for a single frame."""
        paths = self._get_paths(scene_id, frame_id)

        # 1. Load RGB image
        img = cv2.imread(paths['rgb'])
        if img is None:
            raise ValueError(f"Failed to load image: {paths['rgb']}")
        height, width = img.shape[:2]

        # 2. Load instance mask
        label_mask = None
        if os.path.exists(paths['label']):
            label_mask = cv2.imread(paths['label'], cv2.IMREAD_UNCHANGED)

        # 3. Load grasp annotations
        rect_data = None
        if os.path.exists(paths['rect']):
            rect_data = np.load(paths['rect'])

        return img, label_mask, rect_data, (height, width)

    def _extract_bbox_from_mask(self, label_mask, object_id):
        """
        Extract bounding box from instance mask.

        Returns:
            [x1, y1, x2, y2] or None if object not found
        """
        mask = (label_mask == object_id)
        if not np.any(mask):
            return None

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        if len(row_indices) == 0 or len(col_indices) == 0:
            return None

        rmin, rmax = row_indices[0], row_indices[-1]
        cmin, cmax = col_indices[0], col_indices[-1]

        return [cmin, rmin, cmax, rmax]  # x1, y1, x2, y2

    def _compute_grasp_params(self, rect_entries, bbox_cx, bbox_cy, bbox_w, bbox_h):
        """
        Compute 6-DOF grasp parameters from rect.npy entries.

        Uses score-weighted averaging of top-k grasps to create a more robust
        grasp target that represents the distribution of valid grasps.

        Args:
            rect_entries: (M, 7) = [center_x, center_y, open_x, open_y, height, score, object_id]
            bbox_cx, bbox_cy: bbox center coordinates
            bbox_w, bbox_h: bbox dimensions

        Returns:
            (grasp_dx, grasp_dy, cos2theta, sin2theta, normalized_width, best_score)

            grasp_dx, grasp_dy: normalized offset from bbox center to grasp center
                grasp_center_x = bbox_cx + grasp_dx * bbox_w
                grasp_center_y = bbox_cy + grasp_dy * bbox_h

        Angle: theta = arctan2(open_y - center_y, open_x - center_x)
        Width: width = 2 * sqrt((open_x - center_x)^2 + (open_y - center_y)^2)
        """
        if rect_entries is None or len(rect_entries) == 0:
            # Default: grasp at bbox center
            return 0.0, 0.0, 1.0, 0.0, 0.3, 0.0

        # Strategy: Use score-weighted average of top-k grasps
        # This creates a more robust target that represents multiple valid grasps
        top_k = min(5, len(rect_entries))
        sorted_indices = np.argsort(rect_entries[:, 5])[::-1][:top_k]
        top_grasps = rect_entries[sorted_indices]

        # Score-weighted averaging
        scores = top_grasps[:, 5]
        weights = scores / (scores.sum() + 1e-6)

        # Weighted average of grasp centers
        grasp_cx = np.sum(weights * top_grasps[:, 0])
        grasp_cy = np.sum(weights * top_grasps[:, 1])

        # For angle, we need to average in cos/sin space to handle wrap-around
        cos2thetas = []
        sin2thetas = []
        widths = []
        for g in top_grasps:
            dx = g[2] - g[0]  # open_x - center_x
            dy = g[3] - g[1]  # open_y - center_y
            theta = np.arctan2(dy, dx)
            cos2thetas.append(np.cos(2 * theta))
            sin2thetas.append(np.sin(2 * theta))
            widths.append(2 * np.sqrt(dx**2 + dy**2))

        # Weighted average in cos/sin space
        avg_cos2theta = np.sum(weights * np.array(cos2thetas))
        avg_sin2theta = np.sum(weights * np.array(sin2thetas))
        # Normalize to unit circle
        norm = np.sqrt(avg_cos2theta**2 + avg_sin2theta**2) + 1e-6
        cos2theta = float(avg_cos2theta / norm)
        sin2theta = float(avg_sin2theta / norm)

        # Weighted average width
        avg_width = np.sum(weights * np.array(widths))
        normalized_width = float(min(avg_width / 100.0, 1.0))

        # Best score (for confidence)
        best_score = float(scores[0])

        # Compute grasp center offset (normalized by bbox size)
        if bbox_w > 0 and bbox_h > 0:
            grasp_dx = (grasp_cx - bbox_cx) / bbox_w
            grasp_dy = (grasp_cy - bbox_cy) / bbox_h
            grasp_dx = float(np.clip(grasp_dx, -1.0, 1.0))
            grasp_dy = float(np.clip(grasp_dy, -1.0, 1.0))
        else:
            grasp_dx, grasp_dy = 0.0, 0.0

        return grasp_dx, grasp_dy, cos2theta, sin2theta, normalized_width, best_score

    def _get_all_grasps_for_object(self, rect_entries, bbox_cx, bbox_cy, bbox_w, bbox_h):
        """
        Get ALL valid grasps for an object (for multi-grasp loss computation).

        Returns:
            list of dicts, each containing grasp parameters
        """
        if rect_entries is None or len(rect_entries) == 0:
            return []

        all_grasps = []
        for rect in rect_entries:
            grasp_cx, grasp_cy = rect[0], rect[1]
            open_x, open_y = rect[2], rect[3]
            score = float(rect[5])

            if score < 0.1:  # Filter low-quality grasps
                continue

            # Compute offset
            if bbox_w > 0 and bbox_h > 0:
                grasp_dx = (grasp_cx - bbox_cx) / bbox_w
                grasp_dy = (grasp_cy - bbox_cy) / bbox_h
                grasp_dx = float(np.clip(grasp_dx, -1.0, 1.0))
                grasp_dy = float(np.clip(grasp_dy, -1.0, 1.0))
            else:
                grasp_dx, grasp_dy = 0.0, 0.0

            # Compute angle
            dx = open_x - grasp_cx
            dy = open_y - grasp_cy
            theta = np.arctan2(dy, dx)
            cos2theta = float(np.cos(2 * theta))
            sin2theta = float(np.sin(2 * theta))

            # Compute width
            width = 2 * np.sqrt(dx**2 + dy**2)
            normalized_width = float(min(width / 100.0, 1.0))

            all_grasps.append({
                'grasp_dx': grasp_dx,
                'grasp_dy': grasp_dy,
                'cos2theta': cos2theta,
                'sin2theta': sin2theta,
                'width': normalized_width,
                'score': score,
                'center': (grasp_cx, grasp_cy)
            })

        return all_grasps

    def _build_annotations(self, label_mask, rect_data, img_size):
        """
        Build annotations for a frame.

        Returns:
            (N, 12) = [cls, cx, cy, w, h, track_id, grasp_dx, grasp_dy, cos2theta, sin2theta, grasp_width, grasp_score]

            grasp_dx, grasp_dy: normalized offset from bbox center to grasp center
                grasp_center_x = bbox_cx + grasp_dx * bbox_w
                grasp_center_y = bbox_cy + grasp_dy * bbox_h
        """
        if label_mask is None:
            return np.zeros((0, 12), dtype=np.float32)

        height, width = img_size
        unique_ids = np.unique(label_mask)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)

        annotations = []
        for obj_id in unique_ids:
            # 1. Extract bbox from mask
            bbox = self._extract_bbox_from_mask(label_mask, obj_id)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            # Filter small boxes
            if w < 5 or h < 5:
                continue

            # 2. Get grasp parameters for this object
            # NOTE: rect.npy uses 0-indexed object_id, while label.png uses 1-indexed
            # So we need to subtract 1 from label object_id to match rect object_id
            if rect_data is not None and len(rect_data) > 0:
                rect_obj_id = obj_id - 1  # Convert from label index to rect index
                obj_rects = rect_data[rect_data[:, 6] == rect_obj_id]
            else:
                obj_rects = None

            grasp_dx, grasp_dy, cos2theta, sin2theta, grasp_width, grasp_score = \
                self._compute_grasp_params(obj_rects, cx, cy, w, h)

            # 3. Assemble annotation
            # YOLOX TrainTransform expects: [x1, y1, x2, y2, cls, track_id]
            # (bbox in xyxy format, then class, then track_id)
            # TrainTransform will convert xyxy -> cxcywh internally
            # Extra columns for grasp parameters (will be trimmed for detection training)
            ann = [
                x1, y1, x2, y2,     # bbox (absolute xyxy format)
                0,                  # cls: single class = 0
                int(obj_id),        # track_id = object_id
                grasp_dx, grasp_dy, # grasp center offset (normalized)
                cos2theta,          # grasp angle encoding
                sin2theta,
                grasp_width,        # grasp width (normalized)
                grasp_score         # grasp confidence
            ]
            annotations.append(ann)

        if len(annotations) == 0:
            return np.zeros((0, 12), dtype=np.float32)

        return np.array(annotations, dtype=np.float32)

    def load_anno_from_ids(self, id_):
        """Load annotation by index (for compatibility with YOLOX)."""
        scene_id, frame_id = self.data_index[id_]
        paths = self._get_paths(scene_id, frame_id)

        label_mask = None
        if os.path.exists(paths['label']):
            label_mask = cv2.imread(paths['label'], cv2.IMREAD_UNCHANGED)

        rect_data = None
        if os.path.exists(paths['rect']):
            rect_data = np.load(paths['rect'])

        res = self._build_annotations(label_mask, rect_data, self.img_size)

        # Convert to YOLOX format: (N, 5) = [cls, cx, cy, w, h] in relative coords
        # But we keep absolute coords and 10 dimensions for grasp
        return res

    def load_anno(self, index):
        """Load annotation by index."""
        return self.load_anno_from_ids(index)

    def load_resized_img(self, index):
        """Load and resize image."""
        img, _, _, _ = self._load_frame(*self.data_index[index])
        return img

    def pull_item(self, index):
        """
        Pull a data item.

        Returns:
            img: BGR image
            res: annotations (N, 6) = [cx, cy, w, h, cls, track_id]
                 Note: We only return detection-related columns for YOLOX training.
            rect_data: raw grasp annotations (N, 7) for generating dense GT
            img_info: (height, width, frame_id, scene_id, file_name)
            img_id: index
        """
        scene_id, frame_id = self.data_index[index]

        if self.cache and self.imgs is not None:
            img = self.imgs[index]
            res = self.annotations[index].copy()
            # For cached data, we need to reload rect_data
            paths = self._get_paths(scene_id, frame_id)
            rect_data = np.load(paths['rect']) if os.path.exists(paths['rect']) else None
        else:
            img, label_mask, rect_data, img_size = self._load_frame(scene_id, frame_id)
            res = self._build_annotations(label_mask, rect_data, img_size)

        height, width = img.shape[:2]
        img_info = (height, width, frame_id, scene_id,
                    f"scene_{scene_id:04d}/frame_{frame_id:04d}")

        # Only return detection columns: [cx, cy, w, h, cls, track_id]
        # Grasp head uses dense prediction from FPN, not per-object annotations
        if res.shape[1] > 6:
            res = res[:, :6].copy()

        return img, res.copy(), rect_data, img_info, np.array([index])

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        Get a data item with preprocessing.

        Returns:
            img: preprocessed image tensor
            target: padded annotations
            grasp_gt: dict with dense grasp GT tensors
            img_info: image information
            img_id: index
        """
        img, target, rect_data, img_info, img_id = self.pull_item(index)

        # 原始图像尺寸
        orig_h, orig_w = img.shape[:2]

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        # 生成密集抓取GT
        # 输入尺寸是 self.input_dim (e.g., 640x640)
        # 输出尺寸是 stride=4 (e.g., 160x160)
        output_h, output_w = self.input_dim[0] // 4, self.input_dim[1] // 4
        grasp_gt = generate_grasp_dense_gt(
            rect_data,
            img_size=(orig_h, orig_w),
            output_size=(output_h, output_w),
            input_dim=self.input_dim,  # 传入模型输入尺寸，用于计算正确的缩放
            gaussian_sigma=2.0
        )

        # 转换为 torch tensor
        grasp_gt_tensor = {
            'heatmap': torch.from_numpy(grasp_gt['heatmap']).float(),
            'cos2theta': torch.from_numpy(grasp_gt['cos2theta']).float(),
            'sin2theta': torch.from_numpy(grasp_gt['sin2theta']).float(),
            'width': torch.from_numpy(grasp_gt['width']).float(),
            'quality': torch.from_numpy(grasp_gt['quality']).float(),
        }

        return img, target, grasp_gt_tensor, img_info, img_id


class GraspNetVideoDataset(Dataset):
    """
    GraspNet dataset organized as videos (scenes).
    Each scene is treated as a video with 256 frames.
    """

    def __init__(
        self,
        data_dir="/data/myp/grasp_dataset/scenes",
        scene_ids=None,
        camera="kinect",
        img_size=(720, 1280),
        preproc=None,
    ):
        """
        Args:
            data_dir: Path to GraspNet scenes
            scene_ids: List of scene IDs to include, or None for all
            camera: Camera type
            img_size: Image size
            preproc: Preprocessing transform
        """
        super().__init__(img_size)
        self.data_dir = data_dir
        self.camera = camera
        self.preproc = preproc

        # Get scene list
        if scene_ids is None:
            scene_ids = list(range(190))
        self.scene_ids = [s for s in scene_ids if self._scene_exists(s)]

        logger.info(f"GraspNetVideoDataset: {len(self.scene_ids)} scenes")

    def _scene_exists(self, scene_id):
        scene_dir = os.path.join(self.data_dir, f"scene_{scene_id:04d}", self.camera)
        return os.path.exists(scene_dir)

    def __len__(self):
        return len(self.scene_ids)

    def get_scene(self, scene_idx):
        """
        Get all frames from a scene.

        Returns:
            List of (img, annotations, img_info) tuples
        """
        scene_id = self.scene_ids[scene_idx]
        scene_dir = os.path.join(self.data_dir, f"scene_{scene_id:04d}", self.camera)

        frames = []
        for frame_id in range(256):
            rgb_path = os.path.join(scene_dir, "rgb", f"{frame_id:04d}.png")
            if not os.path.exists(rgb_path):
                continue

            img = cv2.imread(rgb_path)
            label_path = os.path.join(scene_dir, "label", f"{frame_id:04d}.png")
            rect_path = os.path.join(scene_dir, "rect", f"{frame_id:04d}.npy")

            label_mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED) if os.path.exists(label_path) else None
            rect_data = np.load(rect_path) if os.path.exists(rect_path) else None

            # Build annotations
            dataset = GraspNetDataset.__new__(GraspNetDataset)
            dataset.img_size = self.input_dim
            annotations = dataset._build_annotations(label_mask, rect_data, img.shape[:2])

            img_info = (img.shape[0], img.shape[1], frame_id, scene_id,
                       f"scene_{scene_id:04d}/frame_{frame_id:04d}")

            frames.append((img, annotations, img_info))

        return frames


def grasp_collate_fn(batch):
    """
    自定义collate函数，处理包含抓取GT的批量数据

    Args:
        batch: list of (img, target, grasp_gt, img_info, img_id)

    Returns:
        imgs: (B, 3, H, W) tensor
        targets: (B, max_objs, 6) tensor
        grasp_targets: dict of (B, C, H, W) tensors
        img_infos: list of img_info tuples
        img_ids: (B,) tensor
    """
    imgs, targets, grasp_gts, img_infos, img_ids = zip(*batch)

    # Stack images
    # 预处理后图像已经是 (C, H, W) 格式的 numpy array 或 tensor
    processed_imgs = []
    for img in imgs:
        if isinstance(img, np.ndarray):
            # 如果是 (H, W, C) 格式转为 (C, H, W)
            if img.ndim == 3 and img.shape[2] in [1, 3]:
                img = torch.from_numpy(img).permute(2, 0, 1).float()
            else:
                # 已经是 (C, H, W) 格式
                img = torch.from_numpy(img).float()
        processed_imgs.append(img)
    imgs = torch.stack(processed_imgs, dim=0)

    # Pad targets to same length
    max_objs = max(t.shape[0] for t in targets)
    if max_objs == 0:
        max_objs = 1  # 至少保持1个维度

    batch_targets = []
    for t in targets:
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()
        if t.shape[0] == 0:
            t = torch.zeros((1, 6))
        # Pad to max_objs
        pad_size = max_objs - t.shape[0]
        if pad_size > 0:
            t = torch.cat([t, torch.zeros((pad_size, t.shape[1]))], dim=0)
        batch_targets.append(t)

    targets = torch.stack(batch_targets, dim=0)

    # Stack grasp GT
    grasp_targets = {
        'heatmap': torch.stack([g['heatmap'] for g in grasp_gts], dim=0),
        'cos2theta': torch.stack([g['cos2theta'] for g in grasp_gts], dim=0),
        'sin2theta': torch.stack([g['sin2theta'] for g in grasp_gts], dim=0),
        'width': torch.stack([g['width'] for g in grasp_gts], dim=0),
        'quality': torch.stack([g['quality'] for g in grasp_gts], dim=0),
    }

    # Stack img_ids
    img_ids = torch.cat([torch.from_numpy(i) if isinstance(i, np.ndarray) else i
                         for i in img_ids], dim=0)

    return imgs, targets, grasp_targets, list(img_infos), img_ids
