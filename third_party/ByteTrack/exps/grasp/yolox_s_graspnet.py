#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX-S GraspNet Training Configuration

Training ByteTrack-Grasp on GraspNet-1Billion dataset.
Uses GraspYOLOXHead for joint detection and grasp prediction.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- Model Config ---------------- #
        self.num_classes = 1  # Single class: graspable object
        self.depth = 0.33     # YOLOX-S
        self.width = 0.50     # YOLOX-S
        self.grasp_weight = 2.0  # Loss weight for grasp parameters

        # ---------------- Data Config ---------------- #
        self.data_dir = "/data/myp/grasp_dataset/scenes"
        self.train_scenes = (0, 100)    # scene_0000 - scene_0099 (training)
        self.val_scenes = (100, 130)    # scene_0100 - scene_0129 (validation)
        self.camera = "kinect"

        # GraspNet native resolution
        self.input_size = (640, 640)  # Resize for training
        self.test_size = (640, 640)
        self.random_size = (14, 26)   # Multi-scale training

        # ---------------- DataLoader Config ---------------- #
        self.data_num_workers = 4
        self.batch_size = 8  # Per GPU

        # ---------------- Training Config ---------------- #
        self.max_epoch = 100
        self.warmup_epochs = 5
        self.no_aug_epochs = 15
        self.basic_lr_per_img = 0.001 / 64.0

        self.scheduler = "yoloxwarmcos"
        self.min_lr_ratio = 0.05
        self.ema = True
        self.weight_decay = 5e-4
        self.momentum = 0.9

        self.print_interval = 50
        self.eval_interval = 5
        self.save_history_ckpt = True

        # ---------------- Augmentation Config ---------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.5, 1.5)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0

        # ---------------- Testing Config ---------------- #
        self.test_conf = 0.01
        self.nmsthre = 0.65

        # Experiment name
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self):
        """Build YOLOX model with GraspYOLOXHead."""
        from yolox.models import YOLOPAFPN, YOLOX
        from yolox.models.grasp_yolo_head import GraspYOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels
            )
            head = GraspYOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                grasp_weight=self.grasp_weight,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        """Build training data loader."""
        from yolox.data import (
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            YoloBatchSampler,
        )
        from yolox.data.datasets.graspnet import GraspNetDataset

        # Create dataset
        dataset = GraspNetDataset(
            data_dir=self.data_dir,
            scene_range=self.train_scenes,
            camera=self.camera,
            img_size=self.input_size,
            preproc=GraspTrainTransform(
                max_labels=50,
                flip_prob=0.5,
                hsv_prob=1.0,
            ),
            cache=cache_img,
        )

        # Apply Mosaic augmentation
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=GraspTrainTransform(
                max_labels=120,
                flip_prob=0.5,
                hsv_prob=1.0,
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.scale,
            mixup_scale=(0.5, 1.5),
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob if not no_aug else 0.0,
            mixup_prob=self.mixup_prob if not no_aug else 0.0,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
        }
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Collate function for grasp annotations
        dataloader_kwargs["collate_fn"] = grasp_collate_fn

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Build validation data loader."""
        from yolox.data import DataLoader
        from yolox.data.datasets.graspnet import GraspNetDataset

        valdataset = GraspNetDataset(
            data_dir=self.data_dir,
            scene_range=self.val_scenes,
            camera=self.camera,
            img_size=self.test_size,
            preproc=GraspValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
            "collate_fn": grasp_collate_fn,
        }

        val_loader = DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Build evaluator for validation."""
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)

        evaluator = GraspEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )

        return evaluator

    def get_trainer(self, args):
        """Get trainer with grasp-specific logging."""
        from yolox.core import Trainer
        trainer = Trainer(self, args)
        return trainer


class GraspTrainTransform:
    """
    Training transform for GraspNet dataset.
    Handles 12-dimensional annotations (bbox + grasp params with center offset).
    """

    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        """
        Args:
            image: BGR image
            targets: (N, 12) = [cls, cx, cy, w, h, track_id, grasp_dx, grasp_dy, cos2t, sin2t, width, score]
            input_dim: Target size (h, w)

        Returns:
            image: Preprocessed image
            targets: Padded targets
        """
        import cv2
        import numpy as np

        # Handle empty targets
        if targets is None or len(targets) == 0:
            targets = np.zeros((0, 12), dtype=np.float32)

        boxes = targets[:, 1:5].copy() if len(targets) > 0 else np.zeros((0, 4))
        labels = targets[:, 0].copy() if len(targets) > 0 else np.zeros((0,))
        # grasp_params: [track_id, grasp_dx, grasp_dy, cos2t, sin2t, width, score]
        grasp_params = targets[:, 5:12].copy() if len(targets) > 0 else np.zeros((0, 7))

        # Get image dimensions
        height, width = image.shape[:2]

        # Resize image
        r = min(input_dim[0] / height, input_dim[1] / width)
        new_h, new_w = int(height * r), int(width * r)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_h = input_dim[0] - new_h
        pad_w = input_dim[1] - new_w
        top = pad_h // 2
        left = pad_w // 2

        padded_img = np.full((input_dim[0], input_dim[1], 3), 114, dtype=np.uint8)
        padded_img[top:top+new_h, left:left+new_w] = image
        image = padded_img

        # Scale boxes
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + left  # x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + top   # y

        # Random horizontal flip
        if np.random.random() < self.flip_prob:
            image = image[:, ::-1]
            if len(boxes) > 0:
                boxes[:, 0] = input_dim[1] - boxes[:, 0]
                # Flip grasp center offset: grasp_dx negates
                grasp_params[:, 1] = -grasp_params[:, 1]  # grasp_dx
                # Flip grasp angle: cos(2*theta) stays same, sin(2*theta) negates
                grasp_params[:, 4] = -grasp_params[:, 4]  # sin2theta

        # HSV augmentation
        if np.random.random() < self.hsv_prob:
            image = self._augment_hsv(image)

        # Convert to float and normalize
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = np.ascontiguousarray(image)

        # Assemble targets
        # Format: [cls, cx, cy, w, h, track_id, grasp_dx, grasp_dy, cos2t, sin2t, width, score]
        if len(boxes) > 0:
            labels = labels.reshape(-1, 1)
            targets_out = np.hstack([labels, boxes, grasp_params])
        else:
            targets_out = np.zeros((0, 12), dtype=np.float32)

        # Pad to max_labels
        padded_targets = np.zeros((self.max_labels, 12), dtype=np.float32)
        num_targets = min(len(targets_out), self.max_labels)
        if num_targets > 0:
            padded_targets[:num_targets] = targets_out[:num_targets]

        return image, padded_targets

    def _augment_hsv(self, img, hgain=0.015, sgain=0.7, vgain=0.4):
        """HSV color augmentation."""
        import cv2
        import numpy as np

        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        img_hsv = cv2.merge([
            cv2.LUT(hue, lut_hue),
            cv2.LUT(sat, lut_sat),
            cv2.LUT(val, lut_val)
        ]).astype(np.uint8)

        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


class GraspValTransform:
    """Validation transform for GraspNet dataset."""

    def __init__(self, legacy=False):
        self.legacy = legacy

    def __call__(self, image, targets, input_dim):
        import cv2
        import numpy as np

        height, width = image.shape[:2]

        # Resize
        r = min(input_dim[0] / height, input_dim[1] / width)
        new_h, new_w = int(height * r), int(width * r)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad
        pad_h = input_dim[0] - new_h
        pad_w = input_dim[1] - new_w
        top = pad_h // 2
        left = pad_w // 2

        padded_img = np.full((input_dim[0], input_dim[1], 3), 114, dtype=np.uint8)
        padded_img[top:top+new_h, left:left+new_w] = image
        image = padded_img

        # Convert
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        return image, np.zeros((1, 12), dtype=np.float32)


def grasp_collate_fn(batch):
    """Custom collate function for grasp dataset."""
    import torch
    import numpy as np

    imgs, targets, img_infos, img_ids = zip(*batch)

    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    targets = torch.from_numpy(np.stack(targets, axis=0))

    return imgs, targets, img_infos, img_ids


class GraspEvaluator:
    """Evaluator for grasp detection."""

    def __init__(self, dataloader, img_size, confthre, nmsthre, num_classes):
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes

    def evaluate(self, model, is_distributed, half=False):
        """Run evaluation."""
        import torch
        from tqdm import tqdm

        model.eval()
        device = next(model.parameters()).device

        results = []
        for imgs, targets, img_infos, img_ids in tqdm(self.dataloader, desc="Evaluating"):
            imgs = imgs.to(device)
            if half:
                imgs = imgs.half()

            with torch.no_grad():
                outputs = model(imgs)

            # Process outputs...
            results.append(outputs)

        # Compute metrics
        # TODO: Implement grasp-specific evaluation metrics
        return 0.0, 0.0, ""  # ap50, ap50_95, summary
