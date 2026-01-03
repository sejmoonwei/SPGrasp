#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
YOLOX-S + Dense Grasp Head 训练配置

架构: YOLOX检测 + GraspDenseHead密集抓取预测
优势:
- 检测与抓取解耦，各自独立预测
- 密集抓取预测，每个像素都有抓取参数
- 每个检测物体可采样多个有效抓取点
"""

import os
import torch
import torch.nn as nn

from yolox.exp import Exp as MyExp
from yolox.models import (
    YOLOXGrasp,
    build_yolox_grasp,
    GraspDenseLoss,
)


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # Model config
        self.num_classes = 1  # Single class: graspable object
        self.depth = 0.33     # YOLOX-S
        self.width = 0.50     # YOLOX-S

        # Training config
        self.max_epoch = 100
        self.eval_interval = 100  # 每100个epoch验证一次
        self.ckpt_interval = 5    # 每5个epoch保存一次权重
        self.warmup_epochs = 5

        # Adam优化器配置 (比SGD使用更小的学习率)
        self.use_adam = True  # 使用Adam优化器
        self.adam_lr = 1e-4   # Adam基础学习率
        self.adam_lr_backbone = 1e-5  # Backbone使用更小的学习率

        # SGD配置 (备用)
        self.basic_lr_per_img = 0.01 / 64.0
        self.weight_decay = 5e-4
        self.momentum = 0.9

        # Data config
        self.data_num_workers = 4  # 减少数据加载线程，避免OOM
        self.data_dir = "/data/myp/grasp_dataset/scenes"
        self.train_scenes = (0, 100)   # scene_0000 - scene_0099
        self.val_scenes = (100, 130)   # scene_0100 - scene_0129

        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = None  # 禁用多尺度训练，固定输入尺寸

        # Grasp loss config (dense architecture)
        self.grasp_loss_weight = 2.0          # 总抓取损失权重
        self.grasp_heatmap_weight = 1.0       # 热图损失权重
        self.grasp_angle_weight = 1.0         # 角度损失权重
        self.grasp_width_weight = 1.0         # 宽度损失权重
        self.grasp_quality_weight = 0.5       # 质量损失权重
        self.grasp_gaussian_sigma = 2.0       # 高斯核大小

        # Experiment name
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_grasp_loss_fn(self):
        """获取抓取损失函数实例"""
        return GraspDenseLoss(
            heatmap_weight=self.grasp_heatmap_weight,
            angle_weight=self.grasp_angle_weight,
            width_weight=self.grasp_width_weight,
            quality_weight=self.grasp_quality_weight,
            gaussian_sigma=self.grasp_gaussian_sigma,
        )

    def get_model(self):
        """Build YOLOX-Grasp model with configured loss function"""
        # 初始化函数：设置BatchNorm参数
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        # 只在模型未创建时创建
        if getattr(self, "model", None) is None:
            self.model = build_yolox_grasp(
                num_classes=self.num_classes,
                depth=self.depth,
                width=self.width,
                with_grasp_head=True,
            )

        # 初始化模型参数
        self.model.apply(init_yolo)

        # Initialize detection head
        self.model.head.initialize_biases(1e-2)

        # Set grasp loss function with configured weights
        grasp_loss_fn = self.get_grasp_loss_fn()
        self.model.set_grasp_loss_fn(grasp_loss_fn, self.grasp_loss_weight)

        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        """Get data loader with grasp annotations"""
        from yolox.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
        )
        from yolox.data.datasets import GraspNetDataset
        from yolox.data.datasets.graspnet import grasp_collate_fn
        import torch.distributed as dist

        # Dataset
        # 使用 input_size 而不是原图尺寸，确保 grasp GT 与模型输出尺寸匹配
        dataset = GraspNetDataset(
            data_dir=self.data_dir,
            scene_range=self.train_scenes,
            camera="kinect",
            img_size=self.input_size,  # (640, 640) 与模型输入匹配
            preproc=TrainTransform(
                max_labels=100,
                p=0.5,  # 随机翻转概率
            ),
            cache=cache_img,
        )

        self.dataset = dataset

        # Sampler
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset),
            seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
            "collate_fn": grasp_collate_fn,  # 使用自定义collate函数处理抓取GT
        }

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Get evaluation data loader"""
        from yolox.data import ValTransform
        from yolox.data.datasets import GraspNetDataset

        dataset = GraspNetDataset(
            data_dir=self.data_dir,
            scene_range=self.val_scenes,
            camera="kinect",
            img_size=self.test_size,  # 使用 test_size 与模型输入匹配
            preproc=ValTransform(),
            cache=False,
        )

        if is_distributed:
            import torch.distributed as dist
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
        }

        val_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Get evaluator for grasp detection"""
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)

        # Use COCO evaluator for detection metrics
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def get_optimizer(self, batch_size):
        """
        Get optimizer - joint training of detection + grasp

        训练策略:
        - 联合训练所有参数 (backbone + neck + head + grasp_head)
        - 检测器需要适应GraspNet数据集的物体特征（遮挡、不完整等）
        - 使用不同学习率: backbone较小, head较大
        - 支持SGD和Adam两种优化器
        """
        if "optimizer" not in self.__dict__:
            # 分组参数: 不同模块使用不同学习率
            pg_backbone = []  # backbone参数，较小学习率
            pg_other_weight = []  # 其他模块weight，正常学习率
            pg_bias = []  # 所有bias，无weight decay
            pg_bn = []  # BatchNorm，无weight decay

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                if 'backbone' in name:
                    if 'bias' in name:
                        pg_bias.append(param)
                    elif 'bn' in name or isinstance(param, nn.BatchNorm2d):
                        pg_bn.append(param)
                    else:
                        pg_backbone.append(param)
                else:
                    if 'bias' in name:
                        pg_bias.append(param)
                    elif 'bn' in name:
                        pg_bn.append(param)
                    else:
                        pg_other_weight.append(param)

            # 选择优化器
            use_adam = getattr(self, 'use_adam', False)

            if use_adam:
                # Adam优化器
                lr = self.adam_lr
                lr_backbone = getattr(self, 'adam_lr_backbone', lr * 0.1)

                optimizer = torch.optim.AdamW([
                    {"params": pg_bn, "weight_decay": 0.0},  # BN无decay
                    {"params": pg_backbone, "lr": lr_backbone, "weight_decay": self.weight_decay},  # backbone较小lr
                    {"params": pg_other_weight, "weight_decay": self.weight_decay},  # neck/head/grasp_head正常lr
                    {"params": pg_bias, "weight_decay": 0.0},  # bias无decay
                ], lr=lr, betas=(0.9, 0.999))

                print(f"[Optimizer] Using AdamW")
                print(f"[Optimizer] Backbone lr: {lr_backbone:.2e}, Other lr: {lr:.2e}")
            else:
                # SGD优化器
                if self.warmup_epochs > 0:
                    lr = self.warmup_lr
                else:
                    lr = self.basic_lr_per_img * batch_size

                optimizer = torch.optim.SGD([
                    {"params": pg_bn, "weight_decay": 0.0},  # BN无decay
                    {"params": pg_backbone, "lr": lr * 0.1, "weight_decay": self.weight_decay},  # backbone较小lr
                    {"params": pg_other_weight, "weight_decay": self.weight_decay},  # neck/head/grasp_head正常lr
                    {"params": pg_bias, "weight_decay": 0.0},  # bias无decay
                ], lr=lr, momentum=self.momentum, nesterov=True)

                print(f"[Optimizer] Using SGD")
                print(f"[Optimizer] Backbone lr: {lr * 0.1:.2e}, Other lr: {lr:.2e}")

            self.optimizer = optimizer

            # Log parameter groups
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[Optimizer] Total: {total_params:,}, Trainable: {trainable_params:,}")

        return self.optimizer
