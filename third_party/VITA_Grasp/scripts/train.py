#!/usr/bin/env python3
"""
Training script for VITA-Grasp model with DDP multi-GPU support.

Usage:
    # Single GPU
    python scripts/train.py --config configs/default.yaml

    # Multi-GPU with torchrun (recommended)
    CUDA_VISIBLE_DEVICES=2,3,5,7 torchrun --nproc_per_node=4 scripts/train.py \
        --config configs/internvit_grasp.yaml \
        --batch_size 4 \
        --epochs 100

    # Or with command line arguments
    python scripts/train.py \
        --dataset_root /data/myp/grasp_dataset/scenes \
        --output_dir outputs/qwen2vl_grasp \
        --batch_size 4 \
        --epochs 100
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import yaml

# Wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Qwen2VLGrasp, build_qwen2vl_grasp
from models import InternViTGrasp, build_internvit_grasp
from datasets import GraspNetDataset, build_graspnet_dataloader
from loss import GraspLoss, build_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for VITA-Grasp model with DDP multi-GPU support."""

    def __init__(self, config: dict):
        self.config = config

        # Distributed training setup
        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Set device based on local rank
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = torch.device('cpu')

        # Only rank 0 handles logging and saving
        self.is_main_process = (self.rank == 0)

        if self.is_main_process:
            logger.info(f"Distributed training: {self.is_distributed}")
            logger.info(f"World size: {self.world_size}, Rank: {self.rank}, Local rank: {self.local_rank}")
            logger.info(f"Device: {self.device}")

        # Setup output directory (only main process)
        self.output_dir = Path(config['output_dir'])
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Save config
            with open(self.output_dir / 'config.yaml', 'w') as f:
                yaml.dump(config, f)

        # Synchronize before continuing
        if self.is_distributed:
            dist.barrier()

        # Build model
        if self.is_main_process:
            logger.info("Building model...")
        self.model = self._build_model()
        self.model.to(self.device)

        # Convert BatchNorm to SyncBatchNorm for distributed training
        # Note: Disabled by default as InternViT uses LayerNorm, only decoder has BatchNorm
        use_sync_bn = config.get('use_sync_bn', False)
        if self.is_distributed and use_sync_bn:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self.is_main_process:
                logger.info("Converted BatchNorm layers to SyncBatchNorm")

        # Apply torch.compile for PyTorch 2.0+ optimization
        # Note: First compilation can take 5-10 minutes, set to False for faster startup
        use_compile = config.get('use_compile', False)
        if use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(
                    self.model,
                    mode='reduce-overhead',  # Good balance for training
                    dynamic=True,
                )
                if self.is_main_process:
                    logger.info("Model compiled with torch.compile (mode=reduce-overhead)")
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"torch.compile failed: {e}, using uncompiled model")

        # Wrap model with DDP
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,  # No unused parameters in this model
            )
            if self.is_main_process:
                logger.info("Model wrapped with DistributedDataParallel")

        # Build dataloaders
        if self.is_main_process:
            logger.info("Building dataloaders...")
        self.train_loader = self._build_dataloader('train')
        self.val_loader = self._build_dataloader('val')

        # Store sampler for epoch setting
        self.train_sampler = self.train_loader.sampler if self.is_distributed else None

        # Build optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Build loss function
        self.criterion = build_loss(
            config.get('loss_type', 'default'),
            pos_weight=config.get('pos_weight', 5.0),
            ang_weight=config.get('ang_weight', 5.0),
            wid_weight=config.get('wid_weight', 1.0),
            sem_weight=config.get('sem_weight', 1.0),
            focal_alpha=config.get('focal_alpha', 2.0),
            focal_beta=config.get('focal_beta', 4.0),
            pos_thresh=config.get('pos_thresh', 0.5),
        )

        if self.is_main_process:
            logger.info(f"Loss config: focal_alpha={config.get('focal_alpha', 2.0)}, "
                       f"focal_beta={config.get('focal_beta', 4.0)}, "
                       f"pos_thresh={config.get('pos_thresh', 0.5)}")

        # Mixed precision training
        # Use bfloat16 for InternViT (doesn't need GradScaler)
        self.use_amp = config.get('use_amp', True)
        self.use_bfloat16 = config.get('model_type') == 'internvit' and not config.get('freeze_vit', False)
        # Only use GradScaler for float16, not bfloat16
        self.scaler = GradScaler('cuda') if (self.use_amp and not self.use_bfloat16) else None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Resume from checkpoint
        if config.get('resume'):
            self._load_checkpoint(config['resume'])

        # Initialize wandb (only main process)
        self.use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE and self.is_main_process
        if self.use_wandb:
            wandb_config = {
                'model_type': config.get('model_type', 'qwen2vl'),
                'pretrained_model': config.get('pretrained_model'),
                'freeze_vit': config.get('freeze_vit', True),
                'decoder_channels': config.get('decoder_channels', 256),
                'output_size': config.get('output_size', [480, 640]),
                'batch_size': config.get('batch_size', 4),
                'batch_size_total': config.get('batch_size', 4) * self.world_size,
                'num_gpus': self.world_size,
                'epochs': config.get('epochs', 100),
                'learning_rate': config.get('learning_rate', 1e-4),
                'weight_decay': config.get('weight_decay', 0.01),
                'warmup_epochs': config.get('warmup_epochs', 5),
                'use_amp': config.get('use_amp', True),
                'loss_type': config.get('loss_type', 'default'),
                'pos_weight': config.get('pos_weight', 5.0),
                'ang_weight': config.get('ang_weight', 5.0),
                'wid_weight': config.get('wid_weight', 1.0),
                'sem_weight': config.get('sem_weight', 1.0),
            }
            wandb.init(
                project=config.get('wandb_project', 'vita-grasp'),
                name=config.get('wandb_run_name', f"{config.get('model_type', 'qwen2vl')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                config=wandb_config,
                dir=str(self.output_dir),
                resume='allow' if config.get('resume') else False,
            )
            logger.info(f"Wandb initialized: {wandb.run.url}")

    def _build_model(self):
        """Build grasp model based on model_type."""
        model_type = self.config.get('model_type', 'qwen2vl')

        if model_type == 'internvit':
            # Build InternViT-Grasp model (ByteTrack-style: square input, stride=4 output)
            logger.info("Building InternViT-Grasp model...")
            return build_internvit_grasp(
                pretrained_model=self.config.get('pretrained_model', 'OpenGVLab/InternViT-300M-448px'),
                freeze_vit=self.config.get('freeze_vit', True),
                vit_lr_scale=self.config.get('vit_lr_scale', 0.1),
                decoder_channels=self.config.get('decoder_channels', 256),
                num_decoder_layers=self.config.get('num_decoder_layers', 4),
                input_size=tuple(self.config.get('input_size', [640, 640])),
                output_stride=self.config.get('output_stride', 4),  # ByteTrack-style stride=4
            )
        else:
            # Build Qwen2VL-Grasp model (default)
            logger.info("Building Qwen2VL-Grasp model...")
            return build_qwen2vl_grasp(
                pretrained_model=self.config.get('pretrained_model', 'Qwen/Qwen2-VL-2B-Instruct'),
                freeze_vit=self.config.get('freeze_vit', True),
                vit_lr_scale=self.config.get('vit_lr_scale', 0.1),
                decoder_channels=self.config.get('decoder_channels', 256),
                num_decoder_layers=self.config.get('num_decoder_layers', 4),
                output_size=tuple(self.config.get('output_size', [480, 640])),
                use_flash_attn=self.config.get('use_flash_attn', True),
            )

    def _build_dataloader(self, split: str) -> DataLoader:
        """Build dataloader for given split (ByteTrack-style preprocessing, stride=4 GT)."""
        return build_graspnet_dataloader(
            dataset_root=self.config['dataset_root'],
            split=split,
            batch_size=self.config.get('batch_size', 4),
            num_workers=self.config.get('num_workers', 4),
            input_size=tuple(self.config.get('input_size', [640, 640])),
            output_stride=self.config.get('output_stride', 4),  # ByteTrack-style stride=4
            camera_type=self.config.get('camera_type', 'kinect'),
            num_train_scenes=self.config.get('num_train_scenes', 100),
            num_val_scenes=self.config.get('num_val_scenes', 30),
            gaussian_sigma=self.config.get('gaussian_sigma', 2.0),
            max_width=self.config.get('max_width', 100.0),
            pad_value=self.config.get('pad_value', 114),
            is_distributed=self.is_distributed,  # Enable distributed sampler
        )

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer with parameter groups."""
        base_lr = self.config.get('learning_rate', 1e-4)

        # Get parameter groups from model (handle DDP wrapper)
        model = self.model.module if self.is_distributed else self.model
        param_groups = model.get_param_groups(base_lr)

        # Try to use fused AdamW for better performance (PyTorch 2.0+)
        use_fused = self.config.get('use_fused_optimizer', True)
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames

        if use_fused and fused_available and torch.cuda.is_available():
            try:
                optimizer = optim.AdamW(
                    param_groups,
                    lr=base_lr,
                    weight_decay=self.config.get('weight_decay', 0.01),
                    betas=(0.9, 0.999),
                    fused=True,  # Fused CUDA kernel for better performance
                )
                if self.is_main_process:
                    logger.info("Using fused AdamW optimizer (CUDA optimized)")
            except Exception as e:
                if self.is_main_process:
                    logger.warning(f"Fused AdamW failed: {e}, falling back to standard AdamW")
                optimizer = optim.AdamW(
                    param_groups,
                    lr=base_lr,
                    weight_decay=self.config.get('weight_decay', 0.01),
                    betas=(0.9, 0.999),
                )
        else:
            optimizer = optim.AdamW(
                param_groups,
                lr=base_lr,
                weight_decay=self.config.get('weight_decay', 0.01),
                betas=(0.9, 0.999),
            )

        return optimizer

    def _build_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler."""
        num_epochs = self.config.get('epochs', 100)
        warmup_epochs = self.config.get('warmup_epochs', 5)
        min_lr = self.config.get('min_lr', 1e-6)

        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return max(min_lr / self.config.get('learning_rate', 1e-4),
                          0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """Save training checkpoint (only on main process) with proper synchronization."""
        # Only main process saves, but all processes must wait
        if self.is_main_process:
            # Get model state dict (handle DDP wrapper)
            model = self.model.module if self.is_distributed else self.model

            checkpoint = {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'world_size': self.world_size,
            }

            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()

            torch.save(checkpoint, self.output_dir / filename)
            logger.info(f"Saved checkpoint: {self.output_dir / filename}")

        # Synchronize all processes after saving
        if self.is_distributed:
            dist.barrier()

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint with proper synchronization."""
        if self.is_main_process:
            logger.info(f"Loading checkpoint: {checkpoint_path}")

        # All processes load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Warn if world_size mismatch
        saved_world_size = checkpoint.get('world_size', 1)
        if saved_world_size != self.world_size and self.is_main_process:
            logger.warning(
                f"World size mismatch: checkpoint saved with {saved_world_size} GPUs, "
                f"but current run has {self.world_size} GPUs"
            )

        # Load model state dict (handle DDP wrapper)
        model = self.model.module if self.is_distributed else self.model
        model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.is_main_process:
            logger.info(f"Resumed from epoch {self.epoch}, global_step {self.global_step}")

        # Synchronize all processes after loading
        if self.is_distributed:
            dist.barrier()

    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()

        # Freeze ViT if specified (handle DDP wrapper)
        if self.config.get('freeze_vit', True):
            model = self.model.module if self.is_distributed else self.model
            model.vit.eval()

        total_loss = 0.0
        loss_components = {'loss_pos': 0, 'loss_ang': 0, 'loss_wid': 0, 'loss_sem': 0}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            targets = batch['grasp_mask'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                # Use bfloat16 for unfrozen InternViT, float16 otherwise
                amp_dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16
                with autocast('cuda', dtype=amp_dtype):
                    outputs = self.model(images)
                    losses = self.criterion(outputs['pred_masks'], targets)
                    loss = losses['loss']

                if self.scaler is not None:
                    # float16 path with GradScaler
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # bfloat16 path without GradScaler
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            else:
                outputs = self.model(images)
                losses = self.criterion(outputs['pred_masks'], targets)
                loss = losses['loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()
            num_batches += 1
            self.global_step += 1

            # Logging (only main process)
            if self.is_main_process and batch_idx % self.config.get('log_interval', 50) == 0:
                logger.info(
                    f"Epoch {self.epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(pos: {losses['loss_pos'].item():.4f}, "
                    f"ang: {losses['loss_ang'].item():.4f}, "
                    f"wid: {losses['loss_wid'].item():.4f}, "
                    f"sem: {losses['loss_sem'].item():.4f})"
                )

                # Wandb step logging
                if self.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/loss_pos': losses['loss_pos'].item(),
                        'train/loss_ang': losses['loss_ang'].item(),
                        'train/loss_wid': losses['loss_wid'].item(),
                        'train/loss_sem': losses['loss_sem'].item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'global_step': self.global_step,
                    })

        # Aggregate losses across all GPUs in distributed training
        if self.is_distributed:
            # Create tensor with all losses and batch count
            loss_tensor = torch.tensor(
                [total_loss, loss_components['loss_pos'], loss_components['loss_ang'],
                 loss_components['loss_wid'], loss_components['loss_sem'], float(num_batches)],
                device=self.device,
                dtype=torch.float32
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

            total_loss = loss_tensor[0].item()
            loss_components['loss_pos'] = loss_tensor[1].item()
            loss_components['loss_ang'] = loss_tensor[2].item()
            loss_components['loss_wid'] = loss_tensor[3].item()
            loss_components['loss_sem'] = loss_tensor[4].item()
            num_batches = int(loss_tensor[5].item())

        # Average losses
        avg_loss = total_loss / max(num_batches, 1)
        for key in loss_components:
            loss_components[key] /= max(num_batches, 1)

        return {'loss': avg_loss, **loss_components}

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate model with proper distributed loss aggregation."""
        self.model.eval()

        total_loss = 0.0
        loss_components = {'loss_pos': 0, 'loss_ang': 0, 'loss_wid': 0, 'loss_sem': 0}
        num_batches = 0

        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            targets = batch['grasp_mask'].to(self.device)

            if self.use_amp:
                amp_dtype = torch.bfloat16 if self.use_bfloat16 else torch.float16
                with autocast('cuda', dtype=amp_dtype):
                    outputs = self.model(images)
                    losses = self.criterion(outputs['pred_masks'], targets)
            else:
                outputs = self.model(images)
                losses = self.criterion(outputs['pred_masks'], targets)

            total_loss += losses['loss'].item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()
            num_batches += 1

        # Aggregate losses across all GPUs in distributed training
        if self.is_distributed:
            # Create tensor with all losses and batch count
            loss_tensor = torch.tensor(
                [total_loss, loss_components['loss_pos'], loss_components['loss_ang'],
                 loss_components['loss_wid'], loss_components['loss_sem'], num_batches],
                device=self.device
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

            total_loss = loss_tensor[0].item()
            loss_components['loss_pos'] = loss_tensor[1].item()
            loss_components['loss_ang'] = loss_tensor[2].item()
            loss_components['loss_wid'] = loss_tensor[3].item()
            loss_components['loss_sem'] = loss_tensor[4].item()
            num_batches = int(loss_tensor[5].item())

        avg_loss = total_loss / max(num_batches, 1)
        for key in loss_components:
            loss_components[key] /= max(num_batches, 1)

        return {'loss': avg_loss, **loss_components}

    def train(self):
        """Main training loop."""
        num_epochs = self.config.get('epochs', 100)
        save_interval = self.config.get('save_interval', 10)

        if self.is_main_process:
            logger.info(f"Starting training for {num_epochs} epochs...")
            logger.info(f"Total training samples: {len(self.train_loader.dataset)}")
            logger.info(f"Total validation samples: {len(self.val_loader.dataset)}")
            if self.is_distributed:
                logger.info(f"Effective batch size: {self.config.get('batch_size', 4)} x {self.world_size} = {self.config.get('batch_size', 4) * self.world_size}")

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            # Set epoch for distributed sampler (important for shuffling)
            if self.is_distributed and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            # Train
            train_metrics = self.train_epoch()
            self.scheduler.step()

            # Validate
            val_metrics = self.validate()

            # Synchronize before logging
            if self.is_distributed:
                dist.barrier()

            epoch_time = time.time() - epoch_start

            # Logging (only main process)
            if self.is_main_process:
                logger.info(
                    f"Epoch {epoch}/{num_epochs} - Time: {epoch_time:.1f}s - "
                    f"Train Loss: {train_metrics['loss']:.4f} - "
                    f"Val Loss: {val_metrics['loss']:.4f} - "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

            # Wandb epoch logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'epoch/train_loss': train_metrics['loss'],
                    'epoch/train_loss_pos': train_metrics['loss_pos'],
                    'epoch/train_loss_ang': train_metrics['loss_ang'],
                    'epoch/train_loss_wid': train_metrics['loss_wid'],
                    'epoch/train_loss_sem': train_metrics['loss_sem'],
                    'epoch/val_loss': val_metrics['loss'],
                    'epoch/val_loss_pos': val_metrics['loss_pos'],
                    'epoch/val_loss_ang': val_metrics['loss_ang'],
                    'epoch/val_loss_wid': val_metrics['loss_wid'],
                    'epoch/val_loss_sem': val_metrics['loss_sem'],
                    'epoch/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch/time': epoch_time,
                })

            # Save checkpoints (only main process, handled in _save_checkpoint)
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(f'checkpoint_epoch{epoch+1}.pt')

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint('best_model.pt')
                if self.is_main_process:
                    logger.info(f"New best model saved (val_loss: {val_metrics['loss']:.4f})")

        # Save final model
        self._save_checkpoint('final_model.pt')
        if self.is_main_process:
            logger.info("Training completed!")

        # Finish wandb
        if self.use_wandb:
            wandb.finish()

        # Cleanup distributed
        if self.is_distributed:
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='Train Qwen2VL-Grasp model')

    # Data
    parser.add_argument('--dataset_root', type=str, default='/data/myp/grasp_dataset/scenes',
                        help='Path to GraspNet scenes directory')
    parser.add_argument('--output_dir', type=str, default='outputs/qwen2vl_grasp',
                        help='Output directory')

    # Model
    parser.add_argument('--model_type', type=str, default='qwen2vl', choices=['qwen2vl', 'internvit'],
                        help='Model type: qwen2vl (Qwen2-VL ViT) or internvit (InternViT-300M)')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='Pretrained model path (auto-set based on model_type if not specified)')
    parser.add_argument('--freeze_vit', action='store_true', default=False,
                        help='Freeze ViT encoder (default: False, train full model)')
    parser.add_argument('--no_freeze_vit', action='store_true', default=False,
                        help='Explicitly unfreeze ViT encoder')
    parser.add_argument('--decoder_channels', type=int, default=256,
                        help='Decoder hidden channels')

    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # Loss
    parser.add_argument('--loss_type', type=str, default='default', choices=['default', 'focal'])
    parser.add_argument('--pos_weight', type=float, default=5.0)
    parser.add_argument('--ang_weight', type=float, default=5.0)
    parser.add_argument('--focal_alpha', type=float, default=2.0,
                        help='Focal loss alpha (higher = stronger penalty on hard false positives)')
    parser.add_argument('--focal_beta', type=float, default=4.0,
                        help='Focal loss beta (lower = less tolerance for negatives near positives)')
    parser.add_argument('--pos_thresh', type=float, default=0.5,
                        help='Threshold for positive samples (higher = stricter positive definition)')

    # Other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Config file path')

    # Wandb
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='vita-grasp',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (auto-generated if not specified)')

    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training if launched with torchrun with proper validation."""
    required_vars = {'RANK', 'WORLD_SIZE', 'LOCAL_RANK'}
    if not required_vars.issubset(os.environ.keys()):
        return False

    try:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        # Validate MASTER_ADDR and MASTER_PORT (required for multi-node, optional for single-node)
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        if rank == 0:
            logger.info(f"Distributed: MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")

        # Validate rank values
        if not (0 <= rank < world_size):
            raise ValueError(f"Invalid rank: {rank} not in [0, {world_size})")

        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but distributed training requested")

        # Validate local_rank against available GPUs
        num_gpus = torch.cuda.device_count()
        if not (0 <= local_rank < num_gpus):
            raise ValueError(f"Invalid local_rank: {local_rank}, only {num_gpus} GPUs available")

        # Set device first
        torch.cuda.set_device(local_rank)

        # Select backend (nccl for GPU, gloo for CPU fallback)
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'

        # Initialize process group with timeout for slow initialization
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=30),  # Increased timeout for large models
        )

        # Verify initialization
        if not dist.is_initialized():
            raise RuntimeError("Failed to initialize process group")

        if rank == 0:
            logger.info(f"Distributed training initialized: world_size={world_size}, backend={backend}")
            logger.info(f"Using {num_gpus} GPUs on this node")

        return True

    except Exception as e:
        logger.error(f"Failed to setup distributed training: {e}")
        raise


def main():
    args = parse_args()

    # Setup distributed training
    is_distributed = setup_distributed()

    # Load config from file or use args
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = vars(args)

    # Override with command line args
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value

    # Auto-set pretrained_model based on model_type if not specified
    if config.get('pretrained_model') is None:
        model_type = config.get('model_type', 'qwen2vl')
        if model_type == 'internvit':
            config['pretrained_model'] = 'OpenGVLab/InternViT-300M-448px'
        else:
            config['pretrained_model'] = 'Qwen/Qwen2-VL-2B-Instruct'

    # Auto-set output_dir based on model_type if using default
    if config.get('output_dir') == 'outputs/qwen2vl_grasp':
        model_type = config.get('model_type', 'qwen2vl')
        if model_type == 'internvit':
            config['output_dir'] = 'outputs/internvit_grasp'

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
