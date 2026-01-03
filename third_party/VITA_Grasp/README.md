# VITA-Grasp

**ViT Encoder + Grasp Decoder for 4-DOF Grasp Prediction**

A foundation model baseline for robotic grasping using pretrained ViT encoders. Supports multiple ViT backends including InternViT (VITA without LLM) and Qwen2-VL ViT.

## Supported Models

| Model | ViT Encoder | Parameters | Input Size | Recommended |
|-------|-------------|------------|------------|-------------|
| **InternViT-Grasp** | InternViT-300M | ~300M | 448×448 | ✓ |
| Qwen2VL-Grasp | Qwen2-VL ViT | ~675M | Dynamic | |

**InternViT-Grasp = VITA without LLM**: Uses VITA's visual encoder for dense grasp prediction.

## Architecture

```
Image (H, W, 3)
    │
    ▼
┌─────────────────────────────────────┐
│  ViT Encoder (frozen)               │
│  - InternViT-300M (~300M)           │
│  - or Qwen2-VL ViT (~675M)          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Grasp Decoder (~5-8M)              │
│  - FPN-style upsampling             │
│  - 5-channel output heads           │
└─────────────────────────────────────┘
    │
    ▼
5-Channel Output (H, W, 5):
  - Channel 0: Position heatmap
  - Channel 1: cos(2θ) angle
  - Channel 2: sin(2θ) angle
  - Channel 3: Grasp width
  - Channel 4: Semantic mask
```

## Features

- **Multiple ViT Backends**: InternViT-300M (recommended) or Qwen2-VL ViT
- **Lightweight Decoder**: Only ~5M trainable parameters when ViT is frozen
- **Single-frame Prediction**: No temporal modeling, suitable for fair comparison
- **5-Channel Output**: Compatible with SPGrasp annotation format

## Requirements

```bash
pip install torch>=2.0 transformers>=4.37.0 opencv-python matplotlib pyyaml
pip install flash-attn --no-build-isolation  # Optional, for faster attention
```

## Quick Start

### Training with InternViT (Recommended)

```bash
# Using InternViT-300M (VITA without LLM)
python scripts/train.py \
    --model_type internvit \
    --dataset_root /data/myp/grasp_dataset/scenes \
    --output_dir outputs/internvit_grasp

# Or using Qwen2-VL ViT
python scripts/train.py \
    --model_type qwen2vl \
    --output_dir outputs/qwen2vl_grasp
```

### Using Config File

```bash
python scripts/train.py --config configs/internvit_grasp.yaml
```

### Inference

```bash
python scripts/inference.py \
    --checkpoint outputs/internvit_grasp/best_model.pt \
    --image_path test_image.png \
    --output_dir outputs/predictions
```

## Configuration

**configs/internvit_grasp.yaml**:
```yaml
model:
  type: "internvit"
  pretrained_model: "OpenGVLab/InternViT-300M-448px"
  freeze_vit: true
  decoder_channels: 256
  output_size: [480, 640]

training:
  batch_size: 8
  epochs: 100
  learning_rate: 1.0e-4
```

## Model Comparison

| Model | ViT Params | Decoder Params | Total Trainable | Speed |
|-------|------------|----------------|-----------------|-------|
| InternViT-Grasp | 300M (frozen) | ~5M | ~5M | Fast |
| Qwen2VL-Grasp | 675M (frozen) | ~8M | ~8M | Medium |
| SPGrasp | 650M (frozen) | ~10M | ~10M | Medium |

## Dataset

Uses GraspNet-1Billion dataset:

```
scenes/
├── scene_0000/
│   └── kinect/
│       ├── rgb/      # RGB images
│       ├── label/    # Instance masks
│       └── rect/     # Grasp annotations
├── scene_0001/
│   └── ...
```

## Citation

```bibtex
@article{spgrasp2024,
  title={SPGrasp: Spatiotemporal Prompt-driven Grasp Synthesis in Dynamic Scenes},
  author={...},
  year={2024}
}

@article{internvit2024,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and others},
  year={2024}
}
```

## License

Apache 2.0
