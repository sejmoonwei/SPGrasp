# Deprecated Sparse Architecture

This directory contains the deprecated sparse grasp prediction architecture.

## Files

- `grasp_yolo_head.py`: Sparse GraspYOLOXHead (per-anchor grasp prediction)
- `grasp_losses.py`: Loss functions for sparse architecture

## Status

These files are **deprecated** and kept only for reference.

The current recommended architecture is the **dense grasp prediction**:
- `grasp_dense_head.py`: GraspDenseHead (per-pixel grasp prediction)
- `yolox_grasp.py`: YOLOXGrasp integrated model

## Migration

If you were using the sparse architecture, please migrate to the dense architecture:

```python
# Old (sparse) - DEPRECATED
from yolox.models import GraspYOLOXHead, GraspLoss

# New (dense) - RECOMMENDED
from yolox.models import (
    YOLOXGrasp,
    build_yolox_grasp,
    GraspDenseHead,
    GraspDenseLoss,
    sample_grasps_in_bbox,
)

# Build model
model = build_yolox_grasp(num_classes=1, with_grasp_head=True)
```
