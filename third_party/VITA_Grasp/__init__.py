# VITA-Grasp: ViT-based Foundation Model Baseline for Robotic Grasping
#
# Supports multiple ViT encoders + Grasp Decoder for 4-DOF grasp prediction.
# No LLM module, no temporal modeling - designed for single-frame prediction.
#
# Supported models:
#   - InternViT-Grasp: InternViT-300M encoder (VITA without LLM) [Recommended]
#   - Qwen2VL-Grasp: Qwen2-VL ViT encoder
#
# Usage:
#   from VITA_Grasp.models import InternViTGrasp, build_internvit_grasp
#   from VITA_Grasp.models import Qwen2VLGrasp, build_qwen2vl_grasp
#   from VITA_Grasp.datasets import GraspNetDataset
#   from VITA_Grasp.loss import GraspLoss

__version__ = "0.2.0"
