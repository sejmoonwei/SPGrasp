# ByteTrack-Grasp 使用说明

基于ByteTrack的多目标抓取检测与跟踪框架。

## 项目结构

```
ByteTrack/
├── exps/grasp/
│   └── yolox_s_grasp_dense.py    # 训练配置文件
├── tools/
│   ├── train.py                   # 训练入口
│   └── demo_grasp_track.py        # 推理演示
├── yolox/
│   ├── models/
│   │   ├── grasp_dense_head.py   # 密集抓取预测头
│   │   └── yolox_grasp.py        # YOLOX-Grasp模型
│   ├── data/datasets/
│   │   └── graspnet.py           # GraspNet数据集加载
│   ├── core/
│   │   └── trainer.py            # GraspTrainer训练器
│   └── tracker/
│       └── byte_tracker.py       # ByteTrack跟踪器
└── YOLOX_outputs/                 # 训练输出目录
```

## 环境依赖

```bash
pip install torch torchvision
pip install loguru tabulate tensorboard
pip install opencv-python numpy
pip install cython_bbox  # 用于跟踪
```

## 数据集

使用 GraspNet-1Billion 数据集：
- 数据路径: `/data/myp/grasp_dataset/scenes/`
- 训练集: scene_0000 ~ scene_0099 (100个场景)
- 验证集: scene_0100 ~ scene_0129 (30个场景)

数据结构：
```
scenes/
├── scene_0000/
│   └── kinect/
│       ├── rgb/          # RGB图像 (720x1280)
│       ├── label/        # 实例分割mask
│       └── rect/         # 抓取标注 (.npy)
├── scene_0001/
...
```

---

## 训练

### 单GPU训练

```bash
cd /data/myp/sam2/sam2grasp/third_party/ByteTrack

# 基础训练 (GPU 0, batch_size=16)
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    -f exps/grasp/yolox_s_grasp_dense.py \
    -d 1 \
    -b 16 \
    --fp16 \
    -expn yolox_s_grasp_dense_bs16

# 参数说明:
#   -f: 实验配置文件
#   -d: GPU数量
#   -b: batch size
#   --fp16: 混合精度训练
#   -expn: 实验名称 (输出目录名)
```

### 多GPU训练 (DDP)

```bash
# 4卡训练 (GPU 0,2,4,6)
CUDA_VISIBLE_DEVICES=0,2,4,6 python tools/train.py \
    -f exps/grasp/yolox_s_grasp_dense.py \
    -d 4 \
    -b 64 \
    --fp16 \
    -expn yolox_s_grasp_dense_4gpu

# 或使用 torchrun
CUDA_VISIBLE_DEVICES=0,2,4,6 torchrun --nproc_per_node=4 tools/train.py \
    -f exps/grasp/yolox_s_grasp_dense.py \
    -d 4 \
    -b 64 \
    --fp16 \
    -expn yolox_s_grasp_dense_4gpu
```

### 恢复训练

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    -f exps/grasp/yolox_s_grasp_dense.py \
    -d 1 \
    -b 16 \
    --fp16 \
    -expn yolox_s_grasp_dense_bs16 \
    --resume \
    -c YOLOX_outputs/yolox_s_grasp_dense_bs16/latest_ckpt.pth.tar
```

### 训练输出

```
YOLOX_outputs/yolox_s_grasp_dense_bs16/
├── best_ckpt.pth.tar      # 最佳模型
├── latest_ckpt.pth.tar    # 最新checkpoint
├── train_log.txt          # 训练日志
└── events.out.*           # TensorBoard日志
```

### 监控训练

```bash
# 查看日志
tail -f YOLOX_outputs/yolox_s_grasp_dense_bs16/train_log.txt

# TensorBoard
tensorboard --logdir YOLOX_outputs/yolox_s_grasp_dense_bs16 --port 6006
```

---

## 推理

### 单场景推理

```bash
cd /data/myp/sam2/sam2grasp/third_party/ByteTrack

CUDA_VISIBLE_DEVICES=0 python tools/demo_grasp_track.py \
    -f exps/grasp/yolox_s_grasp_dense.py \
    -c YOLOX_outputs/yolox_s_grasp_dense_bs16/best_ckpt.pth.tar \
    --scene_id 100 \
    --save_result

# 参数说明:
#   -f: 实验配置文件
#   -c: 模型权重路径
#   --scene_id: 场景ID (0-189)
#   --save_result: 保存可视化结果
#   --fp16: 使用FP16推理 (可选，加速)
```

### 输出格式

推理结果保存在 `YOLOX_outputs/` 目录：
- 可视化视频/图像
- 检测结果JSON (bbox + 抓取参数)

---

## 配置文件说明

`exps/grasp/yolox_s_grasp_dense.py` 主要配置：

```python
# 模型配置
self.num_classes = 1      # 单类别: 可抓取物体
self.depth = 0.33         # YOLOX-S
self.width = 0.50         # YOLOX-S

# 训练配置
self.max_epoch = 100
self.eval_interval = 100  # 每100个epoch验证一次
self.ckpt_interval = 5    # 每5个epoch保存一次权重
self.warmup_epochs = 5

# AdamW优化器配置 (推荐)
self.use_adam = True      # 使用AdamW优化器
self.adam_lr = 1e-4       # Head/Neck学习率
self.adam_lr_backbone = 1e-5  # Backbone学习率 (更小，微调)

# SGD配置 (备用，设置use_adam=False启用)
self.basic_lr_per_img = 0.01 / 64.0
self.weight_decay = 5e-4
self.momentum = 0.9

# 数据配置
self.data_num_workers = 4   # 数据加载线程 (过大会OOM)
self.data_dir = "/data/myp/grasp_dataset/scenes"
self.train_scenes = (0, 100)    # 训练场景范围
self.val_scenes = (100, 130)    # 验证场景范围
self.input_size = (640, 640)
self.random_size = None         # None=固定尺寸, (14,26)=多尺度训练

# 抓取损失配置
self.grasp_loss_weight = 2.0
self.grasp_heatmap_weight = 1.0
self.grasp_angle_weight = 1.0
self.grasp_width_weight = 1.0
self.grasp_quality_weight = 0.5
```

### 学习率调度 (AdamW)

| 阶段 | Epochs | 调度方式 |
|------|--------|----------|
| Warmup | 1-5 | 线性从0上升到adam_lr |
| 主训练 | 6-100 | Cosine衰减到 min_lr_ratio (5%) |

### 参数分组策略

| 参数组 | 学习率 | Weight Decay |
|--------|--------|--------------|
| BatchNorm | adam_lr (1e-4) | 0 |
| Backbone weights | adam_lr_backbone (1e-5) | 5e-4 |
| Head/Neck weights | adam_lr (1e-4) | 5e-4 |
| Bias | adam_lr (1e-4) | 0 |

---

## 模型架构

```
输入图像 (640x640)
    │
    ▼
┌─────────────┐
│  Backbone   │  (CSPDarknet)
│  (YOLOX-S)  │
└─────────────┘
    │
    ▼
┌─────────────┐
│    PAFPN    │  (特征金字塔)
└─────────────┘
    │
    ├──────────────────┐
    ▼                  ▼
┌─────────────┐  ┌─────────────┐
│ YOLOXHead   │  │GraspDenseHead│
│ (检测)      │  │ (抓取预测)   │
└─────────────┘  └─────────────┘
    │                  │
    ▼                  ▼
  BBox              Grasp GT
  [x,y,w,h,conf]    - heatmap (160x160)
                    - cos2θ, sin2θ
                    - width
                    - quality
```

---

## 常见问题

### 1. CUDA内存不足
```bash
# 减小batch size
-b 8  # 或更小
```

### 2. 训练速度慢
```bash
# 增加数据加载线程 (修改配置文件)
self.data_num_workers = 8  # 或更多
```

### 3. 多GPU训练失败
确保使用的GPU都可用：
```bash
nvidia-smi  # 检查GPU状态
CUDA_VISIBLE_DEVICES=0,1,2,3 python ...  # 指定可用GPU
```

### 4. np.float 弃用警告
已修复，使用 `np.float64` 替代。

---

## 预训练权重

推理用权重位置: `YOLOX_outputs/ckpts_for_inf/`

| 文件名 | 优化器 | Epoch | 说明 |
|--------|--------|-------|------|
| `sgd_epoch20.pth.tar` | SGD | 20 | SGD训练20个epoch |
| `adam_epoch6_v1.pth.tar` | Adam | 6 | Adam训练6个epoch (v1) |
| `adam_epoch6_v2.pth.tar` | Adam | 6 | Adam训练6个epoch (v2) |

### 训练指令

**从零开始训练 (AdamW)**:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py -f exps/grasp/yolox_s_grasp_dense.py -d 1 -b 16 --fp16 -expn yolox_s_grasp_adam
```

**从预训练权重微调 (加载权重，重新初始化AdamW优化器)**:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py -f exps/grasp/yolox_s_grasp_dense.py -d 1 -b 16 --fp16 -c YOLOX_outputs/ckpts_for_inf/sgd_epoch20.pth.tar -expn yolox_s_grasp_adam_finetune
```

**恢复训练 (加载权重+优化器状态，继续训练)**:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py -f exps/grasp/yolox_s_grasp_dense.py -d 1 -b 16 --fp16 --resume -c YOLOX_outputs/xxx/latest_ckpt.pth.tar
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `-c` | 加载权重，重新初始化optimizer |
| `--resume -c` | 恢复训练，加载权重+optimizer状态 |
| `-expn` | 实验名称，输出目录名 |
| `-b` | batch size (推荐16，过大会OOM) |
| `--fp16` | 混合精度训练 |

---

## 参考

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [GraspNet-1Billion](https://graspnet.net/)
