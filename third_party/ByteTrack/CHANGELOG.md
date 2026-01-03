# ByteTrack-Grasp 修改记录

## 2025-12-18: 优化器改进与训练配置

### 新增: AdamW优化器支持

**文件**: `exps/grasp/yolox_s_grasp_dense.py`

**背景**: SGD默认学习率0.01对于联合训练可能过高，参考SPGrasp使用1e-4到1e-5的学习率

**新增配置**:
```python
# Adam优化器配置 (比SGD使用更小的学习率)
self.use_adam = True  # 使用Adam优化器
self.adam_lr = 1e-4   # Adam基础学习率
self.adam_lr_backbone = 1e-5  # Backbone使用更小的学习率

# 评估间隔
self.eval_interval = 100  # 每100个epoch验证一次
```

**参数分组策略**:
| 参数组 | 学习率 | Weight Decay |
|--------|--------|--------------|
| BatchNorm | adam_lr (1e-4) | 0 |
| Backbone weights | adam_lr_backbone (1e-5) | 5e-4 |
| Head/Neck weights | adam_lr (1e-4) | 5e-4 |
| Bias | adam_lr (1e-4) | 0 |

**学习率调度**:
- Warmup阶段 (前5个epoch): 线性从0上升到adam_lr
- 主训练阶段: Cosine衰减，从adam_lr逐渐降至接近0

**训练指令 (AdamW)**:
```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
    -f exps/grasp/yolox_s_grasp_dense.py \
    -d 1 -b 64 --fp16 -o \
    -c yolox_s.pth.tar
```

**断点续训 (AdamW)**:
```bash
CUDA_VISIBLE_DEVICES=2 python tools/train.py \
    -f exps/grasp/yolox_s_grasp_dense.py \
    -d 1 -b 64 --fp16 \
    --resume \
    -c YOLOX_outputs/yolox_s_grasp_dense/latest_ckpt.pth.tar
```

**切换回SGD**:
在配置文件中设置 `self.use_adam = False`

---

## 2025-12-18: Bug修复与可视化增强

### 修复1: 评估器兼容GraspDense模型输出

**文件**: `yolox/evaluators/coco_evaluator.py`

**问题**: 训练在epoch结束评估时崩溃
```
AttributeError: 'dict' object has no attribute 'new'
```

**原因**: GraspDense模型返回dict格式 `{'det_outputs': tensor, 'grasp_outputs': tensor}`，但评估器期望单独的tensor

**修复**:
```python
outputs = model(imgs)
# 兼容GraspDense模型的dict输出格式
if isinstance(outputs, dict):
    outputs = outputs['det_outputs']
```

### 修复2: 抓取采样边界检查

**文件**: `yolox/models/grasp_dense_head.py` - `sample_grasps_in_bbox()`

**问题**: 部分抓取中心点落在检测框bbox外部

**原因**: 特征图stride离散化导致采样点可能超出bbox边界

**修复**:
```python
# 转换回原图坐标后添加边界检查
x = (fx1 + rx) * stride + stride / 2
y = (fy1 + ry) * stride + stride / 2

# 边界检查：确保抓取中心在bbox内
if x < x1 or x > x2 or y < y1 or y > y2:
    continue
```

### 修复3: 热度图可视化对齐

**文件**: `tools/demo_grasp_track.py` - `visualize_grasp_outputs()`

**问题**: 热度图与RGB图像不对齐（压缩/错位）

**原因**: GraspNet图像720×1280使用letterbox缩放到640×640，热度图160×160只有左上90×160是有效区域

**修复**:
```python
# 计算letterbox后的有效区域
scaled_h = int(h * ratio)
scaled_w = int(w * ratio)
valid_h = int(np.ceil(scaled_h / stride))  # 90
valid_w = int(np.ceil(scaled_w / stride))  # 160

# 只取有效区域再resize
heatmap = heatmap_full[:valid_h, :valid_w]
heatmap_resized = cv2.resize(heatmap_color, (w, h))
```

### 新增: 热度图预处理（阈值截断+高斯模糊）

**文件**: `tools/demo_grasp_track.py` - `visualize_grasp_outputs()`

**功能**: 使热度图可视化更干净，类似SPGrasp效果

**参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `heatmap_threshold` | 0.4 | sigmoid后阈值，低于此值被抑制 |
| `gaussian_kernel` | 11 | 高斯模糊核大小 |

**输出文件**:
```
heatmaps/
├── heatmap_XXXX.jpg      # 处理后（阈值+模糊）
├── heatmap_raw_XXXX.jpg  # 原始热度图
└── grasp_all_XXXX.jpg    # 四通道综合图
```

### 训练指令

**从头训练**:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    -f exps/grasp/yolox_s_grasp_dense.py \
    -d 1 -b 16 --fp16 -o \
    -c yolox_s.pth.tar
```

**断点续训**:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    -f exps/grasp/yolox_s_grasp_dense.py \
    -d 1 -b 16 --fp16 \
    --resume \
    -c YOLOX_outputs/yolox_s_grasp_dense_fixed/latest_ckpt.pth.tar
```

### 评估指令

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
    -f exps/grasp/yolox_s_grasp_dense.py \
    -d 1 -b 16 \
    -c YOLOX_outputs/yolox_s_grasp_dense_fixed/latest_ckpt.pth.tar
```

### 推理指令

```bash
CUDA_VISIBLE_DEVICES=0 python tools/demo_grasp_track.py graspnet \
    -f exps/grasp/yolox_s_grasp_dense.py \
    -c YOLOX_outputs/yolox_s_grasp_dense_fixed/latest_ckpt.pth.tar \
    --path /data/myp/grasp_dataset/scenes \
    --scene_id 100 \
    --conf 0.3 \
    --grasp_topk 5 \
    --save_result
```

### 损失下降记录 (Epoch 11-20)

| Epoch | total_loss | iou_loss | grasp_loss |
|-------|------------|----------|------------|
| 11 | 4.199 | 1.192 | 2.360 |
| 15 | 3.803 | 0.994 | 2.248 |
| 20 | 3.517 | 0.832 | 2.234 |

**下降幅度 (Epoch 11→20)**:
- total_loss: -16.2%
- iou_loss: -30.2%
- grasp_loss: -5.3%

---

## 2025-12-16: GraspNet 抓取检测与跟踪集成

### 新增文件

1. **exps/grasp/yolox_s_grasp_dense.py**
   - YOLOX-S + Dense Grasp 实验配置
   - 单类别检测 (num_classes=1: graspable object)
   - 输入尺寸: 640×640 (固定，禁用多尺度训练)
   - 训练场景: 0-99, 验证场景: 100-129

2. **yolox/models/yolox_grasp.py**
   - YOLOXGrasp: 集成检测+抓取的主模型
   - GraspDenseHead: 密集抓取预测头 (输出160×160)
   - 预测通道: heatmap(1) + cos2θ(1) + sin2θ(1) + width(1) + quality(1) = 5

3. **yolox/models/losses/grasp_loss.py**
   - GraspDenseLoss: 密集抓取损失函数
   - 组件: Focal Loss (heatmap) + Smooth L1 (angle, width, quality)

4. **yolox/data/datasets/graspnet.py**
   - GraspNetDataset: GraspNet数据集加载器
   - 支持 kinect/realsense 相机
   - generate_grasp_dense_gt(): 生成密集抓取GT标签

5. **yolox/utils/grasp_utils.py**
   - sample_grasps_in_bbox(): 从密集预测采样抓取点
   - grasp_nms(): 抓取姿态NMS
   - decode_grasp_angle(): 角度解码 (cos2θ, sin2θ → θ)

6. **tools/demo_grasp_track.py**
   - GraspPredictor: 抓取检测推理器
   - 支持三种模式: image/video/graspnet
   - 集成ByteTrack跟踪
   - 可视化多抓取姿态

### 关键修改

1. **yolox/data/datasets/graspnet.py - 坐标变换修复**
   ```python
   # 修复前 (错误): 不同的x/y缩放比例
   scale_x = out_W / img_W  # 160/1280 = 0.125
   scale_y = out_H / img_H  # 160/720 = 0.222 (错误!)

   # 修复后 (正确): 统一缩放比例，与preproc一致
   r = min(input_H / img_H, input_W / img_W)  # 640/1280 = 0.5
   scale = r * (out_H / input_H)              # 0.5 * 0.25 = 0.125
   scale_x = scale_y = scale                  # 统一比例
   ```
   - 原因: GraspNet图像为720×1280，非512×512
   - preproc使用letterbox缩放保持宽高比

2. **exps/grasp/yolox_s_grasp_dense.py - 禁用多尺度训练**
   ```python
   self.random_size = None  # 禁用多尺度训练，固定640×640输入
   ```
   - 原因: 抓取GT在数据加载时生成，多尺度会导致坐标不匹配

3. **yolox/evaluators/coco_evaluator.py - 兼容5元素batch**
   ```python
   # 修复前
   for cur_iter, (imgs, _, info_imgs, ids) in enumerate(...):

   # 修复后
   for cur_iter, batch_data in enumerate(...):
       imgs, _, info_imgs, ids = batch_data[:4]
   ```
   - 原因: GraspNet返回5个值 (含grasp_data)，标准数据集返回4个

4. **tools/demo_grasp_track.py - 多抓取可视化**
   - 新增 `--grasp_topk` 参数，支持每个检测显示多个抓取点
   - `_combine_det_and_grasp()` 返回 `multi_grasps` 字典
   - `plot_grasp_tracking()` 支持绘制多抓取姿态

### 模型架构

```
Input (640×640×3)
    ↓
YOLOX Backbone (CSPDarknet)
    ↓
YOLOX Neck (PAFPN)
    ↓
┌───────────────────┬────────────────────┐
│   YOLOX Head      │   GraspDenseHead   │
│   (Detection)     │   (Grasp)          │
│   - bbox          │   - heatmap        │
│   - objectness    │   - cos2θ, sin2θ   │
│   - class         │   - width          │
│                   │   - quality        │
└───────────────────┴────────────────────┘
    ↓                       ↓
  80×80, 40×40, 20×20    160×160×5
    ↓                       ↓
┌─────────────────────────────────────────┐
│         ByteTrack Tracker               │
│   - Multi-object tracking               │
│   - Grasp association per track         │
└─────────────────────────────────────────┘
```

### 训练配置

| 参数 | 值 |
|------|-----|
| Backbone | YOLOX-S (depth=0.33, width=0.5) |
| 输入尺寸 | 640×640 (固定) |
| Batch Size | 64 (推荐) |
| Epochs | 100 |
| 优化器 | AdamW (默认) / SGD (可选) |
| 学习率 (AdamW) | 1e-4 (Head), 1e-5 (Backbone) |
| 学习率 (SGD) | 0.01/64 * batch_size |
| 预训练 | yolox_s.pth.tar (COCO) |
| Grasp Loss权重 | 2.0 |
| 评估间隔 | 100 epochs |

### 使用说明

**训练:**
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    -f exps/grasp/yolox_s_grasp_dense.py \
    -d 1 -b 16 --fp16 -o \
    -c yolox_s.pth.tar
```

**推理:**
```bash
python tools/demo_grasp_track.py graspnet \
    -f exps/grasp/yolox_s_grasp_dense.py \
    -c YOLOX_outputs/yolox_s_grasp_dense_fixed/latest_ckpt.pth.tar \
    --path /data/myp/grasp_dataset/scenes \
    --scene_id 100 \
    --conf 0.3 \
    --grasp_topk 5 \
    --save_result
```

### 检测类别

- **num_classes = 1**: 单一类别 "可抓取物体"
- 无需prompt，端到端检测

### 推理性能基准测试

测试环境: NVIDIA RTX 3090 (24GB), CUDA 11.x, PyTorch 2.x

#### 模型推理时间 (仅网络前向)

| 精度 | 最小值 | 最大值 | 平均值 | FPS |
|------|--------|--------|--------|-----|
| FP32 | 11.59 ms | 13.73 ms | 11.89 ms | **84.1** |
| FP16 | 13.21 ms | 16.74 ms | 13.63 ms | 73.4 |

#### 端到端推理时间 (预处理 + 模型 + 后处理 + 跟踪)

| 组件 | 最小值 | 最大值 | 平均值 |
|------|--------|--------|--------|
| 预处理 | 16.72 ms | 32.90 ms | 21.42 ms |
| 模型推理 | 8.21 ms | 16.33 ms | 10.16 ms |
| 后处理(NMS) | 0.91 ms | 147.47 ms* | 1.67 ms |
| ByteTrack跟踪 | 1.07 ms | 3.17 ms | 1.66 ms |
| **总计 (E2E)** | **27.16 ms** | **186.27 ms*** | **34.91 ms** |

*异常值由GC或首帧初始化造成

| 统计量 | 值 | FPS |
|--------|-----|-----|
| 平均值 | 34.91 ms | 28.6 |
| 中位数 | 33.60 ms | **29.8** |
| P95 | 42.98 ms | 23.3 |
| P99 | 47.96 ms | 20.8 |

**结论**: 端到端约 **30 FPS**，满足实时抓取需求

### 依赖

- PyTorch >= 1.7
- YOLOX
- ByteTrack
- GraspNet数据集
