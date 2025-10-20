# TORE CUDA v0.0.4

TORE (Time-Ordered REpresentation) CUDA实现，支持事件相机的时空表示构建，新增时间窗口滑动功能，适用于长时序事件流的分段处理。

## 版本信息
- 当前版本: 0.0.4
- **新增功能**: 时间窗口滑动（Temporal Sliding Windows）⭐
  - 支持滑动窗口切分长时序事件流
  - 自动尾部处理和views对齐
  - 统一API支持所有resize模式
- 保留功能: v0.0.3的所有功能（正方形crop/padding、按比例resize）
- 向后兼容: 完全兼容v0.0.1、v0.0.2和v0.0.3版本的所有接口

## 安装

```bash
pip install .
```

## 核心概念

TORE将事件流转换为时空体积表示，每个像素位置保留最近的K个事件时间戳，用于后续的深度学习处理。

## v0.0.4 新特性：时间窗口滑动（Temporal Sliding Windows）

### 🎯 为什么需要窗口滑动？

**问题场景**：
```
事件流：100万个事件，持续时间0.105秒
直接处理：构建单个TORE，时序信息粗糙
```

**窗口滑动方案**：
```
窗口长度：20ms
滑动步长：10ms

View 1: [0ms,  20ms]   的事件 → TORE_1
View 2: [10ms, 30ms]   的事件 → TORE_2
View 3: [20ms, 40ms]   的事件 → TORE_3
...
View N: [80ms, 100ms]  的事件 → TORE_N

输出：[Batch, Views, 2K, H, W]
```

**优势**：
- ✅ 更精细的时序建模
- ✅ 适配时序模型（Transformer、LSTM、TCN等）
- ✅ 提取局部时序特征
- ✅ 数据增强（通过窗口重叠）

### 📐 窗口滑动机制

#### 基本参数
- **window_length**：窗口长度（毫秒），如20.0ms
- **stride**：滑动步长（毫秒），如10.0ms
- **tail_threshold**：尾部合并阈值（默认0.5，表示stride的50%）

#### 滑动规则

**标准滑动**：
```
时间范围：0-100ms
窗口长度：20ms
滑动步长：10ms

View 1:  [0,  20)   ✓
View 2:  [10, 30)   ✓
View 3:  [20, 40)   ✓
View 4:  [30, 50)   ✓
View 5:  [40, 60)   ✓
View 6:  [50, 70)   ✓
View 7:  [60, 80)   ✓
View 8:  [70, 90)   ✓
View 9:  [80, 100)  ✓

总计：9个views
```

#### 尾部处理规则

**规则1：剩余时间 < stride × tail_threshold**
```
例如：数据到100.2ms
- 标准窗口：[0, 20), [10, 30), ..., [80, 100)
- 下一窗口起点：90ms
- 剩余时间：100.2 - 90 = 10.2ms
- 判断：10.2ms < 10ms × 0.5？ 否
- 处理：创建最后一个view [90, 100.2]

例如：数据到90.2ms
- 下一窗口起点：80ms
- 剩余时间：90.2 - 80 = 10.2ms
- 判断：10.2ms < 10ms × 0.5？ 否
- 处理：创建最后一个view [80, 90.2]

例如：数据到83ms
- 下一窗口起点：70ms，创建view [70, 83]
- 下一窗口起点：80ms
- 剩余时间：83 - 80 = 3ms
- 判断：3ms < 10ms × 0.5？ 是（3 < 5）
- 处理：不创建新view，保持view [70, 83]
```

**规则核心**：当剩余时间片段过短（< stride的50%）时，不单独成view，避免产生过短的时间窗口。

### 🔄 Batch内Views对齐

**问题场景**：
```
Batch内4个样本：
- Sample 1: 时间跨度100.0ms → 9个views
- Sample 2: 时间跨度100.0ms → 9个views
- Sample 3: 时间跨度100.0ms → 9个views
- Sample 4: 时间跨度105.0ms → 10个views（多出5ms，产生额外view）
```

**对齐策略**：截断到最小值
```python
min_views = min(9, 9, 9, 10) = 9

输出：[4, 9, 2K, H, W]
说明：Sample 4的第10个view被丢弃
```

**信息损失分析**：
- Sample 4损失5ms数据
- 相对于105ms总时长，损失 < 5%
- 换取统一的张量形状，便于批处理

## API接口详解

### 统一的Windowed API（v0.0.4新增）⭐

```python
tore_cuda.tore_build_batch_stacked_windowed(
    events_list: List[torch.Tensor],      # 每个元素[Nei,4]，时间戳单位微秒
    H: int,                               # 输出高度（无resize时使用）
    W: int,                               # 输出宽度（无resize时使用）
    K: int,                               # 每个像素保留的事件数
    window_length: float,                 # 窗口长度（毫秒），如20.0
    stride: float,                        # 滑动步长（毫秒），如10.0
    t0: float,                            # 最小时间间隔（微秒）
    tmax: float,                          # 最大时间间隔（微秒）
    
    # Resize模式控制
    resize_mode: str = "none",            # "none", "resize", "square", "square_crop"
    orig_H_list: Optional[List[int]] = None,  # 原始高度列表（resize模式需要）
    orig_W_list: Optional[List[int]] = None,  # 原始宽度列表（resize模式需要）
    target_H: Optional[int] = None,       # 目标高度（resize模式）
    target_W: Optional[int] = None,       # 目标宽度（resize模式）
    target_size: Optional[int] = None,    # 目标正方形尺寸（square/square_crop模式）
    
    # Windowed参数
    tail_threshold: float = 0.5,          # 尾部合并阈值（stride的比例）
    
    # 输出参数
    out_chw: bool = True,                 # 输出格式
    dtype: Optional[torch.dtype] = None   # 输出数据类型
) -> torch.Tensor  # [B, Views, 2K, H, W] or [B, Views, 2, K, H, W]
```

### Resize模式说明

| resize_mode | 说明 | 需要的参数 | 输出尺寸 | 适用场景 |
|-------------|------|-----------|---------|---------|
| `"none"` | 无resize，使用原始坐标 | `H`, `W` | H×W | 原始分辨率处理 |
| `"resize"` | 按比例resize | `orig_H_list`, `orig_W_list`, `target_H`, `target_W` | target_H×target_W | 自由尺寸调整 |
| `"square"` | 正方形padding模式 | `orig_H_list`, `orig_W_list`, `target_size` | target_size×target_size | 保留全部内容 |
| `"square_crop"` | 正方形crop模式 | `orig_H_list`, `orig_W_list`, `target_size` | target_size×target_size | 充满画面，适合检测/分类 |

### 时间单位说明

- **事件时间戳**：微秒（μs），例如 `1644767900477638`
- **window_length**：毫秒（ms），例如 `20.0` 表示20毫秒
- **stride**：毫秒（ms），例如 `10.0` 表示10毫秒
- **t0, tmax**：微秒（μs），用于TORE的log变换范围

## 使用示例

### 示例1：基础Windowed（无resize）

```python
import torch
import tore_cuda

# 4个事件样本，每个约100ms，100万个事件
events_list = [events1, events2, events3, events4]

# 基础windowed：20ms窗口，10ms步长
tore = tore_cuda.tore_build_batch_stacked_windowed(
    events_list,
    H=720, W=1280, K=4,
    window_length=20.0,  # 20ms
    stride=10.0,         # 10ms
    t0=150.0,
    tmax=5_000_000.0,
    resize_mode="none",
    tail_threshold=0.5,
    out_chw=True
)

print(tore.shape)  # [4, 9, 8, 720, 1280]
# 4个样本，每个9个views，8通道（2*K=2*4），720×1280分辨率
```

### 示例2：Windowed + ViT分类（square_crop模式）⭐

```python
import torch
import tore_cuda

# 事件相机：1280×720
# ViT要求：518×518正方形输入
# 时序建模：每20ms一个view，步长10ms

events_list = load_event_data()  # 加载事件数据
batch_size = len(events_list)

orig_H_list = [720] * batch_size
orig_W_list = [1280] * batch_size

# Windowed + square_crop
tore = tore_cuda.tore_build_batch_stacked_windowed(
    events_list,
    H=720, W=1280, K=8,
    window_length=20.0,
    stride=10.0,
    t0=100.0,
    tmax=1_000_000.0,
    resize_mode="square_crop",  # 充满画面
    orig_H_list=orig_H_list,
    orig_W_list=orig_W_list,
    target_size=518,
    tail_threshold=0.5,
    out_chw=True,
    dtype=torch.bfloat16
)

print(tore.shape)  # [B, Views, 16, 518, 518]
# 例如：[4, 9, 16, 518, 518]

# 方案1：ViT逐view处理 + 时序聚合
B, V, C, H, W = tore.shape
tore_reshaped = tore.view(B * V, C, H, W)  # [36, 16, 518, 518]
vit_features = vit_backbone(tore_reshaped)  # [36, D]
vit_features = vit_features.view(B, V, -1)  # [4, 9, D]

# 时序建模（Transformer/LSTM/TCN）
temporal_output = temporal_model(vit_features)  # [4, num_classes]

# 方案2：直接使用Video Transformer
video_output = video_transformer(tore)  # 输入[B, Views, C, H, W]
```

### 示例3：Windowed + 目标检测

```python
import torch
import tore_cuda

# 目标检测：需要充满画面，使用square_crop

events_list = load_detection_data()

tore = tore_cuda.tore_build_batch_stacked_windowed(
    events_list,
    H=720, W=1280, K=6,
    window_length=20.0,
    stride=10.0,
    t0=150.0,
    tmax=5_000_000.0,
    resize_mode="square_crop",
    orig_H_list=[720] * len(events_list),
    orig_W_list=[1280] * len(events_list),
    target_size=640,  # YOLO输入尺寸
    out_chw=True
)

print(tore.shape)  # [B, Views, 12, 640, 640]

# 逐view检测
B, V, C, H, W = tore.shape
detections_per_view = []
for v in range(V):
    view_tore = tore[:, v, :, :, :]  # [B, C, H, W]
    dets = yolo_detector(view_tore)
    detections_per_view.append(dets)

# 时序关联（tracking）
tracks = temporal_associator(detections_per_view)
```

### 示例4：Windowed + 语义分割（square模式）

```python
import torch
import tore_cuda

# 语义分割：需要完整场景，使用square（padding）模式

events_list = load_segmentation_data()

tore = tore_cuda.tore_build_batch_stacked_windowed(
    events_list,
    H=720, W=1280, K=4,
    window_length=20.0,
    stride=10.0,
    t0=150.0,
    tmax=5_000_000.0,
    resize_mode="square",  # padding模式，保留全部内容
    orig_H_list=[720] * len(events_list),
    orig_W_list=[1280] * len(events_list),
    target_size=512,
    out_chw=True
)

print(tore.shape)  # [B, Views, 8, 512, 512]

# 逐view分割
B, V, C, H, W = tore.shape
for v in range(V):
    view_tore = tore[:, v, :, :, :]
    seg_map = segmentation_model(view_tore)  # [B, num_classes, H, W]
    
    # 需要mask掉padding区域
    # 假设padding在上下，每边约96像素
    padding_mask = create_mask(seg_map.shape, top=96, bottom=96)
    seg_map_masked = seg_map * padding_mask
```

### 示例5：不同窗口参数的对比

```python
import torch
import tore_cuda

events = load_single_sample()  # 100ms的事件流
events_list = [events]

# 配置1：大窗口，小步长（高重叠）
tore1 = tore_cuda.tore_build_batch_stacked_windowed(
    events_list, H=720, W=1280, K=4,
    window_length=50.0,  # 50ms
    stride=10.0,         # 10ms
    t0=150.0, tmax=5_000_000.0,
    resize_mode="none"
)
print(f"大窗口小步长: {tore1.shape}")  # [1, 6, 8, 720, 1280]
# 特点：views少，每个view包含更多时序信息，重叠度高

# 配置2：小窗口，大步长（低重叠）
tore2 = tore_cuda.tore_build_batch_stacked_windowed(
    events_list, H=720, W=1280, K=4,
    window_length=20.0,  # 20ms
    stride=20.0,         # 20ms（无重叠）
    t0=150.0, tmax=5_000_000.0,
    resize_mode="none"
)
print(f"小窗口大步长: {tore2.shape}")  # [1, 5, 8, 720, 1280]
# 特点：views多，每个view时序信息少，无重叠

# 配置3：标准配置（50%重叠）
tore3 = tore_cuda.tore_build_batch_stacked_windowed(
    events_list, H=720, W=1280, K=4,
    window_length=20.0,  # 20ms
    stride=10.0,         # 10ms（50%重叠）
    t0=150.0, tmax=5_000_000.0,
    resize_mode="none"
)
print(f"标准配置: {tore3.shape}")  # [1, 9, 8, 720, 1280]
# 特点：平衡，常用配置
```

## 📊 窗口参数选择指南

### 窗口长度（window_length）选择

| 窗口长度 | Views数量 | 适用场景 | 优缺点 |
|---------|----------|---------|--------|
| **10ms** | 多（~19个） | 快速运动、高帧率需求 | ✅ 时序精细 ❌ 每个view事件少 |
| **20ms** | 中（~9个） | 通用场景（推荐） | ✅ 平衡 |
| **50ms** | 少（~3个） | 慢速场景、降低计算量 | ✅ 计算效率高 ❌ 时序粗糙 |

### 滑动步长（stride）选择

| 步长 | 重叠度 | 适用场景 | 优缺点 |
|------|--------|---------|--------|
| **= window_length** | 0% | 无重叠，节省计算 | ✅ 计算量小 ❌ 可能丢失边界信息 |
| **= window_length / 2** | 50% | 通用推荐 | ✅ 平衡 |
| **< window_length / 2** | >50% | 需要高时序连续性 | ✅ 时序平滑 ❌ 计算量大 |

### 推荐配置

| 任务类型 | window_length | stride | 重叠度 | resize_mode |
|---------|--------------|--------|--------|-------------|
| **分类** | 20ms | 10ms | 50% | square_crop |
| **检测** | 20ms | 10ms | 50% | square_crop |
| **分割** | 20ms | 10ms | 50% | square |
| **跟踪** | 10ms | 5ms | 50% | square_crop |
| **动作识别** | 50ms | 25ms | 50% | square_crop |

## 🎯 实际应用案例

### 案例1：事件相机动作识别

```python
import torch
import tore_cuda
from torch.utils.data import DataLoader

# 数据集：每个样本约1秒（1000ms）的事件流
class ActionDataset:
    def __init__(self, data_root):
        self.samples = load_action_samples(data_root)
    
    def __getitem__(self, idx):
        events, label = self.samples[idx]
        return events, label  # events: [N, 4], label: int

dataset = ActionDataset('/path/to/data')
dataloader = DataLoader(dataset, batch_size=8, collate_fn=custom_collate)

for events_list, labels in dataloader:
    # 滑动窗口：50ms窗口，25ms步长
    # 1000ms / 25ms ≈ 40个views
    tore = tore_cuda.tore_build_batch_stacked_windowed(
        events_list,
        H=260, W=346, K=10,
        window_length=50.0,
        stride=25.0,
        t0=100.0,
        tmax=1_000_000.0,
        resize_mode="square_crop",
        orig_H_list=[260] * len(events_list),
        orig_W_list=[346] * len(events_list),
        target_size=224,
        out_chw=True,
        dtype=torch.float16
    )
    
    # tore.shape: [8, ~40, 20, 224, 224]
    
    # 3D CNN或Video Transformer
    predictions = action_recognition_model(tore)  # [8, num_actions]
    loss = criterion(predictions, labels)
```

### 案例2：实时事件流处理

```python
import torch
import tore_cuda
import queue
import threading

class RealtimeEventProcessor:
    def __init__(self, camera, model):
        self.camera = camera
        self.model = model
        self.event_buffer = []
        self.window_length = 20.0  # ms
        self.stride = 10.0  # ms
        
    def process_stream(self):
        while True:
            # 累积事件
            new_events = self.camera.get_events()  # [M, 4]
            self.event_buffer.append(new_events)
            
            # 合并事件
            events = torch.cat(self.event_buffer, dim=0)
            
            # 检查时间跨度
            time_span = (events[:, 2].max() - events[:, 2].min()).item() / 1000.0  # μs → ms
            
            if time_span >= 100.0:  # 累积100ms
                # 构建windowed TORE
                tore = tore_cuda.tore_build_batch_stacked_windowed(
                    [events],
                    H=480, W=640, K=4,
                    window_length=self.window_length,
                    stride=self.stride,
                    t0=150.0,
                    tmax=500_000.0,
                    resize_mode="square_crop",
                    orig_H_list=[480],
                    orig_W_list=[640],
                    target_size=224,
                    out_chw=True
                )
                
                # 推理
                result = self.model(tore)  # [1, Views, ...]
                
                # 处理结果
                self.handle_result(result)
                
                # 滑动缓冲区：保留最后50ms
                keep_time = events[:, 2].max() - 50000  # 50ms in μs
                self.event_buffer = [events[events[:, 2] > keep_time]]
```

## ⚠️ 常见问题和解决方案

### 问题1：Views数量不一致导致截断

```python
# 问题：4个样本产生了不同数量的views
events_list = [events1, events2, events3, events4]
# events1-3: 100.0ms → 9 views
# events4: 105.0ms → 10 views

tore = tore_cuda.tore_build_batch_stacked_windowed(
    events_list, H=720, W=1280, K=4,
    window_length=20.0, stride=10.0,
    t0=150.0, tmax=5_000_000.0,
    resize_mode="none"
)
print(tore.shape)  # [4, 9, 8, 720, 1280]
# events4的第10个view被截断

# 解决方案：确保数据集的时间跨度一致
# 在数据预处理时统一裁剪到相同时长
```

### 问题2：窗口内事件过少

```python
# 问题：窗口太小，某些窗口内事件很少
tore = tore_cuda.tore_build_batch_stacked_windowed(
    events_list, H=720, W=1280, K=4,
    window_length=1.0,  # 1ms，太小！
    stride=1.0,
    t0=150.0, tmax=5_000_000.0,
    resize_mode="none"
)

# 解决方案：
# 1. 增大窗口长度（如20ms）
# 2. 检查事件密度，确保每ms有足够事件
# 3. 使用更小的K值
```

### 问题3：内存溢出

```python
# 问题：Views太多，导致内存溢出
# 1000ms / 1ms = 1000 views，太多！

# 解决方案1：增大stride，减少views数量
tore = tore_cuda.tore_build_batch_stacked_windowed(
    events_list, H=720, W=1280, K=4,
    window_length=20.0,
    stride=20.0,  # 从10ms增大到20ms
    t0=150.0, tmax=5_000_000.0,
    resize_mode="none"
)

# 解决方案2：减小batch_size
# 从batch=16减少到batch=4

# 解决方案3：使用较小的分辨率
tore = tore_cuda.tore_build_batch_stacked_windowed(
    events_list, H=720, W=1280, K=4,
    window_length=20.0, stride=10.0,
    t0=150.0, tmax=5_000_000.0,
    resize_mode="resize",
    orig_H_list=[720] * len(events_list),
    orig_W_list=[1280] * len(events_list),
    target_H=360,  # 降低分辨率
    target_W=640
)
```

### 问题4：尾部处理不符合预期

```python
# 问题：最后一个view时长不一致

# 理解tail_threshold参数
# tail_threshold=0.5（默认）：
#   - 剩余时间 < stride * 0.5时，不创建新view
#   - 剩余时间 >= stride * 0.5时，创建新view

# 例子：
# window_length=20ms, stride=10ms, tail_threshold=0.5
# 数据时长：103ms

# 计算：
# Views: [0,20), [10,30), [20,40), ..., [80,100)
# 下一窗口起点：90ms
# 剩余：103-90=13ms
# 13ms >= 10*0.5=5ms，创建view [90,103]

# 如果想要更严格的尾部控制：
tore = tore_cuda.tore_build_batch_stacked_windowed(
    events_list, H=720, W=1280, K=4,
    window_length=20.0, stride=10.0,
    tail_threshold=0.8,  # 提高阈值，更倾向于不创建短尾部
    t0=150.0, tmax=5_000_000.0,
    resize_mode="none"
)
```

## 📋 API总览

### v0.0.4完整API列表（16个接口）

**基础接口（3个，v0.0.1）**：
- `tore_build_single`
- `tore_build_batch`
- `tore_build_batch_stacked`

**按比例resize接口（3个，v0.0.2）**：
- `tore_build_single_resized`
- `tore_build_batch_resized`
- `tore_build_batch_stacked_resized`

**正方形Padding模式接口（3个，v0.0.2）**：
- `tore_build_single_square`
- `tore_build_batch_square`
- `tore_build_batch_stacked_square`

**正方形Crop模式接口（3个，v0.0.3）**：
- `tore_build_single_square_crop`
- `tore_build_batch_square_crop`
- `tore_build_batch_stacked_square_crop`

**时间窗口滑动接口（1个，v0.0.4新增）** ⭐：
- `tore_build_batch_stacked_windowed`（统一支持所有resize模式）

## 版本升级建议

### 从v0.0.3升级到v0.0.4

**何时使用windowed模式**：
- ✅ 事件流时长 > 50ms
- ✅ 需要时序建模（Transformer/LSTM）
- ✅ 视频级别的任务（动作识别、跟踪）
- ✅ 需要提取局部时序特征

**何时不使用windowed模式**：
- ❌ 事件流时长 < 20ms（太短，分不出views）
- ❌ 纯空间任务（如单帧分类）
- ❌ 实时性要求极高（windowed有额外开销）

**迁移示例**：

```python
# v0.0.3: 单个TORE
tore = tore_cuda.tore_build_batch_stacked_square_crop(
    events_list, orig_H_list, orig_W_list, 518,
    K=4, t0=150.0, tmax=5_000_000.0,
    out_chw=True
)
# 输出：[B, 8, 518, 518]

# v0.0.4: 时序TORE
tore = tore_cuda.tore_build_batch_stacked_windowed(
    events_list, H=720, W=1280, K=4,
    window_length=20.0, stride=10.0,
    t0=150.0, tmax=5_000_000.0,
    resize_mode="square_crop",
    orig_H_list=orig_H_list,
    orig_W_list=orig_W_list,
    target_size=518,
    out_chw=True
)
# 输出：[B, Views, 8, 518, 518]
```

**完全向后兼容**：
- 所有v0.0.1、v0.0.2和v0.0.3的接口保持不变
- 可以无缝升级，逐步尝试windowed功能

