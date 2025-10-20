# TORE CUDA v0.0.3

TORE (Time-Ordered REpresentation) CUDA实现，支持事件相机的时空表示构建，新增灵活的正方形resize策略。

## 版本信息
- 当前版本: 0.0.3
- **新增功能**: 正方形resize的crop模式（以短边为基准，长边中心裁剪）
- 保留功能: v0.0.2的所有功能（按比例resize、正方形padding模式）
- 向后兼容: 完全兼容v0.0.1和v0.0.2版本的所有接口

## 安装

```bash
pip install .
```

## 核心概念

TORE将事件流转换为时空体积表示，每个像素位置保留最近的K个事件时间戳，用于后续的深度学习处理。

## v0.0.3 新特性：两种正方形Resize策略

### 🆚 正方形Resize策略对比

| 特性 | `*_square` (v0.0.2) | `*_square_crop` (v0.0.3新增) |
|------|---------------------|------------------------------|
| **缩放基准** | 以宽度为基准 | 以短边为基准 |
| **长边处理** | Padding（填充空白） | Crop（中心裁剪） |
| **内容覆盖** | 部分区域为空 | 充满整个区域 |
| **信息损失** | 无损失（可能有空白） | 长边信息被裁剪 |
| **适用场景** | 需要保留全部内容 | 需要充满画面，可接受裁剪 |

### 📐 Resize策略图解

假设原始尺寸：1280×720（宽×高），目标尺寸：518×518

**策略1: `*_square` (以宽度为基准)**
```
原始: 1280×720  →  缩放: 518×291  →  padding: 518×518
                                      (上下各填充113.5像素)
结果：完整保留所有内容，但有空白区域
```

**策略2: `*_square_crop` (以短边为基准，v0.0.3新增)**
```
原始: 1280×720  →  缩放: 920×518  →  中心crop: 518×518
                                      (左右各裁剪201像素)
结果：画面充满，但左右边缘被裁剪
```

## API选择指南

### 🔍 如何选择合适的API接口？

根据您的使用场景，选择最适合的API接口：

| 场景 | 推荐API | 说明 |
|------|---------|------|
| **单张图像处理** | `tore_build_single` | 处理单个事件样本，返回单个张量 |
| **批量处理但需要单独访问每个结果** | `tore_build_batch` | 批量处理，返回张量列表，可单独访问每个结果 |
| **批量处理且需要堆叠结果** | `tore_build_batch_stacked` | 批量处理，直接返回堆叠好的batch张量，效率最高 |
| **需要resize到固定尺寸** | `tore_build_*_resized` | 按比例resize事件坐标 |
| **需要正方形输出（保留全部内容）** | `tore_build_*_square` | 正方形TORE表示（可能有padding） |
| **需要正方形输出（充满画面）** | `tore_build_*_square_crop` | 正方形TORE表示（长边裁剪）⭐v0.0.3新增 |

### 📊 接口选择决策树

```
需要resize吗？
├─ 是 → 需要正方形输出吗？
│   ├─ 是 → 需要保留全部内容还是充满画面？
│   │   ├─ 保留全部内容（可接受padding） → 使用 *_square 系列接口
│   │   └─ 充满画面（可接受裁剪） → 使用 *_square_crop 系列接口 ⭐v0.0.3
│   └─ 否 → 使用 *_resized 系列接口
└─ 否 → 使用基础接口（无resize）
    ├─ 单样本 → tore_build_single
    ├─ 批量但需单独结果 → tore_build_batch
    └─ 批量且需堆叠 → tore_build_batch_stacked
```

## API接口详解

### 1. 基础单样本处理（v0.0.1保留）

**使用场景**：
- ✅ **实时推理**：处理单个事件流样本
- ✅ **调试开发**：快速验证算法效果
- ✅ **小数据集实验**：数据量较小时避免批处理开销
- ✅ **内存受限**：GPU内存不足时逐个处理

**不适用场景**：
- ❌ **大批量训练**：效率低于批处理接口
- ❌ **需要梯度回传**：单样本无法利用批归一化等操作

```python
tore_cuda.tore_build_single(
    events: torch.Tensor,      # [N,4] CUDA tensor, 格式 (x,y,t,p)
    H: int,                    # 输出高度
    W: int,                    # 输出宽度
    K: int,                    # 每个像素保留的事件数
    t0: float,                 # 最小时间间隔（秒）
    tmax: float,               # 最大时间间隔（秒）
    t_query: Optional[int],    # 查询时间点（可选，默认使用最大时间）
    out_chw: bool,             # 输出格式：True->(2K,H,W), False->(2,K,H,W)
    dtype: Optional[torch.dtype]  # 输出数据类型（可选，默认torch.float32）
) -> torch.Tensor
```

**返回**: TORE表示张量，形状取决于out_chw参数和输入数据类型

### 2. 基础批量处理（v0.0.1保留）

**使用场景**：
- ✅ **训练流水线**：需要单独访问每个样本的结果
- ✅ **数据增强**：对每个样本进行不同后处理
- ✅ **异构处理**：不同样本需要不同后续操作
- ✅ **调试分析**：需要检查单个样本的中间结果

**不适用场景**：
- ❌ **需要张量运算**：返回的是列表，需要手动stack
- ❌ **GPU内存紧张**：列表形式无法优化内存布局

```python
tore_cuda.tore_build_batch(
    events_list: List[torch.Tensor],  # 事件列表，每个元素[N_i,4]
    H: int,                           # 输出高度
    W: int,                           # 输出宽度
    K: int,                           # 每个像素保留的事件数
    t0: float,                        # 最小时间间隔（秒）
    tmax: float,                      # 最大时间间隔（秒）
    t_query_list: Optional[List[int]], # 每个样本的查询时间点（可选）
    out_chw: bool,                    # 输出格式
    dtype: Optional[torch.dtype]      # 输出数据类型（可选）
) -> List[torch.Tensor]
```

**返回**: TORE表示张量列表，每个元素对应一个输入样本

### 3. 基础批量堆叠处理（v0.0.1保留）

**使用场景**：
- ✅ **深度学习训练**：直接得到batch张量，适配DataLoader
- ✅ **CNN推理**：批量输入到卷积网络
- ✅ **GPU效率优化**：连续内存布局，计算效率最高
- ✅ **张量运算**：可直接进行batch级别的矩阵运算
- ✅ **生产部署**：标准的批处理流水线

**不适用场景**：
- ❌ **需要单独处理每个样本**：返回的是堆叠张量
- ❌ **样本需要不同后处理**：所有样本必须统一处理

```python
tore_cuda.tore_build_batch_stacked(
    events_list: List[torch.Tensor],  # 事件列表，每个元素[N_i,4]
    H: int,                           # 输出高度
    W: int,                           # 输出宽度
    K: int,                           # 每个像素保留的事件数
    t0: float,                        # 最小时间间隔（秒）
    tmax: float,                      # 最大时间间隔（秒）
    t_query_list: Optional[List[int]], # 每个样本的查询时间点（可选）
    out_chw: bool,                    # 输出格式
    dtype: Optional[torch.dtype]      # 输出数据类型（可选）
) -> torch.Tensor
```

**返回**: 堆叠的TORE表示张量，形状(B, 2K, H, W)或(B, 2, K, H, W)

### 4. 带resize的单样本处理（v0.0.2保留）

**使用场景**：
- ✅ **模型输入标准化**：不同分辨率事件相机统一尺寸
- ✅ **内存优化**：降低高分辨率事件的内存占用
- ✅ **预处理管道**：单个样本的resize和TORE构建一体化
- ✅ **算法对比**：测试不同输入尺寸对模型性能的影响

**不适用场景**：
- ❌ **向上resize**：只支持向下resize（目标尺寸 ≤ 原始尺寸）
- ❌ **需要保持原始坐标**：resize会改变事件坐标

```python
tore_cuda.tore_build_single_resized(
    events: torch.Tensor,      # [N,4] CUDA tensor, 格式 (x,y,t,p)
    orig_H: int,               # 原始输入高度
    orig_W: int,               # 原始输入宽度
    target_H: int,             # 目标输出高度
    target_W: int,             # 目标输出宽度
    K: int,                    # 每个像素保留的事件数
    t0: float,                 # 最小时间间隔（秒）
    tmax: float,               # 最大时间间隔（秒）
    t_query: Optional[int],    # 查询时间点（可选）
    out_chw: bool,             # 输出格式
    dtype: Optional[torch.dtype]  # 输出数据类型（可选）
) -> torch.Tensor
```

**特点**:
- 支持向下resize（目标尺寸 ≤ 原始尺寸）
- 事件坐标按比例缩放
- 过滤超出目标范围的事件
- **自动类型转换**：支持int32、float32等多种输入类型，自动转换为float32进行resize计算

### 5. 带正方形resize的单样本处理 - Padding模式（v0.0.2保留）

**使用场景**：
- ✅ **完整内容保留**：需要保留事件流的所有空间信息
- ✅ **语义分割任务**：需要完整的场景信息
- ✅ **低高度事件流**：原始高度较小，避免进一步裁剪
- ✅ **后续需要反向映射**：padding区域可以被标记和处理

**不适用场景**：
- ❌ **密集预测任务**：padding区域会影响预测结果
- ❌ **需要充满画面**：会有空白区域

```python
tore_cuda.tore_build_single_square(
    events: torch.Tensor,      # [N,4] CUDA tensor, 格式 (x,y,t,p)
    orig_H: int,               # 原始输入高度
    orig_W: int,               # 原始输入宽度
    target_size: int,          # 目标正方形边长（如518）
    K: int,                    # 每个像素保留的事件数
    t0: float,                 # 最小时间间隔（秒）
    tmax: float,               # 最大时间间隔（秒）
    t_query: Optional[int],    # 查询时间点（可选）
    out_chw: bool,             # 输出格式
    dtype: Optional[torch.dtype]  # 输出数据类型（可选）
) -> torch.Tensor
```

**特点**:
- 以宽度为基准进行等比例缩放
- 高度>目标尺寸：中心crop
- 高度<目标尺寸：上下填充空白区域
- 自动过滤无效坐标

### 6. 带正方形resize的单样本处理 - Crop模式（v0.0.3新增）⭐

**使用场景**：
- ✅ **目标检测**：充满画面，提高检测效率
- ✅ **图像分类**：最大化利用输入区域
- ✅ **注意力机制**：避免padding区域干扰attention
- ✅ **高宽比接近1:1的场景**：裁剪损失较小
- ✅ **ViT模型**：充分利用patch区域，无空白

**不适用场景**：
- ❌ **需要完整场景信息**：长边会被裁剪
- ❌ **极端宽高比**：裁剪损失过大（如16:9变1:1）

```python
tore_cuda.tore_build_single_square_crop(
    events: torch.Tensor,      # [N,4] CUDA tensor, 格式 (x,y,t,p)
    orig_H: int,               # 原始输入高度
    orig_W: int,               # 原始输入宽度
    target_size: int,          # 目标正方形边长（如518）
    K: int,                    # 每个像素保留的事件数
    t0: float,                 # 最小时间间隔（秒）
    tmax: float,               # 最大时间间隔（秒）
    t_query: Optional[int],    # 查询时间点（可选）
    out_chw: bool,             # 输出格式
    dtype: Optional[torch.dtype]  # 输出数据类型（可选）
) -> torch.Tensor
```

**特点**:
- 以短边为基准进行等比例缩放
- 长边自动中心裁剪到目标尺寸
- 无padding，画面充满整个输出区域
- 自动过滤裁剪区域外的事件

### 7-12. 批量处理接口

类似的，v0.0.3新增了3个批量处理接口，对应crop模式：

- `tore_build_batch_square_crop` - 批量处理，返回列表
- `tore_build_batch_stacked_square_crop` - 批量处理，返回堆叠张量 ⭐推荐

以及v0.0.2保留的接口：
- `tore_build_batch_resized` / `tore_build_batch_stacked_resized`
- `tore_build_batch_square` / `tore_build_batch_stacked_square`

**批量堆叠接口示例**（crop模式，v0.0.3新增）：

```python
tore_cuda.tore_build_batch_stacked_square_crop(
    events_list: List[torch.Tensor],  # 事件列表，每个元素[N_i,4]
    orig_H_list: List[int],           # 每个样本的原始高度
    orig_W_list: List[int],           # 每个样本的原始宽度
    target_size: int,                 # 目标正方形边长
    K: int,                           # 每个像素保留的事件数
    t0: float,                        # 最小时间间隔（秒）
    tmax: float,                      # 最大时间间隔（秒）
    t_query_list: Optional[List[int]], # 每个样本的查询时间点（可选）
    out_chw: bool,                    # 输出格式
    dtype: Optional[torch.dtype]      # 输出数据类型（可选）
) -> torch.Tensor
```

## 使用示例

### 示例1: 对比两种正方形resize策略

```python
import torch
import tore_cuda

# 原始事件数据：1280×720
events = torch.randn(10000, 4, device='cuda')  # [N,4]
orig_H, orig_W = 720, 1280
target_size = 518

# 方法1：Padding模式（保留全部内容）
tore_padding = tore_cuda.tore_build_single_square(
    events, orig_H, orig_W, target_size,
    K=4, t0=150.0, tmax=5_000_000.0,
    out_chw=True
)
print(f"Padding模式: {tore_padding.shape}")  # [8, 518, 518]
# 特点：完整保留所有事件，但有空白区域（上下各约113像素）

# 方法2：Crop模式（充满画面，v0.0.3新增）⭐
tore_crop = tore_cuda.tore_build_single_square_crop(
    events, orig_H, orig_W, target_size,
    K=4, t0=150.0, tmax=5_000_000.0,
    out_chw=True
)
print(f"Crop模式: {tore_crop.shape}")  # [8, 518, 518]
# 特点：画面充满，无空白，但左右边缘被裁剪（左右各约201像素）
```

### 示例2: ViT目标检测训练（推荐crop模式）

```python
# 原始尺寸 1280×720，目标尺寸 518×518
# 使用crop模式确保画面充满，提高检测效率

import torch
import tore_cuda
from torch.utils.data import DataLoader

orig_H, orig_W = 720, 1280
target_size = 518
batch_size = 64

# 假设已有事件数据列表
events_list = load_event_data()
dataset = EventDataset(events_list)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for batch_events in dataloader:
    batch_size = len(batch_events)
    orig_H_list = [orig_H] * batch_size
    orig_W_list = [orig_W] * batch_size
    
    # 使用crop模式（充满画面，无padding干扰）⭐
    tore_batch = tore_cuda.tore_build_batch_stacked_square_crop(
        batch_events, orig_H_list, orig_W_list, target_size,
        K=8, t0=100.0, tmax=1_000_000.0,
        out_chw=True, dtype=torch.bfloat16
    )
    
    # tore_batch.shape: [64, 16, 518, 518]
    # 画面充满，无空白区域，适合密集预测任务
    predictions = detection_model(tore_batch)
```

### 示例3: 语义分割训练（推荐padding模式）

```python
# 语义分割需要完整的场景信息，使用padding模式

import torch
import tore_cuda

orig_H, orig_W = 720, 1280
target_size = 518

events_list = load_segmentation_data()
orig_H_list = [orig_H] * len(events_list)
orig_W_list = [orig_W] * len(events_list)

# 使用padding模式（保留完整场景）
tore_batch = tore_cuda.tore_build_batch_stacked_square(
    events_list, orig_H_list, orig_W_list, target_size,
    K=8, t0=100.0, tmax=1_000_000.0,
    out_chw=True, dtype=torch.float32
)

# 对padding区域进行mask处理
# 假设padding在上下，每边约113像素
padding_mask = create_padding_mask(tore_batch.shape, top=113, bottom=113)

# 模型预测时可以mask掉padding区域
segmentation_pred = segmentation_model(tore_batch)
segmentation_pred_masked = segmentation_pred * padding_mask
```

### 示例4: 实时推理对比

```python
# 实时处理单个事件流
# 根据任务选择合适的resize模式

def process_for_detection(events, orig_H=720, orig_W=1280):
    """目标检测：使用crop模式"""
    return tore_cuda.tore_build_single_square_crop(
        events, orig_H, orig_W, 518,
        K=4, t0=50.0, tmax=500_000.0,
        out_chw=True, dtype=torch.float16
    )

def process_for_segmentation(events, orig_H=720, orig_W=1280):
    """语义分割：使用padding模式"""
    return tore_cuda.tore_build_single_square(
        events, orig_H, orig_W, 518,
        K=4, t0=50.0, tmax=500_000.0,
        out_chw=True, dtype=torch.float16
    )

# 实时循环
while True:
    events = capture_events()
    
    # 根据任务选择
    if task == 'detection':
        tore = process_for_detection(events)
    else:  # segmentation
        tore = process_for_segmentation(events)
    
    result = model(tore.unsqueeze(0))
```

## 📋 完整选择指南总结

### 根据任务类型选择Resize策略

| 任务类型 | 推荐接口 | Resize策略 | 理由 |
|----------|----------|------------|------|
| **目标检测** | `tore_build_*_square_crop` ⭐ | Crop模式 | 充满画面，提高检测密度 |
| **图像分类** | `tore_build_*_square_crop` ⭐ | Crop模式 | 最大化利用输入区域 |
| **语义分割** | `tore_build_*_square` | Padding模式 | 保留完整场景信息 |
| **关键点检测** | `tore_build_*_square` | Padding模式 | 避免边缘关键点丢失 |
| **ViT backbone** | `tore_build_*_square_crop` ⭐ | Crop模式 | 避免padding干扰attention |
| **多分辨率数据** | `tore_build_*_resized` | 自由比例 | 统一尺寸但保持宽高比 |

### 根据原始宽高比选择

| 原始宽高比 | 推荐策略 | 信息损失 |
|-----------|---------|---------|
| **接近1:1 (如640×480)** | Crop模式 ⭐ | 损失<25% |
| **16:10 (如1280×800)** | Crop或Padding | 损失约20% |
| **16:9 (如1280×720)** | Crop或Padding | 损失约30% |
| **21:9 (如2560×1080)** | Padding模式 | Crop损失>50% |

### API数量总览

v0.0.3共提供**15个API接口**：

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

**正方形Crop模式接口（3个，v0.0.3新增）** ⭐：
- `tore_build_single_square_crop`
- `tore_build_batch_square_crop`
- `tore_build_batch_stacked_square_crop`

## ⚠️ 常见错误和解决方案

### 错误1: 选择了不合适的resize策略

```python
# ❌ 错误：语义分割任务使用crop模式，边缘信息丢失
tore = tore_cuda.tore_build_single_square_crop(events, 720, 1280, 518, ...)

# ✅ 正确：语义分割使用padding模式，保留完整信息
tore = tore_cuda.tore_build_single_square(events, 720, 1280, 518, ...)
```

### 错误2: 极端宽高比使用crop模式

```python
# ❌ 错误：21:9超宽屏使用crop，损失>50%内容
tore = tore_cuda.tore_build_single_square_crop(events, 1080, 2560, 518, ...)

# ✅ 正确：使用按比例resize或padding模式
tore = tore_cuda.tore_build_single_resized(events, 1080, 2560, 518, 217, ...)
```

### 错误3: 内存溢出

```python
# ❌ 错误：batch_size太大
tore = tore_cuda.tore_build_batch_stacked(events_list, H, W, K, ...)

# ✅ 正确：分批次处理
batch_size = 32  # 根据GPU内存调整
for i in range(0, len(events_list), batch_size):
    batch = events_list[i:i+batch_size]
    tore_batch = tore_cuda.tore_build_batch_stacked(batch, H, W, K, ...)
```

### 错误4: 数据类型不匹配

```python
# ❌ 错误：事件数据在CPU上
events = torch.tensor([[x,y,t,p]])  # 默认在CPU
tore = tore_cuda.tore_build_single(events, H, W, K, ...)  # 会报错

# ✅ 正确：先转移到CUDA
events = events.cuda()
tore = tore_cuda.tore_build_single(events, H, W, K, ...)
```

## 版本升级建议

### 从v0.0.2升级到v0.0.3

1. **评估现有使用的`*_square`接口**：
   - 如果用于目标检测/分类，考虑迁移到`*_square_crop`
   - 如果用于语义分割/关键点检测，保持使用`*_square`

2. **新项目建议**：
   - 目标检测：优先使用`*_square_crop` ⭐
   - 语义分割：使用`*_square`
   - 不确定时：两种都试试，看效果

3. **完全向后兼容**：
   - 所有v0.0.1和v0.0.2的接口保持不变
   - 可以无缝升级，逐步尝试新功能

