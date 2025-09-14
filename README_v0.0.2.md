# TORE CUDA v0.0.2

TORE (Time-Ordered REpresentation) CUDA实现，支持事件相机的时空表示构建，新增事件resize功能以兼容固定patch size的模型（如ViT）。

## 版本信息
- 当前版本: 0.0.2
- 新增功能: 事件resize支持（按比例resize和正方形resize）
- 新增功能: 自动类型转换支持（int32/float32等自动转换为float32）
- 向后兼容: 完全兼容v0.0.1版本的所有接口

## 安装

```bash
pip install .
```

## 核心概念

TORE将事件流转换为时空体积表示，每个像素位置保留最近的K个事件时间戳，用于后续的深度学习处理。

## API选择指南

### 🔍 如何选择合适的API接口？

根据您的使用场景，选择最适合的API接口：

| 场景 | 推荐API | 说明 |
|------|---------|------|
| **单张图像处理** | `tore_build_single` | 处理单个事件样本，返回单个张量 |
| **批量处理但需要单独访问每个结果** | `tore_build_batch` | 批量处理，返回张量列表，可单独访问每个结果 |
| **批量处理且需要堆叠结果** | `tore_build_batch_stacked` | 批量处理，直接返回堆叠好的batch张量，效率最高 |
| **需要resize到固定尺寸** | `tore_build_*_resized` | 按比例resize事件坐标 |
| **需要正方形输出（如ViT）** | `tore_build_*_square` | 输出正方形TORE表示 |

### 📊 接口选择决策树

```
需要resize吗？
├─ 是 → 需要正方形输出吗？
│   ├─ 是 → 使用 *_square 系列接口
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

### 4. 带resize的单样本处理（v0.0.2新增）

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

### 5. 带正方形resize的单样本处理（v0.0.2新增）

**使用场景**：
- ✅ **ViT模型适配**：Vision Transformer需要正方形输入（224×224, 518×518等）
- ✅ **ImageNet预训练模型**：大多数预训练模型使用正方形输入
- ✅ **固定patch size模型**：如16×16, 14×14等patch划分
- ✅ **正方形卷积核优化**：正方形特征图有利于GPU内存对齐
- ✅ **多尺度训练**：在正方形输入上应用多尺度数据增强

**不适用场景**：
- ❌ **矩形特征保留**：会改变原始宽高比
- ❌ **精确几何计算**：crop/padding会改变事件的空间分布
- ❌ **低宽度事件流**：以宽度为基准可能导致高度信息丢失

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

### 6. 带resize的批量处理（v0.0.2新增）

**使用场景**：
- ✅ **多分辨率训练集**：不同事件相机统一尺寸处理
- ✅ **数据预处理管道**：resize和TORE构建一体化
- ✅ **内存优化训练**：降低高分辨率事件的显存占用
- ✅ **模型输入标准化**：统一不同数据源的输入尺寸

**不适用场景**：
- ❌ **需要保持样本差异**：所有样本resize到相同尺寸
- ❌ **在线学习**：需要动态调整尺寸的情况


```python
tore_cuda.tore_build_batch_resized(
    events_list: List[torch.Tensor],  # 事件列表，每个元素[N_i,4]
    orig_H_list: List[int],           # 每个样本的原始高度
    orig_W_list: List[int],           # 每个样本的原始宽度
    target_H: int,                    # 目标输出高度
    target_W: int,                    # 目标输出宽度
    K: int,                           # 每个像素保留的事件数
    t0: float,                        # 最小时间间隔（秒）
    tmax: float,                      # 最大时间间隔（秒）
    t_query_list: Optional[List[int]], # 每个样本的查询时间点（可选）
    out_chw: bool,                    # 输出格式
    dtype: Optional[torch.dtype]      # 输出数据类型（可选）
) -> List[torch.Tensor]
```

### 7. 带正方形resize的批量处理（v0.0.2新增）

```python
tore_cuda.tore_build_batch_square(
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
) -> List[torch.Tensor]
```

### 8. 带resize的批量堆叠处理（v0.0.2新增）

```python
tore_cuda.tore_build_batch_stacked_resized(
    events_list: List[torch.Tensor],  # 事件列表，每个元素[N_i,4]
    orig_H_list: List[int],           # 每个样本的原始高度
    orig_W_list: List[int],           # 每个样本的原始宽度
    target_H: int,                    # 目标输出高度
    target_W: int,                    # 目标输出宽度
    K: int,                           # 每个像素保留的事件数
    t0: float,                        # 最小时间间隔（秒）
    tmax: float,                      # 最大时间间隔（秒）
    t_query_list: Optional[List[int]], # 每个样本的查询时间点（可选）
    out_chw: bool,                    # 输出格式
    dtype: Optional[torch.dtype]      # 输出数据类型（可选）
) -> torch.Tensor
```

### 9. 带正方形resize的批量堆叠处理（v0.0.2新增）

```python
tore_cuda.tore_build_batch_stacked_square(
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

### 基础使用（保持原始尺寸）
```python
import torch
import tore_cuda

# 事件数据 [N,4] -> (x,y,t,p)
events = torch.tensor([[100,200,100,1], [200,400,200,-1]], 
                     dtype=torch.float32, device='cuda')

# 构建TORE表示
tore = tore_cuda.tore_build_single(events, 720, 1280, 4, 150.0, 5_000_000.0)
print(tore.shape)  # torch.Size([8, 720, 1280])
```

### ViT兼容使用（resize到518×518）
```python
# 原始尺寸 1280×720，目标尺寸 518×518
orig_H, orig_W = 720, 1280
target_size = 518

# 批量处理
events_list = [events1, events2, ...]  # 多个事件样本
orig_H_list = [720] * len(events_list)
orig_W_list = [1280] * len(events_list)

tore_batch = tore_cuda.tore_build_batch_stacked_square(
    events_list, orig_H_list, orig_W_list, target_size,
    K=4, t0=150.0, tmax=5_000_000.0,
    t_query_list=None, out_chw=True, dtype=torch.bfloat16
)
print(tore_batch.shape)  # torch.Size([B, 8, 518, 518])
```

## 📋 完整选择指南总结

### 根据任务类型选择

| 任务类型 | 推荐接口 | 理由 |
|----------|----------|------|
| **研究实验** | `tore_build_single` | 快速验证，灵活调试 |
| **生产推理** | `tore_build_batch_stacked` | 最高效率，标准输出 |
| **ViT训练** | `tore_build_batch_stacked_square` | 正方形输出，适配预训练模型 |
| **多分辨率数据** | `tore_build_batch_stacked_resized` | 统一尺寸，内存优化 |
| **数据预处理** | `tore_build_*_resized` 系列 | 一体化处理，减少I/O |

### 根据数据规模选择

| 数据规模 | 推荐接口 | 配置建议 |
|----------|----------|----------|
| **< 100样本** | `tore_build_single` | 逐个处理，简单直接 |
| **100-1000样本** | `tore_build_batch_stacked` | 中等batch size，平衡效率 |
| **> 1000样本** | `tore_build_batch_stacked` + 分批次 | 避免内存溢出 |
| **流数据** | `tore_build_single` | 实时处理，无需缓存 |

### 根据硬件条件选择

| 硬件条件 | 推荐接口 | 优化策略 |
|----------|----------|----------|
| **高端GPU** | `tore_build_batch_stacked` | 大batch size，充分利用算力 |
| **中端GPU** | `tore_build_batch_stacked` | 中等batch size，如32-64 |
| **低端GPU** | `tore_build_single` 或 小batch | 避免内存溢出 |
| **CPU推理** | `tore_build_single` | 事件数据需先转到CUDA |

## 🎯 实际应用示例

### 示例1: ViT事件分类模型训练
```python
# 事件相机原始分辨率: 1280×720
# ViT模型要求输入: 518×518
# 训练数据: 10,000个事件样本

import torch
import tore_cuda

# 假设已有事件数据列表
events_list = load_event_data()  # 10,000个事件样本
orig_H, orig_W = 720, 1280
target_size = 518  # ViT输入尺寸
batch_size = 64    # 根据GPU内存调整

# 创建DataLoader
dataset = EventDataset(events_list)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for batch_events in dataloader:
    batch_size = len(batch_events)
    
    # 准备原始尺寸列表
    orig_H_list = [orig_H] * batch_size
    orig_W_list = [orig_W] * batch_size
    
    # 使用正方形resize接口
    tore_batch = tore_cuda.tore_build_batch_stacked_square(
        batch_events, orig_H_list, orig_W_list, target_size,
        K=8, t0=100.0, tmax=1_000_000.0,
        out_chw=True, dtype=torch.bfloat16
    )
    
    # tore_batch.shape: [64, 16, 518, 518]
    # 直接输入到ViT模型
    predictions = vit_model(tore_batch)
```

### 示例2: 实时事件流推理
```python
# 实时处理单个事件流
# 原始分辨率: 640×480
# 目标分辨率: 320×240 (降低计算量)

def process_event_stream(events, orig_H=480, orig_W=640):
    """处理单个事件流样本"""
    tore = tore_cuda.tore_build_single_resized(
        events, orig_H, orig_W, 240, 320,  # resize到320×240
        K=4, t0=50.0, tmax=500_000.0,
        out_chw=True, dtype=torch.float16
    )
    return tore

# 实时循环
while True:
    events = capture_events()  # 捕获事件数据
    tore = process_event_stream(events)
    result = model(tore.unsqueeze(0))  # 添加batch维度
```

### 示例3: 多分辨率数据集处理
```python
# 处理来自不同事件相机的数据
# 相机1: 1280×720, 相机2: 640×480, 相机3: 320×240
# 统一resize到: 256×256

events_cam1 = load_camera1_data()  # 1280×720
events_cam2 = load_camera2_data()  # 640×480
events_cam3 = load_camera3_data()  # 320×240

# 分别处理不同分辨率
tore1 = tore_cuda.tore_build_batch_stacked_resized(
    events_cam1, [720]*len(events_cam1), [1280]*len(events_cam1), 256, 256,
    K=6, t0=100.0, tmax=1_000_000.0, out_chw=True
)

tore2 = tore_cuda.tore_build_batch_stacked_resized(
    events_cam2, [480]*len(events_cam2), [640]*len(events_cam2), 256, 256,
    K=6, t0=100.0, tmax=1_000_000.0, out_chw=True
)

tore3 = tore_cuda.tore_build_batch_stacked_resized(
    events_cam3, [240]*len(events_cam3), [320]*len(events_cam3), 256, 256,
    K=6, t0=100.0, tmax=1_000_000.0, out_chw=True
)

# 现在所有数据都是256×256，可以统一处理
all_tore = torch.cat([tore1, tore2, tore3], dim=0)
```

## ⚠️ 常见错误和解决方案

### 错误1: 内存溢出
```python
# ❌ 错误：batch_size太大
tore = tore_cuda.tore_build_batch_stacked(events_list, H, W, K, ...)

# ✅ 正确：分批次处理
batch_size = 32  # 根据GPU内存调整
for i in range(0, len(events_list), batch_size):
    batch = events_list[i:i+batch_size]
    tore_batch = tore_cuda.tore_build_batch_stacked(batch, H, W, K, ...)
```

### 错误2: 数据类型不匹配
```python
# ❌ 错误：事件数据在CPU上
events = torch.tensor([[x,y,t,p]])  # 默认在CPU
tore = tore_cuda.tore_build_single(events, H, W, K, ...)  # 会报错

# ✅ 正确：先转移到CUDA
events = events.cuda()
tore = tore_cuda.tore_build_single(events, H, W, K, ...)
```

### 错误3: 向上resize
```python
# ❌ 错误：试图向上resize
tore = tore_cuda.tore_build_single_resized(events, 240, 320, 480, 640, ...)

# ✅ 正确：只支持向下resize
tore = tore_cuda.tore_build_single_resized(events, 480, 640, 240, 320, ...)
```

## 版本信息

```python
print(tore_cuda.tore_version())  # "0.0.2"
print(tore_cuda.__version__)     # "0.0.2"
```

## 注意事项

1. **事件数据格式**: 必须是CUDA tensor，形状[N,4]，列顺序为(x,y,t,p)
2. **坐标范围**: x∈[0,W), y∈[0,H), t≥0, p∈{-1,+1}
3. **resize限制**: 只允许向下resize（目标尺寸 ≤ 原始尺寸）
4. **内存要求**: 大批量处理时需要足够的GPU内存
5. **数据类型**: 支持float32、bfloat16、float16

## 性能优化建议

1. **批处理**: 尽量使用批量接口而非循环调用单样本
2. **数据类型**: 根据模型需求选择合适的数据类型（bfloat16/float16可节省内存）
3. **事件过滤**: 在输入前过滤掉明显异常的事件数据
4. **内存管理**: 大batch size时考虑分批次处理

## 更新日志

### v0.0.2 (当前版本)
- 新增事件resize功能（按比例和正方形）
- 保持完全向后兼容
- 优化内存使用

### v0.0.1
- 初始版本，基础TORE功能