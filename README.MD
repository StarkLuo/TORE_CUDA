# TORE-CUDA: Time-Ordered Recent Event Volumes for Event Cameras

CUDA implementation of TORE, based on the paper: "Time-ordered recent event (tore) volumes for event cameras" (Baldwin R W, Liu R, Almatrafi M, et al. TPAMI, 2022) 

## Overview

TORE converts asynchronous event streams into dense, time-ordered volume representations that preserve temporal information.

## Features 

- **Flexible output formats**: Support for CHW and non-CHW tensor layouts  
- **Multiple data types**: Float32, BFloat16, and Float16 support
- **Batch processing**: Efficient batch operations for multiple event sequences

## Installation

```bash
git clone <repository-url>
cd tore_cuda
pip install .
```

Requirements 
- PyTorch with CUDA support 
- CUDA toolkit 

## Usage

### Basic Usage

```python
import torch
import tore_cuda

# Create sample events: [N, 4] tensor with (x, y, t, p)
# x, y: pixel coordinates 
# t: timestamp 
# p: polarity (1 for positive, -1 for negative) 
events = torch.tensor([
    [100, 200, 1000, 1],
    [100, 200, 1500, -1],
    [101, 201, 1200, 1]
], dtype=torch.float32, device='cuda')

# Build TORE volume 
tore_volume = tore_cuda.tore_build_single(
    events=events,
    H=480,           # Image height 
    W=640,           # Image width 
    K=8,             # Number of recent events per pixel | 每个像素的近期事件数
    t0=1.0,          # Minimum time difference (microseconds) | 最小时间差(微秒)
    tmax=1000000.0,  # Maximum time difference (microseconds) | 最大时间差(微秒)
    t_query=None,    # Query time (None = use max timestamp) | 查询时间(None=使用最大时间戳)
    out_chw=True,    # Output format: True=(2K,H,W), False=(2,K,H,W)
    dtype=torch.float32  # Output data type 

print(f"TORE volume shape: {tore_volume.shape}")
# Output: torch.Size([16, 480, 640]) for out_chw=True, K=8
```

### Batch Processing 

```python
# Process multiple event sequences 
events_list = [events1, events2, events3]  # List of event tensors

# Method 1: Returns list of tensors 
tore_volumes = tore_cuda.tore_build_batch(
    events_list=events_list,
    H=480, W=640, K=8,
    t0=1.0, tmax=1000000.0,
    t_query_list=None,
    out_chw=True,
    dtype=torch.float32
)

# Method 2: Returns stacked tensor 
tore_volumes_stacked = tore_cuda.tore_build_batch_stacked(
    events_list=events_list,
    H=480, W=640, K=8,
    t0=1.0, tmax=1000000.0,
    t_query_list=None,
    out_chw=True,
    dtype=torch.float32
)
```

### Advanced Parameters 

- **t_query**: Query time for TORE computation. If None, uses the maximum timestamp in events
- **t_query_list**: List of query times for batch processing (one per sample)
- **out_chw**: Output tensor layout 
  - `True`: `(2K, H, W)` - Channels-first format 
  - `False`: `(2, K, H, W)` - Separate polarity and time dimensions 
- **dtype**: Output data type (`torch.float32`, `torch.bfloat16`, `torch.float16`)


## Tips 

- Use smaller `K` values for faster computation and lower memory usage
- BFloat16/Float16 can provide 2x speedup with minimal precision loss
- Batch processing is more efficient than individual calls

## Example: Event Camera Preprocessing 

```python
def preprocess_events(events, image_size=(480, 640), K=8):
    """Convert raw events to TORE volume for deep learning"""
    H, W = image_size
    
    if not events.is_cuda:
        events = events.cuda()
    
    # Build TORE volume 
    tore = tore_cuda.tore_build_single(
        events=events,
        H=H, W=W, K=K,
        t0=1.0, tmax=1e6,
        out_chw=True,
        dtype=torch.bfloat16
    )
    
    return tore  # Shape: (2*K, H, W) 
```





### Citation Requirements

This is an independent implementation based on the research paper. Please cite both this implementation and the original paper when using this code.

When using this implementation, please cite:

1. This implementation: https://github.com/StarkLuo/TORE_CUDA
2. The original paper: Baldwin R W, Liu R, Almatrafi M, et al. "Time-ordered recent event (tore) volumes for event cameras." TPAMI, 2022.

