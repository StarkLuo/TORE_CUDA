import torch
import tore_cuda

def test_resize_functionality():
    """测试resize功能"""
    device = 'cuda'
    
    # 创建测试事件数据 (x,y,t,p)
    # 原始尺寸为720x1280
    events = torch.tensor([
        [100, 200, 100, 1],
        [200, 400, 200, -1],
        [300, 600, 150, 1],
        [400, 800, 250, -1],
        [500, 1000, 300, 1]
    ], dtype=torch.float32, device=device)
    
    print("原始事件数据:")
    print(f"事件数量: {events.shape[0]}")
    print(f"事件坐标范围: x=[{events[:,0].min():.0f}, {events[:,0].max():.0f}], y=[{events[:,1].min():.0f}, {events[:,1].max():.0f}]")
    
    # 测试1: 按比例resize到360x640
    print("\n=== 测试1: 按比例resize到360x640 ===")
    target_H, target_W = 360, 640
    out1 = tore_cuda.tore_build_single_resized(
        events, 720, 1280, target_H, target_W, 4, 150.0, 5_000_000.0,
        None, True, torch.bfloat16
    )
    print(f"输出形状: {out1.shape}")
    print(f"输出dtype: {out1.dtype}")
    print(f"期望形状: torch.Size([8, {target_H}, {target_W}])")
    
    # 测试2: resize到正方形518x518
    print("\n=== 测试2: resize到正方形518x518 ===")
    target_size = 518
    out2 = tore_cuda.tore_build_single_square(
        events, 720, 1280, target_size, 4, 150.0, 5_000_000.0,
        None, True, torch.bfloat16
    )
    print(f"输出形状: {out2.shape}")
    print(f"输出dtype: {out2.dtype}")
    print(f"期望形状: torch.Size([8, {target_size}, {target_size}])")
    
    # 测试3: 批量处理带resize
    print("\n=== 测试3: 批量处理带resize ===")
    events_list = [events, events]  # 两个相同的事件列表
    orig_H_list = [720, 720]  # 原始高度列表
    orig_W_list = [1280, 1280]  # 原始宽度列表
    
    out3 = tore_cuda.tore_build_batch_stacked_resized(
        events_list, orig_H_list, orig_W_list, 360, 640, 4, 150.0, 5_000_000.0,
        None, True, torch.bfloat16
    )
    print(f"批量输出形状: {out3.shape}")
    print(f"批量输出dtype: {out3.dtype}")
    print(f"期望形状: torch.Size([2, 8, 360, 640])")
    
    # 测试4: 批量正方形resize
    print("\n=== 测试4: 批量正方形resize ===")
    out4 = tore_cuda.tore_build_batch_stacked_square(
        events_list, orig_H_list, orig_W_list, 518, 4, 150.0, 5_000_000.0,
        None, True, torch.bfloat16
    )
    print(f"批量正方形输出形状: {out4.shape}")
    print(f"批量正方形输出dtype: {out4.dtype}")
    print(f"期望形状: torch.Size([2, 8, 518, 518])")
    
    # 检查版本号
    print(f"\n=== 版本信息 ===")
    print(f"当前版本: {tore_cuda.tore_version()}")
    print(f"期望版本: 0.0.2")
    
    print("\n✅ 所有测试完成!")

if __name__ == "__main__":
    test_resize_functionality()