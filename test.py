import torch, tore_cuda
device = 'cuda'
# 伪造一点点事件 (x,y,t,p)
ev = torch.tensor([[10,20,100, 1],
                   [10,20,200,-1],
                   [11,21,150, 1]], dtype=torch.float32, device=device)
out = tore_cuda.tore_build_single(ev, 720, 1280, 4, 150.0, 5_000_000.0,
                                  None, True, torch.bfloat16)
print(out.shape, out.dtype)  # torch.Size([8, 720, 1280]) torch.bfloat16
