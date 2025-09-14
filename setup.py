from setuptools import setup

# 尝试直接导入 torch 的扩展工具；若失败（例如纯元数据阶段），降级为无扩展配置
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    import torch, os
    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
    ext_modules = [
        CUDAExtension(
            name='tore_cuda',
            sources=['tore_cuda.cpp', 'tore_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            },
            extra_link_args=[f'-Wl,-rpath,{torch_lib_dir}']
        )
    ]
    cmdclass = {'build_ext': BuildExtension}
except Exception as e:
    print('[setup] WARN: torch not importable during setup; building without extensions.\n', e)
    ext_modules = []
    cmdclass = {}

setup(
    name='tore_cuda',
    version='0.0.2',
    description='TORE CUDA kernels with resize support',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
