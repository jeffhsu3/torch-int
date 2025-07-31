import os
import subprocess
from setuptools import setup, find_packages
import torch
from torch.utils import cpp_extension

# Get the compute capability of the GPU
compute_capability = torch.cuda.get_device_capability()
cuda_arch = compute_capability[0] * 100 + compute_capability[1] * 10

class BuildExtensionWithCutlass(cpp_extension.BuildExtension):
    def run(self):
        # Build the CUTLASS submodule
        subprocess.run(['bash', 'build_cutlass.sh'], check=True)
        super().run()

setup(
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='torch_int._CUDA',
            sources=[
                'torch_int/kernels/linear.cu',
                'torch_int/kernels/bmm.cu',
                'torch_int/kernels/fused.cu',
                'torch_int/kernels/bindings.cpp',
            ],
            include_dirs=['torch_int/kernels/include', 'submodules/cutlass/include'],
            extra_link_args=['-lcublas_static', '-lcublasLt_static',
                             '-lculibos', '-lcudart', '-lcudart_static',
                             '-lrt', '-lpthread', '-ldl', '-L/usr/lib/x86_64-linux-gnu/', '-Lsubmodules/cutlass/build/lib'],
            extra_compile_args={'cxx': ['-std=c++14', '-O3'],
                                'nvcc': ['-O3', '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', f'-DCUDA_ARCH={cuda_arch}']},
        ),
    ],
    cmdclass={
        'build_ext': BuildExtensionWithCutlass.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests']),
)
