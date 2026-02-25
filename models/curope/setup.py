# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

"""
Build script for the `curope` C++/CUDA (or ROCm/HIP) extension.

CUDA:
  python setup.py build_ext --inplace

ROCm:
  python setup.py build_ext --inplace

Notes for ROCm:
- PyTorch will hipify `.cu` sources automatically when `torch.version.hip` is set.
- Do NOT pass CUDA-only flags (e.g. `--ptxas-options` or `-gencode`) under ROCm.
- To control ROCm GPU targets, set `PYTORCH_ROCM_ARCH` (recommended).
"""

from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_cuda_or_hip_build_flags():
    """
    Returns (nvcc_or_hipcc_flags, define_macros, undef_macros).
    PyTorch uses the `nvcc` key in extra_compile_args for both CUDA and ROCm
    (it will route to nvcc or hipcc depending on the backend).
    """
    if torch.version.hip:
        # ROCm / HIP build
        # Keep flags minimal and portable. Architecture selection should be done via:
        #   export PYTORCH_ROCM_ARCH="gfx90a;gfx1100;..."
        nvcc_flags = ["-O3", "-ffast-math"]
        define_macros = [("USE_ROCM", None)]
        undef_macros = ["__HIP_NO_HALF_CONVERSIONS__"]
        return nvcc_flags, define_macros, undef_macros

    # CUDA build
    from torch import cuda

    try:
        # compile for all possible CUDA architectures visible to the local toolchain
        cuda_arch_flags = cuda.get_gencode_flags().replace("compute=", "arch=").split()
    except Exception:
        cuda_arch_flags = []

    nvcc_flags = ["-O3", "--ptxas-options=-v", "--use_fast_math"] + cuda_arch_flags
    define_macros = []
    undef_macros = []
    return nvcc_flags, define_macros, undef_macros


nvcc_flags, define_macros, undef_macros = get_cuda_or_hip_build_flags()

ext_modules = [
    CUDAExtension(
        name="curope",
        sources=[
            "curope.cpp",
            "kernels.cu",
        ],
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": nvcc_flags,
        },
    )
]

setup(
    name="curope",
    ext_modules=ext_modules,
    cmdclass={
        # Disabling ninja tends to be more robust across varied build environments/containers.
        "build_ext": BuildExtension.with_options(use_ninja=False)
    },
)
