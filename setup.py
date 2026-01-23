from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths
import os

DEBUG = os.environ.get("BUILD_DEBUG", "0") == "1"
cxx_flags = ["-O0", "-g", "-std=c++17"] if DEBUG else ["-O3", "-std=c++17"]
nvcc_flags = ["-O0", "-G", "-lineinfo"] if DEBUG else ["-O3", "-use_fast_math"]

ext_modules = [
    CUDAExtension(
        name="mwpose3d.models.utils.sdtw_cuda.sdtw_cuda",
        sources=[
            "mwpose3d/models/utils/sdtw_cuda/sdtw_cuda.cpp",
            "mwpose3d/models/utils/sdtw_cuda/sdtw_cuda_kernel.cu",
        ],
        include_dirs=include_paths(),
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    ),
]

setup(
    name="mwpose3d",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)