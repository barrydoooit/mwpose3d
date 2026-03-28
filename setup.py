from setuptools import setup, find_packages
import os

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths
    _torch_cuda = torch.version.cuda is not None
except (ImportError, OSError):
    # torch not available or DLL load failed (e.g. WinError 1114 on Windows).
    # Skip CUDA extension — the pure-Python package will still install correctly.
    _torch_cuda = False

DEBUG = os.environ.get("BUILD_DEBUG", "0") == "1"
cxx_flags = ["-O0", "-g", "-std=c++17"] if DEBUG else ["-O3", "-std=c++17"]
nvcc_flags = ["-O0", "-G", "-lineinfo"] if DEBUG else ["-O3", "-use_fast_math"]

ext_modules = []
cmdclass = {}

# CPU-only wheels typically have torch.version.cuda == None
if _torch_cuda:
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
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="mwpose3d",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
