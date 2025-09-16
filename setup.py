# setup.py (at repo root, parallel to ./mwpose3d)
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths
import os

# Toggle debug vs release with BUILD_DEBUG=1
DEBUG = os.environ.get("BUILD_DEBUG", "0") == "1"
cxx_flags = ["-O0", "-g", "-std=c++17"] if DEBUG else ["-O3", "-std=c++17"]
nvcc_flags = ["-O0", "-G", "-lineinfo"] if DEBUG else ["-O3", "-use_fast_math"]

# Optional: pin GPU archs (or set TORCH_CUDA_ARCH_LIST in env)
# nvcc_flags += ["-gencode=arch=compute_80,code=sm_80"]

ext_modules = [
    # Existing smoother extension
    CUDAExtension(
        name="mwpose3d.evaluation.postprocessing.smoother_experimental.cuda_pose_smoother",
        sources=[
            "mwpose3d/evaluation/postprocessing/smoother_experimental/csrc/bindings.cpp",
            "mwpose3d/evaluation/postprocessing/smoother_experimental/csrc/pose_smoother.cu",
            "mwpose3d/evaluation/postprocessing/smoother_experimental/csrc/gaussian_ema.cu",
        ],
        include_dirs=[
            "mwpose3d/evaluation/postprocessing/smoother_experimental/csrc"
        ] + include_paths(),  # torch + pybind11
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    ),

    # New SDTW extension
    CUDAExtension(
        name="mwpose3d.models.utils.sdtw_cuda.sdtw_cuda",
        sources=[
            "mwpose3d/models/utils/sdtw_cuda/sdtw_cuda.cpp",
            "mwpose3d/models/utils/sdtw_cuda/sdtw_cuda_kernel.cu",
        ],
        include_dirs=include_paths(),  # no local headers, just torch/pybind11
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    ),
]

setup(
    name="mwpose3d",
    version="0.0.0",
    packages=find_packages(include=["mwpose3d", "mwpose3d.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True, no_python_abi_suffix=True),},
    include_package_data=True,
    zip_safe=False,
)
