from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths
import os

PKG = "mwpose3d.evaluation.postprocessing.smoother_experimental"
CSRC = os.path.join("mwpose3d", "evaluation", "postprocessing", "smoother_experimental", "csrc")
# os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"
ext = CUDAExtension(
    name=f"{PKG}.cuda_pose_smoother",
    sources=[
        os.path.join(CSRC, "bindings.cpp"),
        os.path.join(CSRC, "pose_smoother.cu"),
        os.path.join(CSRC, "gaussian_ema.cu"), 
    ],
    include_dirs=[CSRC] + include_paths(),  # adds torch + pybind11 include paths
    extra_compile_args={
        # "cxx": ["-O3", "-std=c++17"],
        # "nvcc": ["-O3", "-use_fast_math"],
        "cxx": ["-O0", "-g", "-std=c++17"],
        "nvcc": ["-O0", "-G", "-lineinfo"]  # remove "-use_fast_math" while debugging
    },
)

setup(
    name="postprocessing-smoother-experimental",
    version="0.1.0",
    packages=find_packages(include=["mwpose3d.evaluation.postprocessing.smoother_experimental"]),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
