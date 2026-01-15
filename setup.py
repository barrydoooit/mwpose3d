# setup.py (at repo root)
from setuptools import setup, find_packages
import os
import sys

# Check if CUDA is available
def cuda_available():
    """Check if CUDA is available on this system."""
    # No CUDA on macOS
    if sys.platform == "darwin":
        return False
    # Check for CUDA_HOME environment variable
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.exists(cuda_home):
        return True
    # Check common CUDA paths
    for path in ["/usr/local/cuda", "/opt/cuda"]:
        if os.path.exists(path):
            return True
    return False

ext_modules = []
cmdclass = {}

if cuda_available():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths
    
    # Toggle debug vs release with BUILD_DEBUG=1
    DEBUG = os.environ.get("BUILD_DEBUG", "0") == "1"
    cxx_flags = ["-O0", "-g", "-std=c++17"] if DEBUG else ["-O3", "-std=c++17"]
    nvcc_flags = ["-O0", "-G", "-lineinfo"] if DEBUG else ["-O3", "-use_fast_math"]

    ext_modules = [
        # Smoother extension
        CUDAExtension(
            name="mwpose3d.evaluation.postprocessing.smoother_experimental.cuda_pose_smoother",
            sources=[
                "mwpose3d/evaluation/postprocessing/smoother_experimental/csrc/bindings.cpp",
                "mwpose3d/evaluation/postprocessing/smoother_experimental/csrc/pose_smoother.cu",
                "mwpose3d/evaluation/postprocessing/smoother_experimental/csrc/gaussian_ema.cu",
            ],
            include_dirs=[
                "mwpose3d/evaluation/postprocessing/smoother_experimental/csrc"
            ] + include_paths(),
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        ),
        # SDTW extension
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
    cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=True, no_python_abi_suffix=True)}
else:
    print("⚠️  CUDA not available - skipping CUDA extensions (smoother, SDTW)")
    print("   The package will work but without GPU-accelerated smoothing.")

setup(
    name="mwpose3d",
    version="0.0.0",
    packages=find_packages(include=["mwpose3d", "mwpose3d.*", "apps", "apps.*", "projects", "projects.*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
    zip_safe=False,
)
