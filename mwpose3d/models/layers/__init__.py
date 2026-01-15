# Sparse blocks require mmdet + spconv (CUDA only)
# Make import optional for Mac/CPU-only environments
try:
    from .sparse_block import SparseBasicBlock, SparseBottleneck, make_sparse_convmodule
except ImportError:
    SparseBasicBlock = None
    SparseBottleneck = None
    make_sparse_convmodule = None
