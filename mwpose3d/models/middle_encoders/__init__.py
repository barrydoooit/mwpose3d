# Sparse encoder requires spconv (CUDA only)
# Make import optional for Mac/CPU-only environments
try:
    from .sparse_encoder import SparseEncoder
    __all__ = ['SparseEncoder']
except ImportError:
    SparseEncoder = None
    __all__ = []
