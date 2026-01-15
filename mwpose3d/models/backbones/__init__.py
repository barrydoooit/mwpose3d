# SECOND backbone uses mmcv operations
# Make import optional for environments without full mmcv
try:
    from .second import SECOND
    __all__ = ["SECOND"]
except ImportError:
    SECOND = None
    __all__ = []
