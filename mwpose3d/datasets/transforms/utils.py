import numpy as np
from typing import Iterable, Optional, Tuple, Union, Sequence

def make_row_affine(R: Optional[np.ndarray]=None, t: Optional[Iterable]=None, dtype=np.float32) -> np.ndarray:
    """
    Build a 4x4 homogeneous matrix M for row vectors so that:
        [x y z 1] @ M
    If R is a standard 3x3 rotation (column-vector convention), we store R^T
    so that your row-vector usage 'pts.dot(R.T) + t' matches '[x 1] @ M'.
    """
    M = np.eye(4, dtype=dtype)
    if R is not None:
        M[:3, :3] = R.T  # row-vector block
    if t is not None:
        M[:3, 3] = np.asarray(t, dtype=dtype)
    return M

def _ensure_T_tuple(input_dict: dict, key: str, n: int) -> None:
    """Initialize input_dict[key] to a tuple of I_4 if missing or wrong length."""
    I = np.eye(4, dtype=np.float32)
    if key not in input_dict or input_dict[key] is None:
        input_dict[key] = tuple(I.copy() for _ in range(n))
    else:
        T = input_dict[key]
        if not isinstance(T, (tuple, list)) or len(T) != n:
            input_dict[key] = tuple(I.copy() for _ in range(n))

def _broadcast_As(A_or_As: Union[np.ndarray, Sequence[np.ndarray]], n: int) -> Tuple[np.ndarray, ...]:
    """
    If given a single 4x4 A, broadcast it to (n,) tuple.
    If given a sequence, assert length n and return as tuple.
    """
    if isinstance(A_or_As, np.ndarray):
        return tuple(A_or_As.copy() for _ in range(n))
    # sequence case
    As = tuple(A_or_As)
    if len(As) != n:
        raise ValueError(f"Expected {n} per-frame transforms, got {len(As)}.")
    return As

def compose_into(input_dict: dict, key: str, A_or_As: Union[np.ndarray, Sequence[np.ndarray]], n: int) -> None:
    """
    Post-multiply the per-frame accumulator(s) for 'key' by A_or_As.
    - key is 'T_pcd' or 'T_skel'
    - n is the number of frames for that modality
    - A_or_As is either a single 4x4 (broadcast to all frames) or a sequence of 4x4 (one per frame)
    """
    _ensure_T_tuple(input_dict, key, n)
    As = _broadcast_As(A_or_As, n)
    T_old = input_dict[key]
    input_dict[key] = tuple(T_old[i] @ As[i] for i in range(n))
