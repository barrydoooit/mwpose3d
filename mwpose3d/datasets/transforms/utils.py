import logging
import warnings
import numpy as np
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Union, Sequence

from mwpose3d.datasets.transforms.base import KEYS_OF_SYNCABLE_SEQUENCES
from mwpose3d.datasets.transforms.loading import LoadMultiFrameFromH5
from mwpose3d.datasets.utils import pseudo_collate, pseudo_decollate, pseudo_recollate_into

def make_row_affine(R: Optional[np.ndarray]=None, t: Optional[Iterable]=None, dtype=np.float32) -> np.ndarray:
    """
    Creates a 4x4 affine transformation matrix for row-vector multiplication (p' = p @ T).
    """
    M = np.eye(4, dtype=dtype)
    if R is not None:
        # The rotation matrix R is applied as p' = p @ R.T, so we place R.T here.
        M[:3, :3] = R.T
    if t is not None:
        # The translation vector t is placed in the last row for row-vector multiplication.
        M[3, :3] = np.asarray(t, dtype=dtype)
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

def apply_frame_selection(
        input_dict: dict,
        keep_indices: Optional[Iterable[int]] = None,
        target_seq_len: Optional[int] = None,
        *,
        skip_keys: set | None = None) -> dict:
    if skip_keys is None: skip_keys = set()
    if target_seq_len is None:
        target_seq_len = len(input_dict[LoadMultiFrameFromH5.PCD_FRAMES])
    if keep_indices is not None:
        keep = list(map(int, keep_indices))
        prev_remaining = input_dict.get('remaining_frames_idx', list(range(target_seq_len)))
        input_dict['remaining_frames_idx'] = [prev_remaining[i] for i in keep]
    else:
        keep = input_dict.get('remaining_frames_idx', list(range(target_seq_len)))
    for key in KEYS_OF_SYNCABLE_SEQUENCES:
        if key in skip_keys: continue
        if key in input_dict:
            val = input_dict[key]
            L = len(val)
            if isinstance(val, (list, tuple)):
                if L >= target_seq_len and all(i < L for i in keep):
                    slices = [val[i] for i in keep]
                    input_dict[key] = tuple(slices) if isinstance(val, tuple) else slices

def _apply_transforms(sample: Any, transforms: List[Callable[[Any], Any]]) -> Any:
    for t in transforms:
        sample = t(sample)
    return sample

def apply_per_sample_transforms_serial(big_batch: dict, transforms: List[Callable[[Any], Any]], inplace=True) -> Any:
    samples = pseudo_decollate(big_batch)
    samples = [_apply_transforms(s, transforms) for s in samples]
    if not inplace:
        return pseudo_collate(samples)
    return pseudo_recollate_into(samples, big_batch)

from concurrent.futures import ProcessPoolExecutor
import math, os

_GLOBAL_TRANSFORMS: List = []

def _init_worker(transforms, suppress_import_warnings: bool = True):
    """Initializer runs once per worker process."""
    global _GLOBAL_TRANSFORMS
    _GLOBAL_TRANSFORMS = transforms
    if suppress_import_warnings:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    for name in ("OpenGL", "OpenGL.acceleratesupport"):
       logging.getLogger(name).setLevel(logging.ERROR)

def _transform_worker(sample: Any) -> Any:
    for t in _GLOBAL_TRANSFORMS:
        sample = t(sample)
    return sample

def _share_cpu_tensors_inplace(samples):
    try:
        import torch
    except Exception:
        return
    def _share(x):
        if isinstance(x, torch.Tensor) and x.device.type == "cpu":
            x.share_memory_()
        elif isinstance(x, Mapping):
            for v in x.values(): _share(v)
        elif isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            for v in x: _share(v)
    for s in samples: _share(s)

def _default_chunksize(n: int, workers: int) -> int:
    return max(1, math.ceil(n / (8 * max(1, workers))))

def _apply_parallel_with_executor(
    big_batch: dict,
    *,
    executor,
    inplace: bool,
    chunksize: Optional[int] = None,
    share_cpu_tensors: bool = False,
) -> Any:
    samples = pseudo_decollate(big_batch)
    # Only meaningful for process-based pools:
    if share_cpu_tensors and isinstance(executor, ProcessPoolExecutor):
        _share_cpu_tensors_inplace(samples)

    pool_size = getattr(executor, "_max_workers", os.cpu_count() or 1)
    if chunksize is None:
        chunksize = _default_chunksize(len(samples), pool_size)

    results = list(executor.map(_transform_worker, samples, chunksize=chunksize))
    return pseudo_recollate_into(results, big_batch) if inplace else pseudo_collate(results)