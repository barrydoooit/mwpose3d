from contextlib import contextmanager
from collections.abc import Sequence
import time
from typing import Union
import numpy as np
import torch

from mwpose3d.datasets.transforms.base import OnlineEnabled
from .base import BasePostProcessing
from .base import POSTPROCESSING


@OnlineEnabled
@POSTPROCESSING.register_module()
class SkeletonBackToOriginalCoord(BasePostProcessing):
    """
    Undo the accumulated row-vector transform T_skel:
        [x y z 1] @ inv(T_skel)

    Supported datasample structures (return structure matches input):
      1) Single datasample whose `pred`/`gt` are 1D (3K) or 2D (T, C) tensors/arrays.
      2) List of datasamples (batch) where each has 1D or 2D `pred`/`gt`.

    `data_batch_dict['T_skel']` is canonized into a List[List[np.ndarray]]
    with shape [B][T] (B=batch, T=frames), accepting interchangeable container
    types (list/tuple combinations, etc.) and single 4x4 matrices.
    """

    def __init__(self, online_mode: bool = False, process_with_torch: bool = True):
        super().__init__(online_mode)
        self.process_with_torch = process_with_torch

    # ----------------------------- helpers ---------------------------------

    @staticmethod
    def _extract_Q_t_row(T: Union[np.ndarray, torch.Tensor]):
        """
        Row-vector affine blocks:
        T = [[Q, 0],
            [t^T, 1]]

        Returns:
        Q: (3,3)
        t: (3,)
        """
        Q = T[:3, :3]
        t = T[3, :3]
        return Q, t

    @staticmethod
    def _is_matrix44(x) -> bool:
        return isinstance(x, np.ndarray) and x.shape == (4, 4)

    @staticmethod
    def _is_container(x) -> bool:
        # Treat list/tuple (and other Sequence types except str/bytes) as containers.
        return isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))

    @staticmethod
    def _canonize_T_skel(T_skel):
        """
        Returns per_batch_T: List[List[np.ndarray]] with shape [B][T],
        where per_batch_T[b][t] is a 4x4 np.ndarray (float32).

        Accepts source formats with interchangeable containers:
          - Sequence of matrices:            [T]                  -> B=1
          - Sequence of (sequence of mats):  [T][B] (frame-major) -> convert to [B][T]
          - Single 4x4 matrix:               -> [[M]]
          - Nested containers: peel until one of the above matches

        Note: This function intentionally does not require specific container
        types (list vs tuple). Any Sequence works.
        """
        if T_skel is None:
            return None

        # Single 4x4 matrix
        if SkeletonBackToOriginalCoord._is_matrix44(T_skel):
            return [[np.asarray(T_skel, dtype=np.float32)]]

        # Container cases
        if SkeletonBackToOriginalCoord._is_container(T_skel):
            if len(T_skel) == 0:
                raise ValueError("T_skel is an empty container.")

            first = T_skel[0]

            # Case: [T] where each element is a 4x4 matrix
            if all(SkeletonBackToOriginalCoord._is_matrix44(x) for x in T_skel):
                return [[np.asarray(x, dtype=np.float32) for x in T_skel]]

            # Case: [T][B] where inner elements are 4x4 matrices (frame-major)
            if (SkeletonBackToOriginalCoord._is_container(first)
                and len(first) > 0
                and all(SkeletonBackToOriginalCoord._is_matrix44(y) for y in first)):
                T = len(T_skel)
                B = len(first)
                # Validate consistent B across frames
                for t in range(T):
                    assert len(T_skel[t]) == B, "Inconsistent batch size across frames in T_skel"
                per_batch = [[] for _ in range(B)]
                for t in range(T):
                    for b in range(B):
                        per_batch[b].append(np.asarray(T_skel[t][b], dtype=np.float32))
                return per_batch

            # Otherwise, peel one level and retry (handles deeper nesting or mixed containers)
            return SkeletonBackToOriginalCoord._canonize_T_skel(first)

        raise TypeError(f"Unexpected T_skel structure of type {type(T_skel)}")

    @staticmethod
    def _build_Qinv_t_list(per_frame_T):
        """
        From a list of 4x4 row-vector transforms for a single sample,
        build lists of (Q^{-1}, t) as float32.
        """
        Qinv_list, t_list = [], []
        for M in per_frame_T:
            Q, t = SkeletonBackToOriginalCoord._extract_Q_t_row(np.asarray(M))
            Qinv = np.linalg.inv(Q).astype(np.float32, copy=False)
            t = t.astype(np.float32, copy=False)
            Qinv_list.append(Qinv)
            t_list.append(t)
        return Qinv_list, t_list

    @staticmethod
    def _apply_inv_single_tensor(arr: torch.Tensor, per_frame_T):  # NEW
        if arr is None:
            return None
        assert isinstance(arr, torch.Tensor)

        if arr.numel() == 0:
            return arr

        device = arr.device
        orig_dtype = arr.dtype
        work_dtype = torch.float32  # compute in fp32, cast back

        # Precompute Q^{-1} and t on-device
        Qinv_list, t_list = [], []
        if arr.dim() == 1:
            per_frame_T = [per_frame_T[-1]]
        
        for M in per_frame_T:
            Q_t, t_t = SkeletonBackToOriginalCoord._extract_Q_t_row(torch.as_tensor(M, dtype=work_dtype, device=device))
            Qinv = torch.linalg.inv(Q_t)
            Qinv_list.append(Qinv)
            t_list.append(t_t)

        def untransform(pts, Qinv_, t_):
            return (pts.to(work_dtype) - t_) @ Qinv_
        # 1D (3K) -> use last frame
        if arr.dim() == 1:
            K = arr.numel() // 3
            if K == 0:
                return arr
            L = K * 3
            out = arr.clone()
            Qinv_, t_ = Qinv_list[-1], t_list[-1]
            pts = out[:L].view(-1, 3)
            pts = untransform(pts, Qinv_, t_)
            out[:L] = pts.reshape(-1).to(orig_dtype)
            return out

        # 2D (T, C) -> per-frame
        if arr.dim() == 2:
            T = arr.shape[0]
            assert T == len(per_frame_T), \
                f"Sequence length mismatch: data T={T} vs T_skel T={len(per_frame_T)}"
            C = arr.shape[1]
            L = (C // 3) * 3
            if L == 0:
                return arr
            out = arr.clone()
            for i in range(T):
                Qinv_, t_ = Qinv_list[i], t_list[i]
                pts = out[i, :L].view(-1, 3)
                pts = untransform(pts, Qinv_, t_)
                out[i, :L] = pts.reshape(-1).to(orig_dtype)
            return out

        # 3D (T, K, 3) -> per-frame
        if arr.dim() == 3 and arr.shape[-1] == 3:
            T = arr.shape[0]
            assert T == len(per_frame_T), \
                f"Sequence length mismatch: data T={T} vs T_skel T={len(per_frame_T)}"
            out = arr.clone()
            for i in range(T):
                Qinv_, t_ = Qinv_list[i], t_list[i]
                pts = out[i].reshape(-1, 3)
                pts = untransform(pts, Qinv_, t_)
                out[i] = pts.view_as(out[i]).to(orig_dtype)
            return out

        print(f"[SkeletonBackToOriginalCoord] (torch) Unhandled tensor shape {arr.shape}; leaving unchanged.")
        return arr

    @staticmethod
    def _apply_inv_single_array(arr: np.ndarray, per_frame_T):
        """
        Apply the inverse transform to a *single* sample's array using the
        given per-frame transforms (list of 4x4). Handles:
          - 1D (3K)            -> uses last frame's inverse
          - 2D (T, C)          -> per-frame; C may be 3K or >=3 (extras preserved)
          - 3D (T, K, 3)       -> per-frame
        """
        if arr is None:
            return None

        arr = np.asarray(arr)
        if arr.size == 0:
            return arr

        Qinv_list, t_list = SkeletonBackToOriginalCoord._build_Qinv_t_list(per_frame_T)

        def untransform(pts, Qinv_, t_):
            # pts: (N,3), float32 pipeline
            return (pts.astype(np.float32, copy=False) - t_) @ Qinv_

        # 1D: use last frame transform
        if arr.ndim == 1:
            K = arr.size // 3
            if K == 0:
                return arr
            L = K * 3
            out = arr.copy()
            Qinv_, t_ = Qinv_list[-1], t_list[-1]
            pts = out[:L].reshape(-1, 3)
            pts = untransform(pts, Qinv_, t_)
            out[:L] = pts.reshape(-1).astype(arr.dtype, copy=False)
            return out

        # 2D: per-frame along dim-0
        if arr.ndim == 2:
            T = arr.shape[0]
            assert T == len(per_frame_T), \
                f"Sequence length mismatch: data T={T} vs T_skel T={len(per_frame_T)}"
            C = arr.shape[1]
            # Use the largest multiple of 3 within C; extras (if any) are preserved
            L = (C // 3) * 3
            if L == 0:
                return arr
            out = arr.copy()
            for i in range(T):
                Qinv_, t_ = Qinv_list[i], t_list[i]
                pts = out[i, :L].reshape(-1, 3)
                pts = untransform(pts, Qinv_, t_)
                out[i, :L] = pts.reshape(-1).astype(arr.dtype, copy=False)
            return out

        # 3D: treat dim-0 as frames, last dim as xyz
        if arr.ndim == 3 and arr.shape[-1] == 3:
            T = arr.shape[0]
            assert T == len(per_frame_T), \
                f"Sequence length mismatch: data T={T} vs T_skel T={len(per_frame_T)}"
            out = arr.copy()
            for i in range(T):
                Qinv_, t_ = Qinv_list[i], t_list[i]
                pts = out[i].reshape(-1, 3)
                pts = untransform(pts, Qinv_, t_)
                out[i] = pts.reshape(out[i].shape).astype(arr.dtype, copy=False)
            return out

        # Unknown layout: leave unchanged
        print(f"[SkeletonBackToOriginalCoord] Unhandled array shape {arr.shape}; leaving unchanged.")
        return arr

    @contextmanager
    def _numpy_views(self, datasample, fields=('pred', 'gt')):
        """Yield NumPy views of the given fields on a SINGLE datasample, then write back in original type."""
        originals, views = {}, {}
        for f in fields:
            x = getattr(datasample, f, None)
            originals[f] = x
            if x is None:
                views[f] = None
            elif (torch is not None) and isinstance(x, torch.Tensor):
                views[f] = x.detach().cpu().numpy()
            elif isinstance(x, np.ndarray):
                views[f] = x
            else:
                views[f] = np.asarray(x)
        try:
            yield views
        finally:
            for f in fields:
                orig, new = originals[f], views[f]
                if orig is None:
                    continue
                if (torch is not None) and isinstance(orig, torch.Tensor):
                    setattr(datasample, f, torch.as_tensor(new, dtype=orig.dtype, device=orig.device))
                else:
                    # Keep numpy for non-tensors; preserve dtype if possible
                    if isinstance(orig, np.ndarray) and isinstance(new, np.ndarray) and new.dtype != orig.dtype:
                        new = new.astype(orig.dtype, copy=False)
                    setattr(datasample, f, new)

    def transform(self, data_batch_dict, datasample):
        T_skel = data_batch_dict.get('T_skel', None)
        if T_skel is None:
            print("Warning: T_skel is None, skipping untransform.")
            return data_batch_dict, datasample

        per_batch_T = self._canonize_T_skel(T_skel)  # [B][T]

        # Case A: datasample is a list (batch)
        if isinstance(datasample, list):
            B_data = len(datasample)
            B_T = len(per_batch_T)
            assert B_data == B_T, \
                f"Batch size mismatch: datasample B={B_data} vs T_skel B={B_T}"
            for b, ds in enumerate(datasample):
                pred, gt = getattr(ds, 'pred', None), getattr(ds, 'gt', None)

                if self.process_with_torch:
                    ds.pred = self._apply_inv_single_tensor(pred, per_batch_T[b])
                    ds.gt   = self._apply_inv_single_tensor(gt,   per_batch_T[b])
                else:
                    # Original NumPy path
                    with self._numpy_views(ds, fields=('pred', 'gt')) as arrs:
                        arrs['pred'] = self._apply_inv_single_array(arrs.get('pred'), per_batch_T[b])
                        arrs['gt']   = self._apply_inv_single_array(arrs.get('gt'),   per_batch_T[b])
            return data_batch_dict, datasample

        # Case B: single datasample object (B==1)
        else:
            assert len(per_batch_T) >= 1, "Empty T_skel after canonization."
            per_frame_T = per_batch_T[0]
            pred, gt = getattr(datasample, 'pred', None), getattr(datasample, 'gt', None)

            if self.process_with_torch:
                datasample.pred = self._apply_inv_single_tensor(pred, per_frame_T)
                datasample.gt   = self._apply_inv_single_tensor(gt,   per_frame_T)
                return data_batch_dict, datasample
            else:
                with self._numpy_views(datasample, fields=('pred', 'gt')) as arrs:
                    arrs['pred'] = self._apply_inv_single_array(arrs.get('pred'), per_frame_T)
                    arrs['gt']   = self._apply_inv_single_array(arrs.get('gt'),   per_frame_T)
                return data_batch_dict, datasample