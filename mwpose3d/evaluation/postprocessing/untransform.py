from contextlib import contextmanager
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
    Modifies datasample.pred / datasample.gt in place.
    """
    def __init__(self, online_mode: bool = False):
        super().__init__(online_mode)

    @staticmethod
    def _extract_Q_t_row(T):
        """
        For row vectors, we want p' = p @ Q + t.
        Handle either:
        - top-right translation:  t = T[:3, 3]
        - bottom-left translation: t = T[3, :3]
        """
        Q = T[:3, :3]  # this is the linear part used with row vectors in your code
        if np.any(T[3, :3] != 0):    # bottom-left convention
            t = T[3, :3]
        else:                        # top-right convention (your current T_skel)
            t = T[:3, 3]
        return Q, t
    
    @contextmanager
    def _numpy_views(self, datasample, fields=('pred', 'gt')):
        """
        Yield NumPy views of the given fields, then write back in original type.
        Now also supports fields that are a list/tuple of tensors/ndarrays.
        The list dimension is treated as batch (B).
        """
        originals, views = {}, {}

        def _to_numpy(x):
            if x is None:
                return None
            if (torch is not None) and isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            if isinstance(x, np.ndarray):
                return x
            return np.asarray(x)

        for f in fields:
            x = getattr(datasample, f, None)
            originals[f] = x
            if x is None:
                views[f] = None
            elif isinstance(x, (list, tuple)):
                # list-of-tensors/ndarrays -> list-of-numpy
                views[f] = [ _to_numpy(e) for e in x ]
            else:
                views[f] = _to_numpy(x)

        try:
            yield views
        finally:
            for f in fields:
                orig, new = originals[f], views[f]
                if orig is None:
                    continue

                if isinstance(orig, (list, tuple)):
                    # elementwise restore type/device; preserve tuple/list type
                    restored = []
                    for e_orig, e_new in zip(orig, new):
                        if (torch is not None) and isinstance(e_orig, torch.Tensor):
                            restored.append(torch.as_tensor(e_new, dtype=e_orig.dtype, device=e_orig.device))
                        else:
                            restored.append(e_new)
                    if isinstance(orig, tuple):
                        restored = tuple(restored)
                    setattr(datasample, f, restored)
                else:
                    if (torch is not None) and isinstance(orig, torch.Tensor):
                        setattr(datasample, f, torch.as_tensor(new, dtype=orig.dtype, device=orig.device))
                    else:
                        setattr(datasample, f, new)

    @staticmethod
    def _apply_inv_inplace(arr: np.ndarray, Q: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Inverse of p' = p @ Q + t  ->  p = (p' - t) @ Q^{-1}
        Works for 1D 3K[+tail], 2D (N,3K), (N,3), (N,>=3), and 3D (N,K,3).
        """
        if arr is None:
            return arr
        Qinv = np.linalg.inv(Q).astype(np.float32, copy=False)
        t = t.astype(np.float32, copy=False)

        def do_pts(pts):
            return (pts.astype(np.float32, copy=False) - t) @ Qinv

        if arr.ndim == 1:
            K = arr.size // 3
            if K:
                arr[:3*K] = do_pts(arr[:3*K].reshape(-1, 3)).reshape(-1).astype(arr.dtype, copy=False)
            return arr

        if arr.ndim == 2:
            N, C = arr.shape
            if C == 3:
                arr[:, :3] = do_pts(arr[:, :3]).astype(arr.dtype, copy=False)
                return arr
            if C % 3 == 0:
                arr[:, :C] = do_pts(arr[:, :C].reshape(-1, 3)).reshape(N, C).astype(arr.dtype, copy=False)
                return arr
            # first 3 are xyz; keep extras
            arr[:, :3] = do_pts(arr[:, :3]).astype(arr.dtype, copy=False)
            return arr

        if arr.ndim == 3 and arr.shape[-1] == 3:
            N, K, _ = arr.shape
            arr[:] = do_pts(arr.reshape(-1, 3)).reshape(N, K, 3).astype(arr.dtype, copy=False)
            return arr

        return arr  # unknown layout


    def transform(self, data_batch_dict, datasample):
        import numpy as np

        def extract_Q_t_row(M: np.ndarray):
            # row-vector convention: [x y z 1] @ M
            Q = M[:3, :3]
            t = M[:3, 3]
            return Q, t

        def canonize_T_skel(T_skel):
            """
            Returns per_batch_T: List[List[np.ndarray]]
            - len(per_batch_T) == B
            - len(per_batch_T[b]) == T
            - per_batch_T[b][t] is 4x4 np.ndarray
            Accepts:
            - List[Tuple[np.ndarray]]  -> [T][B]
            - Tuple[np.ndarray] (B==1) -> [T]
            """
            # Case: Tuple[np.ndarray] -> batch=1
            if isinstance(T_skel, tuple) and all(isinstance(x, np.ndarray) for x in T_skel):
                return [list(T_skel)]  # B=1

            # Case: List[Tuple[np.ndarray]] -> frame-major
            if isinstance(T_skel, list) and len(T_skel) > 0 and isinstance(T_skel[0], tuple):
                T = len(T_skel)
                B = len(T_skel[0])
                per_batch = [[] for _ in range(B)]
                for t in range(T):
                    assert len(T_skel[t]) == B, "Inconsistent batch size across frames in T_skel"
                    for b in range(B):
                        per_batch[b].append(np.asarray(T_skel[t][b], dtype=np.float32))
                return per_batch

            # Fallback: already a single 4x4 (treat as B=1, T=1)
            if isinstance(T_skel, np.ndarray) and T_skel.shape == (4, 4):
                return [[T_skel]]

            # If still nested, peel once and retry
            if isinstance(T_skel, (list, tuple)) and len(T_skel) > 0:
                return canonize_T_skel(T_skel[0])

            raise TypeError(f"Unexpected T_skel structure: {type(T_skel)}")

        def apply_inv_per_frame(arr, per_batch_T):
            """
            Apply inverse of per-frame row-vector transforms to various layouts.

            Supports:
            - list/tuple of (T, C) arrays/tensors  -> batch list with sequence first
            - (B, T, C) ndarray
            - (T, C) ndarray (assumes B==1)
            - (B, C) ndarray with T==1
            - (C,) flat vector

            Returns with the SAME container/shape as input.
            """
            if arr is None:
                return None

            # build per-batch Qinv/t once
            def _extract_Q_t_row(M: np.ndarray):
                Q = M[:3, :3]
                t = M[:3, 3]
                return Q, t

            B = len(per_batch_T)
            T = len(per_batch_T[0])
            Qinv = [[np.linalg.inv(_extract_Q_t_row(M)[0]).astype(np.float32) for M in per_batch_T[b]] for b in range(B)]
            tvec = [[_extract_Q_t_row(M)[1].astype(np.float32) for M in per_batch_T[b]] for b in range(B)]

            def _untransform_pts(pts, Qinv_, t_):
                # pts: (N, 3) float32-ish
                return (pts.astype(np.float32, copy=False) - t_) @ Qinv_

            # --- NEW: list/tuple (batch list) ---
            if isinstance(arr, (list, tuple)):
                assert len(arr) == B, f"B mismatch: data {len(arr)} vs T_skel {B}"
                out_list = []
                for b in range(B):
                    xb = np.asarray(arr[b])
                    assert xb.ndim == 2 and xb.shape[0] == T, \
                        f"Expect (T, C) per batch item; got {xb.shape} for batch {b}"
                    C = xb.shape[1]
                    assert C % 3 == 0, "Channel dim must be multiple of 3"
                    K = C // 3
                    xb_out = xb.copy()
                    for i in range(T):
                        pts = xb_out[i, :3*K].reshape(-1, 3)
                        pts = _untransform_pts(pts, Qinv[b][i], tvec[b][i])
                        xb_out[i, :3*K] = pts.reshape(-1)
                    out_list.append(xb_out)
                # preserve original container type
                return tuple(out_list) if isinstance(arr, tuple) else out_list

            # --- Existing cases ---
            arr_np = np.asarray(arr)

            if arr_np.ndim == 3:
                assert arr_np.shape[0] == B, f"B mismatch: data {arr_np.shape[0]} vs T_skel {B}"
                assert arr_np.shape[1] == T, f"T mismatch: data {arr_np.shape[1]} vs T_skel {T}"
                C = arr_np.shape[2]; assert C % 3 == 0
                K = C // 3
                out = arr_np.copy()
                for b in range(B):
                    for i in range(T):
                        pts = out[b, i, :3*K].reshape(-1, 3)
                        pts = _untransform_pts(pts, Qinv[b][i], tvec[b][i])
                        out[b, i, :3*K] = pts.reshape(-1)
                return out

            if arr_np.ndim == 2 and arr_np.shape[0] == T:
                C = arr_np.shape[1]; assert C % 3 == 0
                K = C // 3
                out = arr_np.copy()
                for i in range(T):
                    pts = out[i, :3*K].reshape(-1, 3)
                    pts = _untransform_pts(pts, Qinv[0][i], tvec[0][i])
                    out[i, :3*K] = pts.reshape(-1)
                return out

            if arr_np.ndim == 2 and arr_np.shape[0] == B and T == 1:
                C = arr_np.shape[1]; assert C % 3 == 0
                K = C // 3
                out = arr_np.copy()
                for b in range(B):
                    pts = out[b, :3*K].reshape(-1, 3)
                    pts = _untransform_pts(pts, Qinv[b][0], tvec[b][0])
                    out[b, :3*K] = pts.reshape(-1)
                return out

            if arr_np.ndim == 1 and (arr_np.size % 3 == 0):
                K = arr_np.size // 3
                out = arr_np.copy()
                pts = out[:3*K].reshape(-1, 3)
                pts = _untransform_pts(pts, Qinv[0][-1], tvec[0][-1])
                out[:3*K] = pts.reshape(-1)
                return out

            print(f"[SkeletonBackToOriginalCoord] Unhandled array shape {arr_np.shape}; leaving unchanged.")
            return arr

        T_skel = data_batch_dict.get('T_skel', None)
        if T_skel is None:
            print("Warning: T_skel is None, skipping untransform.")
            return data_batch_dict, datasample

        per_batch_T = canonize_T_skel(T_skel)

        with self._numpy_views(datasample, fields=('pred', 'gt')) as a:
            a['pred'] = apply_inv_per_frame(a.get('pred'), per_batch_T)
            a['gt']   = apply_inv_per_frame(a.get('gt'), per_batch_T)

        return data_batch_dict, datasample