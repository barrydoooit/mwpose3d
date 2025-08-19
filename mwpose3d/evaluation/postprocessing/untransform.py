from contextlib import contextmanager
import numpy as np
import torch

from mwpose3d.datasets.transforms.base import OnlineEnabled
from mwpose3d.datasets.transforms.utils import invert_row_affine
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
        """Yield NumPy views of the given fields, then write back in original type."""
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
        T_skel = data_batch_dict.get('T_skel', None)
        if T_skel is None:
            print("Warning: T_skel is None, skipping untransform.")
            return data_batch_dict, datasample
        while not isinstance(T_skel, np.ndarray):
            T_skel = T_skel[0]
        # T_inv = invert_row_affine(np.asarray(T_skel, dtype=np.float32))
        Q, t = self._extract_Q_t_row(T_skel)

        with self._numpy_views(datasample, fields=('pred', 'gt')) as a:
            if a.get('pred') is not None:
                a['pred'] = self._apply_inv_inplace(a['pred'], Q, t)
            if a.get('gt') is not None:
                a['gt'] = self._apply_inv_inplace(a['gt'], Q, t)
        return data_batch_dict, datasample