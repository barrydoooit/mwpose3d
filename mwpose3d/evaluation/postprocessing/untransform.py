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
    def _apply_inv_inplace(arr: np.ndarray, T_inv: np.ndarray) -> np.ndarray:
        """
        Apply [x y z 1] @ T_inv to the first 3 coords.
        Supports flat (3K[+tail]) or (K, >=3). Returns the same array (modified).
        """
        if arr is None:
            return arr
        A = T_inv.astype(arr.dtype, copy=False)

        if arr.ndim == 1:
            K = arr.size // 3
            if K == 0:
                return arr
            pts = arr[:3*K].reshape(-1, 3)
            ones = np.ones((K, 1), dtype=arr.dtype)
            arr[:3*K] = (np.hstack([pts, ones]) @ A)[:, :3].reshape(-1)
            return arr

        if arr.ndim == 2 and arr.shape[1] >= 3:
            K = arr.shape[0]
            ones = np.ones((K, 1), dtype=arr.dtype)
            arr[:, :3] = (np.hstack([arr[:, :3], ones]) @ A)[:, :3]
            return arr

        return arr

    def transform(self, data_batch_dict, datasample):
        T_skel = data_batch_dict.get('T_skel', None)
        if T_skel is None:
            return data_batch_dict, datasample
        T_skel = T_skel[0]
        T_inv = invert_row_affine(np.asarray(T_skel, dtype=np.float32))

        with self._numpy_views(datasample, fields=('pred', 'gt')) as a:
            if a.get('pred') is not None:
                a['pred'] = self._apply_inv_inplace(a['pred'], T_inv)
            if a.get('gt') is not None:
                a['gt'] = self._apply_inv_inplace(a['gt'], T_inv)

        return data_batch_dict, datasample