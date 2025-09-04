from typing import Optional, Tuple, Union

import numpy as np

from ..utils import compose_into, make_row_affine
from ..base import BaseTransform, OnlineEnabled
from mwpose3d.registry import TRANSFORMS




@OnlineEnabled
@TRANSFORMS.register_module()
class RelativeCoordtoTrackingCentroid(BaseTransform):
    def __init__(self,
                 discretize_resolution: Optional[Union[int, Tuple[float, float, float]]] = None,
                 online_mode: bool = False):
        super().__init__(online_mode)
        if discretize_resolution is None:
            self.discretize_resolution = None
        elif isinstance(discretize_resolution, (int, float)):
            self.discretize_resolution = (float(discretize_resolution),) * 3
        else:
            assert len(discretize_resolution) == 3
            self.discretize_resolution = tuple(float(x) for x in discretize_resolution)

    def _discretize(self, v: np.ndarray) -> np.ndarray:
        if self.discretize_resolution is None:
            return v
        out = v.astype(np.float32).copy()
        for axis, res in enumerate(self.discretize_resolution):
            out[axis] = np.round(out[axis] / res) * res
        return out

    def _apply_t_to_skel_frame(self, f: np.ndarray, t: np.ndarray) -> np.ndarray:
        n3 = (len(f) // 3) * 3
        head = f[:n3].reshape(-1, 3) + t
        if n3 < len(f):
            return np.concatenate([head.ravel(), f[n3:]])
        return head.ravel()

    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        n_pcd = len(pcd_frames)
        skel_frames: Optional[Tuple[np.ndarray]] = input.get('skel_frames', None)
        n_skel = n_pcd if skel_frames is None else len(skel_frames)

        cent = input['track_centroid']
        per_frame = isinstance(cent, (tuple, list))
        # Build per-frame translations (3,) for PCD and (optionally) SKEL
        if per_frame:
            if len(cent) != n_pcd:
                raise ValueError(f"track_centroid tuple length {len(cent)} must match num PCD frames {n_pcd}.")
            t_list = []
            for i in range(n_pcd):
                c = self._discretize(np.asarray(cent[i], dtype=np.float32))
                t_list.append((-c[:3]).astype(np.float32))
            t_list = tuple(t_list)
        else:
            c = self._discretize(np.asarray(cent, dtype=np.float32))
            t_shared = (-c[:3]).astype(np.float32)
            t_list  = [t_shared for _ in range(n_pcd)]

        new_pcd = []
        for i, f in enumerate(pcd_frames):
            t = t_list[i]
            pts = f[:, :3] + t
            new_pcd.append(np.hstack([pts, f[:, 3:]]) if f.shape[1] > 3 else pts)
        input['pcd_frames'] = tuple(new_pcd)
        
        if skel_frames is not None:
            input['skel_frames'] = tuple(
                self._apply_t_to_skel_frame(skel_frames[i], t_list[i])
                for i in range(n_skel)
            )

        # Accumulate per-frame As
        A_pcd = tuple(make_row_affine(R=None, t=t) for t in t_list)
        compose_into(input, 'T_pcd', A_pcd, n=n_pcd)
        A_skel = tuple(make_row_affine(R=None, t=t) for t in t_list)
        compose_into(input, 'T_skel', A_skel, n=n_skel)

        return input