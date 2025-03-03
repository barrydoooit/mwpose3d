from typing import Tuple
import numpy as np

from .base import TRANSFORM, BaseTransform



@TRANSFORM.register_module()
class AddRangeDimension(BaseTransform):
    def __init__(self,
                 insert_idx: int = 4
                 ):
        self.insert_idx = insert_idx
    
    @staticmethod
    def _calc_range(pcd_frame: np.ndarray):
        range = np.linalg.norm(pcd_frame[:, :3], axis=1)
        return range
    
    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        added_frames = []
        for pcd_frame in pcd_frames:
            range = self._calc_range(pcd_frame)
            pcd_frame = np.insert(pcd_frame, self.insert_idx, range, axis=1)
            added_frames.append(pcd_frame)
        input['pcd_frames'] = tuple(added_frames)
        return input
