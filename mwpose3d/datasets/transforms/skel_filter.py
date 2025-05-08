from typing import List, Tuple

import numpy as np

from .base import BaseTransform
from mwpose3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class SkeletonKeypointFilter(BaseTransform):
    def __init__(self,
                 keypoint_involved: List[int],
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)
        self.keypoint_involved = keypoint_involved
    
    def get_sub_skeleton(self, skel_frame: np.ndarray):
        selected = []
        for joint in self.keypoint_involved:
            start = joint * 3
            selected.extend(skel_frame[start:start+3].copy())
        
        return np.array(selected)

    def transform(self, input: dict):
        skel_frames: Tuple[np.ndarray] = input['skel_frames']
        filtered_skel_frames = []
        for skel_frame in skel_frames:
            filtered_skel_frames.append(self.get_sub_skeleton(skel_frame))
        input['skel_frames'] = tuple(filtered_skel_frames)
        return input

@TRANSFORMS.register_module()
class SkeletonCoordNormalization(BaseTransform):
    def __init__(self,
                 means: List[float],
                 stds: List[float],
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)
        self.means = np.array(means)
        self.stds = np.array(stds)

    def transform(self, input: dict):
        skel_frames: Tuple[np.ndarray] = input['skel_frames']
        nomalized_skel_frames = []
        for skel_frame in skel_frames:
            skel_array = (skel_frame - self.means) / self.stds
            nomalized_skel_frames.append(skel_array)
        input['skel_frames'] = tuple(nomalized_skel_frames)
        return input