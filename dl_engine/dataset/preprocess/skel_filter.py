from typing import List, Tuple

import numpy as np

from .base import TRANSFORM, BaseTransform

@TRANSFORM.register_module()
class SkeletonKeypointFilter(BaseTransform):
    def __init__(self,
                 keypoint_involved: List[int],
                 with_pcd_ts: bool = False,
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)
        self.keypoint_involved = keypoint_involved
        self.with_pcd_ts = with_pcd_ts
    
    def get_sub_skeleton(self, skel_frame: np.ndarray):
        selected = []
        for joint in self.keypoint_involved:
            start = joint * 3
            selected.extend(skel_frame[start:start+3])
        if self.with_pcd_ts:
            selected.append(skel_frame[-1])
        
        return np.array(selected)

    def transform(self, input: dict):
        skel_frames: Tuple[np.ndarray] = input['skel_frames']
        filtered_skel_frames = []
        for skel_frame in skel_frames:
            filtered_skel_frames.append(self.get_sub_skeleton(skel_frame))
        input['skel_frames'] = tuple(filtered_skel_frames)
        return input

