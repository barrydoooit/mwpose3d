from typing import List, Tuple, Union

import numpy as np

from .base import BaseTransform
from mwpose3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class SkeletonKeypointFilter(BaseTransform):
    def __init__(self,
                 keypoints_involved: List[int],
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)
        self.keypoints_involved = keypoints_involved
    
    def get_sub_skeleton(self, skel_frame: np.ndarray):
        selected = []
        for joint in self.keypoints_involved:
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

@TRANSFORMS.register_module()
class ToRelativeSkeleton(BaseTransform):
    def __init__(self,
                 keypoints_involved: List[int],
                 anchor_joint: Union[int, Tuple[int, int]],
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)
        self.keypoints_involved = keypoints_involved
        self.anchor_joint = anchor_joint

        # Determine anchor type and prepare offset function
        if isinstance(anchor_joint, int):
            # Single-joint anchor
            anchor_idx = self.keypoints_involved.index(anchor_joint)

            def _get_offset(skel_arr: np.ndarray) -> np.ndarray:
                return skel_arr[anchor_idx]

            self._compute_offset = _get_offset
        else:
            # Pair-of-joints anchor: use center point
            idx1 = self.keypoints_involved.index(anchor_joint[0])
            idx2 = self.keypoints_involved.index(anchor_joint[1])

            def _get_offset(skel_arr: np.ndarray) -> np.ndarray:
                return (skel_arr[idx1] + skel_arr[idx2]) / 2.0

            self._compute_offset = _get_offset

    def transform(self, input: dict):
        skel_frames: Tuple[np.ndarray] = input['skel_frames']
        relative_skel_frames = []

        for skel_frame in skel_frames:
            # reshape to (n_keypoints, 3)
            skel_array = skel_frame.reshape(-1, 3)

            # compute anchor offset and subtract
            offset = self._compute_offset(skel_array)
            skel_array = skel_array - offset

            # flatten back
            skel_array = skel_array.flatten()
            relative_skel_frames.append(skel_array)

        input['skel_frames'] = tuple(relative_skel_frames)
        return input

@TRANSFORMS.register_module()
class SkeletonDatasetTransform(BaseTransform):
    """
        The goal of this Reorder is to receive an input of dataset A and transform the joints and order as if it were dataset B.
        This is specifically useful when training on A and testing on B.
    """
    DONT_CHANGE_AXIS = ()
    VALID_DATASETS = ["mars", "mmfi"]
    MAPS = {
        "mars": {
            "mars": DONT_CHANGE_AXIS,
            "mmfi": (0, 14, 15, 17, 10, 11, 13, 1, 18, 2, 3, 4, 5, 6, 7, 8, 9), # Drops axis (12,16) (ankles)
        },
        "mmfi": {
            "mars": (0, 7, 9, 10, 11, 12, 13, 14, 15, 16, 4, 5, 6, 1, 2, 3, 8), # Don't drop anything, just reorder
            "mmfi": DONT_CHANGE_AXIS,
        },
    }

    def __init__(self, frm: str, to: str, online_mode: bool = False):
        super().__init__(online_mode)
        self.dataset_from = frm.lower()
        self.dataset_to = to.lower()

        assert self.dataset_from in self.VALID_DATASETS, f"{self.dataset_from} not in {self.VALID_DATASETS}"
        assert self.dataset_to in self.VALID_DATASETS, f"{self.dataset_to} not in {self.VALID_DATASETS}"

    def transform(self, input):
        mapping = self.MAPS[self.dataset_from][self.dataset_to]

        data = input['skel_frames']

        # Make sure to crash if this happens,
        # However, I think we always get just a single frame, never an entire batch
        assert len(data) == 1, f"This is likely more than one batch, which is not supported: {data}"

        data = np.array(data).reshape((-1, 3))
        data = self.reorder_axis(data, mapping)
        input['skel_frames'] = tuple([data.flatten()])
        return input

    @staticmethod
    def reorder_axis(data: np.ndarray, axis: tuple):
        if len(axis) > 0:
            return data[axis, :]
        else:
            return data