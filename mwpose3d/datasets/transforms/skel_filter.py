from typing import List, Tuple, Union
from operator import itemgetter

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
    We do the transform by taking the intersection of all joints across all datasets. The intersection is the set of joints that are compatible with all datasets.
    """

    JOINTS: dict[str, dict[str, int]] = {
        "mars": {
            "spine_base": 0,
            "spine_mid": 1,
            "neck": 2,
            "head": 3,
            "shoulder_left": 4,
            "elbow_left": 5,
            "wrist_left": 6,
            "hand_left": 7,
            "shoulder_right": 8,
            "elbow_right": 9,
            "wrist_right": 10,
            "hand_right": 11,
            "hip_left": 12,
            "knee_left": 13,
            "ankle_left": 14,
            "foot_left": 15,
            "hip_right": 16,
            "knee_right": 17,
            "ankle_right": 18,
            "foot_right": 19,
            "spine_shoulder": 20,
            "hand_tip_left": 21,
            "hand_thumb_left": 22,
            "hand_tip_right": 23,
            "hand_thumb_right": 24,
        },
        "mmfi": {
            "spine_base": 0,
            "hip_right": 1,
            "knee_right": 2,
            "ankle_right": 3,
            "hip_left": 4,
            "knee_left": 5,
            "ankle_left": 6,
            "spine_mid": 7,
            "spine_shoulder": 8,
            "neck": 9,
            "head": 10,
            "shoulder_left": 11,
            "elbow_left": 12,
            "wrist_left": 13,
            "shoulder_right": 14,
            "elbow_right": 15,
            "wrist_right": 16,
        },
        "mri": {
            "head": 0,
            "eye_left": 1,
            "eye_right": 2,
            "ear_left": 3,
            "ear_right": 4,
            "shoulder_right": 5,
            "shoulder_left": 6,
            "elbow_right": 7,
            "elbow_left": 8,
            "wrist_right": 9,
            "wrist_left": 10,
            "hip_right": 11,
            "hip_left": 12,
            "knee_right": 13,
            "knee_left": 14,
            "ankle_right": 15,
            "ankle_left": 16,
        },
        "milipoint": {
            "head": 0,  # Technically Nose
            "spine_shoulder": 1,
            "shoulder_right": 2,
            "elbow_right": 3,
            "wrist_right": 4,
            "shoulder_left": 5,
            "elbow_left": 6,
            "wrist_left": 7,
            "hip_right": 8,
            "knee_right": 9,
            "ankle_right": 10,
            "hip_left": 11,
            "knee_left": 12,
            "ankle_left": 13,
            "eye_right": 14,
            "eye_left": 15,
            "ear_right": 16,
            "ear_left": 17,
        },
    }
    VALID_DATASETS: set[str] = set(JOINTS.keys())

    def __init__(self, frm: str, to: str | list[str], online_mode: bool = False):
        super().__init__(online_mode)

        # Parse & check input dataset
        self.dataset_from = frm.lower()
        assert (
                self.dataset_from in self.VALID_DATASETS
        ), f"{self.dataset_from} not in {self.VALID_DATASETS}"

        # Parse & check output dataset
        # This can be a single dataset, or multiple
        if isinstance(to, str):
            if to.lower() == "all":
                self.datasets = self.VALID_DATASETS
            else:
                self.datasets = set(to)
        else:
            self.datasets: set[str] = {str(item).lower() for item in to}
        assert self.datasets.issubset(
            self.VALID_DATASETS
        ), f"Unknown dataset(s): {self.datasets - self.VALID_DATASETS}"

        # Create the intersection of all compatible joints between the datasets
        joints_per_dataset = [
            set(self.JOINTS[dataset].keys()) for dataset in self.datasets
        ]
        self.joint_intersection: list[str] = sorted(
            set.intersection(*joints_per_dataset)
        )

        # Funky method to get multiple keys from a dict at once
        self.mapping: tuple[int] = itemgetter(*self.joint_intersection)(
            self.JOINTS[self.dataset_from]
        )
        print(
            f"Generated mapping for {self.dataset_from} -> {self.datasets} = "
            f"{self.mapping} ({len(self.mapping)} joints)"
        )

    def transform(self, input: dict[str, any]) -> dict[str, any]:
        input_data = input["skel_frames"]

        output_data = []
        for frame_data in input_data:
            # Need to massage the data a bit on order to reorder nicely
            frame_data = np.array(frame_data).reshape((-1, 3))
            frame_data = self.reorder_axis(frame_data, self.mapping)
            output_data.append(frame_data.flatten())

        input["skel_frames"] = tuple(output_data)
        return input

    @staticmethod
    def reorder_axis(data: np.ndarray, axis: tuple) -> np.ndarray:
        return data[axis, :]