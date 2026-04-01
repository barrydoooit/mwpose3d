from typing import List, Literal, Tuple

import h5py
import numpy as np

from mwpose3d.datasets.transforms.utils import apply_frame_selection

from .base import BaseTransform, OnlineEnabled
from mwpose3d.registry import TRANSFORMS



@OnlineEnabled
@TRANSFORMS.register_module()
class PointCloudRangeFilter(BaseTransform):
    def __init__(self,
                 point_cloud_range: List[float],
                 empty_frame_op: Literal['duplicate', 'shift', 'error'] = 'duplicate',
                 backup_frames: int = 0,
                 min_num_frames: int = 1,
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)
        self.empty_frame_op = empty_frame_op
        self.min_num_frames = min_num_frames
        if self.online_mode:
            self.empty_frame_op = 'error'
        self.backup_frames = backup_frames
        self.point_cloud_range = point_cloud_range[:6]
    
    def get_filtered_frames(self, pcd_frames: Tuple[np.ndarray]):
        filtered_frames = []
        for pcd_frame in pcd_frames:
            mask = (pcd_frame[:, 0] > self.point_cloud_range[0]) & \
                   (pcd_frame[:, 0] < self.point_cloud_range[3]) & \
                   (pcd_frame[:, 1] > self.point_cloud_range[1]) & \
                   (pcd_frame[:, 1] < self.point_cloud_range[4]) & \
                   (pcd_frame[:, 2] > self.point_cloud_range[2]) & \
                   (pcd_frame[:, 2] < self.point_cloud_range[5])
            filtered_frames.append(pcd_frame[mask])
        return filtered_frames
    
    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        filtered_frames = self.get_filtered_frames(pcd_frames)
        empty_frame_indices = [i for i, frame in enumerate(filtered_frames) if frame.shape[0] == 0]
        if len(empty_frame_indices) > 0:
           filtered_frames = self.operate_empty_frames(input, filtered_frames, empty_frame_indices)
        input['pcd_frames'] = tuple(filtered_frames)
        return input

    def _is_backup_frame(self, input: dict, idx: int) -> bool:
        remaining = input.get('remaining_frames_idx')
        if remaining is not None and idx < len(remaining):
            return remaining[idx] < self.backup_frames
        return idx < self.backup_frames
    
    def operate_empty_frames(self, input: dict, filtered_frames: List[np.ndarray], empty_frame_indices: List[int]):
        if self.empty_frame_op == 'duplicate':
            empty_set = set(empty_frame_indices)
            for idx in empty_frame_indices:
                if self._is_backup_frame(input, idx):
                    continue # no need to fix the backup frames
                # Prefer previous non-empty frame
                donor = None
                for i in range(idx - 1, -1, -1):
                    if i not in empty_set:
                        donor = i
                        break
                # Fall back to closest next non-empty frame
                if donor is None:
                    for i in range(idx + 1, len(filtered_frames)):
                        if i not in empty_set:
                            donor = i
                            break
                if donor is not None:
                    filtered_frames[idx] = filtered_frames[donor].copy()
            return filtered_frames
        elif self.empty_frame_op == 'shift':
            keep_indices = [i for i in range(len(filtered_frames)) if i not in empty_frame_indices]
            shifted_frames = [filtered_frames[i] for i in keep_indices]
            if len(shifted_frames) < self.min_num_frames:
                raise ValueError(f"There are only {len(shifted_frames)} frames after filtering, but at least {self.min_num_frames} frames are required.")
            apply_frame_selection(input, keep_indices, len(shifted_frames), skip_keys={'pcd_frames'})
            return shifted_frames
        elif self.empty_frame_op == 'error':
            raise RuntimeError("Empty frame found. Current frame should be skipped.")
        else:
            raise ValueError(f"Unknown operation for empty frames: {self.empty_frame_op}")
