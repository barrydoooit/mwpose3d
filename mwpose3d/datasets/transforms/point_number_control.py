from typing import List, Literal, Tuple, Union

import numpy as np

from .base import BaseTransform
from mwpose3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class PointDuplicator(BaseTransform):
    def __init__(self,
                 target_num_points: int,
                 noise_std: float = 0.005,
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)
        self.noise_std = noise_std
        self.target_num_points = target_num_points
    
    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        upsampled_frames = []
        for pcd_frame in pcd_frames:
            num_points = pcd_frame.shape[0]
            if num_points >= self.target_num_points:
                upsampled_frames.append(pcd_frame)
                continue
            if num_points == 0:
                raise ValueError("The point cloud frame is empty.")
            duplicate_idx = np.random.choice(num_points, self.target_num_points - num_points, replace=True)
            duplicate_points = pcd_frame[duplicate_idx]
            
            duplicate_points = duplicate_points.copy()
            rand_shift = np.random.normal(0, self.noise_std, size=(duplicate_points.shape[0], 3))
            rand_shift = np.clip(rand_shift, -0.01, 0.01)
            duplicate_points[:, :3] += rand_shift.astype(duplicate_points.dtype)
            
            # rand_noise = np.random.normal(0, 2, size=duplicate_points.shape[0])
            # duplicate_points[:, -1] += rand_noise.astype(duplicate_points.dtype)
            
            upsampled_frame = np.concatenate([pcd_frame, duplicate_points], axis=0)
            upsampled_frames.append(upsampled_frame)
            
        input['pcd_frames'] = tuple(upsampled_frames)
        return input

    def transform_online(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        num_recent_frames = input.get('num_recent_frames', 1)
        recent_frames = pcd_frames[-num_recent_frames:]
        
        upsampled_recent_frames = []
        for recent_frame in recent_frames:
            num_points = recent_frame.shape[0]
            if num_points >= self.target_num_points:
                upsampled_recent_frames.append(recent_frame)
                continue
            if num_points == 0:
                raise ValueError("The point cloud frame is empty.")
            duplicate_idx = np.random.choice(num_points, self.target_num_points - num_points, replace=True)
            duplicate_points = recent_frame[duplicate_idx]
            
            duplicate_points = duplicate_points.copy()
            noise = np.random.normal(0, self.noise_std, size=(duplicate_points.shape[0], 3))
            duplicate_points[:, :3] += noise.astype(duplicate_points.dtype)
            upsampled_frame = np.concatenate([recent_frame, duplicate_points], axis=0)
            upsampled_recent_frames.append(upsampled_frame)
        
        input['pcd_frames'] = pcd_frames[:-num_recent_frames] + tuple(upsampled_recent_frames)
        return input
    
@TRANSFORMS.register_module()
class PointSortAndClip(BaseTransform):
    def __init__(self,
                 target_num_points: int,
                 sort_dim: Union[int, Tuple[int,]],
                 sort_order: Literal['asc', 'desc'],
                 sort_effective: bool = False,
                 online_mode: bool = False,
                 ):
        super().__init__(online_mode)
        # Normalize sort_dim to a tuple of ints
        if isinstance(sort_dim, int):
            self.sort_dims = (sort_dim,)
        else:
            self.sort_dims = tuple(sort_dim)
        # Ascending or descending
        self.sort_order = 1 if sort_order == "asc" else -1
        self.sort_effective = sort_effective
        self.target_num_points = target_num_points

    def _sort_and_clip(self, frame: np.ndarray) -> np.ndarray:
        # Prepare keys for lexsort: reversed dims so that first dim is primary
        keys = [self.sort_order * frame[:, d] for d in reversed(self.sort_dims)]
        sorted_indices = np.lexsort(keys)

        # Clip and optionally preserve original ordering
        selected = sorted_indices[: self.target_num_points]
        if self.sort_effective:
            return frame[selected]
        else:
            # Preserve original spatial order among selected points
            selected_in_order = np.sort(selected)
            return frame[selected_in_order]

    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray, ...] = input['pcd_frames']
        # Apply sorting and clipping to all frames
        result_frames = [self._sort_and_clip(frame) for frame in pcd_frames]
        input['pcd_frames'] = tuple(result_frames)
        return input

    def transform_online(self, input: dict):
        pcd_frames: Tuple[np.ndarray, ...] = input['pcd_frames']
        num_recent_frames = input.get('num_recent_frames', 1)
        recent = pcd_frames[-num_recent_frames:]

        # Apply sorting and clipping only to recent frames
        clipped_recent = [self._sort_and_clip(frame) for frame in recent]
        input['pcd_frames'] = pcd_frames[:-num_recent_frames] + tuple(clipped_recent)
        return input



@TRANSFORMS.register_module()
class PointPadding(BaseTransform):
    def __init__(self,
                 target_num_points: int,
                 online_mode: bool = False,
                 ):
        super().__init__(online_mode)
        self.target_num_points = target_num_points
    
    def pad(self, pcd_frame: np.ndarray):
        assert pcd_frame.shape[1] >= 3
        n_points = pcd_frame.shape[0]
        if n_points >= self.target_num_points:
            return pcd_frame[:self.target_num_points, :]
        else:
            return np.concatenate([pcd_frame, np.zeros((self.target_num_points - n_points, pcd_frame.shape[1]))], axis=0)
        
    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        result_frames = []
        for pcd_frame in pcd_frames:
            result_frames.append(self.pad(pcd_frame))
        input['pcd_frames'] = tuple(result_frames)
        return input

    def transform_online(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        num_recent_frames = input.get('num_recent_frames', 1) # unprocessed frames
        recent_frames = pcd_frames[-num_recent_frames:]
        
        result_recent_frames = []
        for recent_frame in recent_frames:
            result_recent_frames.append(self.pad(recent_frame))
        input['pcd_frames'] = pcd_frames[:-num_recent_frames] + tuple(result_recent_frames)
        return input