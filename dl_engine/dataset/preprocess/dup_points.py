from typing import List, Tuple

import numpy as np

from .base import TRANSFORM, BaseTransform

@TRANSFORM.register_module()
class PointDuplicator(BaseTransform):
    def __init__(self,
                 target_num_points: int,
                 ):
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
                input.pop('pcd_frames')
                input.pop('skel_frames')
                print(input)
                raise ValueError("The point cloud frame is empty.")
            duplicate_idx = np.random.choice(num_points, self.target_num_points - num_points, replace=True)
            duplicate_points = pcd_frame[duplicate_idx]
            
            noise_std = 0.01
            duplicate_points = duplicate_points.copy()
            noise = np.random.normal(0, noise_std, size=(duplicate_points.shape[0], 3))
            duplicate_points[:, :3] += noise.astype(duplicate_points.dtype)
            upsampled_frame = np.concatenate([pcd_frame, duplicate_points], axis=0)
            upsampled_frames.append(upsampled_frame)
            
        input['pcd_frames'] = tuple(upsampled_frames)
        return input
