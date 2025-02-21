from typing import List, Tuple

import numpy as np

from .base import TRANSFORM, BaseTransform

@TRANSFORM.register_module()
class PointCloudRangeFilter(BaseTransform):
    def __init__(self,
                 point_cloud_range: List[float]):
        self.point_cloud_range = point_cloud_range[:6]
    
    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        filtered_frames = []
        for pcd_frame in pcd_frames:
            mask = (pcd_frame[:, 0] >= self.point_cloud_range[0]) & \
                   (pcd_frame[:, 0] <= self.point_cloud_range[3]) & \
                   (pcd_frame[:, 1] >= self.point_cloud_range[1]) & \
                   (pcd_frame[:, 1] <= self.point_cloud_range[4]) & \
                   (pcd_frame[:, 2] >= self.point_cloud_range[2]) & \
                   (pcd_frame[:, 2] <= self.point_cloud_range[5])
            filtered_frames.append(pcd_frame[mask])
        input['pcd_frames'] = tuple(filtered_frames)
        return input
