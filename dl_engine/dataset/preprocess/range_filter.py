from typing import List, Tuple

import h5py
import numpy as np

from .base import TRANSFORM, BaseTransform

@TRANSFORM.register_module()
class PointCloudRangeFilter(BaseTransform):
    def __init__(self,
                 point_cloud_range: List[float],
                 load_pcd_dim: int,
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)
        self.point_cloud_range = point_cloud_range[:6]
        self.load_pcd_dim = load_pcd_dim
    
    def get_filtered_frames(self, pcd_frames: Tuple[np.ndarray]):
        filtered_frames = []
        for pcd_frame in pcd_frames:
            mask = (pcd_frame[:, 0] >= self.point_cloud_range[0]) & \
                   (pcd_frame[:, 0] <= self.point_cloud_range[3]) & \
                   (pcd_frame[:, 1] >= self.point_cloud_range[1]) & \
                   (pcd_frame[:, 1] <= self.point_cloud_range[4]) & \
                   (pcd_frame[:, 2] >= self.point_cloud_range[2]) & \
                   (pcd_frame[:, 2] <= self.point_cloud_range[5])
            filtered_frames.append(pcd_frame[mask])
        return filtered_frames
    
    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        filtered_frames = self.get_filtered_frames(pcd_frames)
        empty_frame_indices = [i for i, frame in enumerate(filtered_frames) if frame.shape[0] == 0]
        if len(empty_frame_indices) > 0:
            with h5py.File(input['file_path'], 'r') as f:
                for idx in empty_frame_indices:
                    filtered_frames[idx] = self.find_previous_non_empty_frame(input['local_idx'] - len(pcd_frames) + 1 + idx, f)
        input['pcd_frames'] = tuple(filtered_frames)
        return input

    def find_previous_non_empty_frame(self, local_idx: int, file: h5py.File):
        grp = file['pcd']
        index = grp['index'][:]
        for i in range(local_idx - 1, -1, -1):
            start = index[i]
            end = index[i + 1]
            pcd_data = grp['data'][start:end]
            filtered_frame = self.get_filtered_frames((pcd_data[:,:self.load_pcd_dim],))[0]
            if filtered_frame.shape[0] > 0:
                return filtered_frame
        raise ValueError("No previous non-empty frame found.")
    
    def transform_online(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        num_recent_frames = input.get('num_recent_frames', 1)
        recent_frames = pcd_frames[-num_recent_frames:]
        
        filtered_recent_frames = self.get_filtered_frames(recent_frames)
        empty_frame_indices = [i for i, frame in enumerate(filtered_recent_frames) if frame.shape[0] == 0]
        
        if empty_frame_indices:
            raise RuntimeError("Empty frame found in online mode. Current frame should be skipped.")
            # for idx in empty_frame_indices:
            #     if idx == 0:
            #         if len(pcd_frames) <= num_recent_frames:
            #             raise ValueError("Sequence starts with an empty frame.")
            #         filtered_recent_frames[idx] = pcd_frames[-num_recent_frames - 1].copy()
            #     else:
            #         filtered_recent_frames[idx] = filtered_recent_frames[idx - 1].copy()
        input['pcd_frames'] = tuple(pcd_frames[:-num_recent_frames] + filtered_recent_frames)
        
        return input        