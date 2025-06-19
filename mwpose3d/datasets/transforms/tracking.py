from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import h5py
import numpy as np

from .base import BaseTransform, OnlineEnabled
from mwpose3d.registry import TRANSFORMS



@OnlineEnabled
@TRANSFORMS.register_module()
class  RelativeCoordtoTrackingCentroid(BaseTransform):
    def __init__(self,
                 tracker_name: str,
                 anchor_frame: Literal['last'] = 'last',
                 discretize_resolution: Optional[Union[int, Tuple[float]]] = None,
                 ignore_axis: List[int] = [2],
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)
        self.tracker_name = tracker_name
        self.ignore_axis = ignore_axis
        self.anchor_frame = anchor_frame
        self.discretize_resolution = (tuple([discretize_resolution] * 3) \
            if isinstance(discretize_resolution, (int, float)) else discretize_resolution) \
            if discretize_resolution is not None else None

    def transform(self, input: dict):
        local_idx: int = input['local_idx']
        data_files: dict = input['data_file']
        pcd_file = Path(data_files['pcd'])
        tracking_file = pcd_file.parent.with_name("tracking_records") / pcd_file.name
        if not tracking_file.exists():
            raise FileNotFoundError(f"Tracking file {tracking_file} does not exist.")
        
        with h5py.File(tracking_file, 'r') as h5f:
            grp = h5f[self.tracker_name]
            ds = grp['track_records']
            if self.anchor_frame == 'last':
                i = local_idx
                while True:
                    track_centroid = ds[i]
                    if track_centroid.shape[0] > 0:
                        break
                    i -= 1
        
        for axis in self.ignore_axis:
            track_centroid[axis] = 0
        if self.discretize_resolution is not None:
            for axis, resolution in enumerate(self.discretize_resolution):
                track_centroid[axis] = np.round(track_centroid[axis] / resolution) * resolution
            
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        relative_pcd_frames = [
            np.hstack([pcd_frames[i][:, :3] - track_centroid[:3], pcd_frames[i][:, 3:]])
            for i in range(len(pcd_frames))
        ]
        input['pcd_frames'] = tuple(relative_pcd_frames)

        if 'skel_frames' in input:
            skel_frames: Tuple[np.ndarray] = input['skel_frames']
            relative_skel_frames = [
                (skel_frames[i].reshape(-1, 3)[:, :3] - track_centroid[:3]).flatten()
                for i in range(len(skel_frames))
            ]
            input['skel_frames'] = tuple(relative_skel_frames)

        return input