from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import h5py
import numpy as np

from .base import BaseTransform, OnlineEnabled
from mwpose3d.registry import TRANSFORMS




@TRANSFORMS.register_module()
class LoadTrackingRecords(BaseTransform):
    class AnchorFrameType(Enum):
        FIRSTLAST = 'firstlast'
        FIRSTNEXT = 'firstnext'
        FIRSTLASTTHENFIRSTNEXT = 'firstlastthenfirstnext'
    def __init__(self, 
                 tracker_name: str,
                 anchor_frame: Literal['firstlast', 'firstnext', 'firstlastthenfirstnext'] = 'firstlast',
                 ignore_axis: List[int] = [2],
                 translate: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),):
        super().__init__(online_mode=False)
        self.tracker_name = tracker_name
        self.anchor_frame_type = self.AnchorFrameType(anchor_frame)
        self.ignore_axis = ignore_axis
        self.translate = np.array(translate, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.transform_matrix = self.make_transform_matrix()
    
    def make_transform_matrix(self) -> np.ndarray:
        """Create a transformation matrix for translation and rotation."""
        translation_matrix = np.eye(4, dtype=np.float32)
        translation_matrix[:3, 3] = self.translate
        
        rotation_matrix = np.eye(4, dtype=np.float32)
        # Assuming rotation is in radians and in the order of (x, y, z)
        rotation_matrix[:3, :3] = np.array([
            [np.cos(self.rotation[2]), -np.sin(self.rotation[2]), 0],
            [np.sin(self.rotation[2]), np.cos(self.rotation[2]), 0],
            [0, 0, 1]
        ])
        
        return translation_matrix @ rotation_matrix
    
    def get_solid_track_centroid(self, ds: h5py.Dataset, local_idx: int) -> np.ndarray:
        n = len(ds)
        if not (0 <= local_idx < n):
            raise IndexError(f"local_idx {local_idx} out of range [0, {n-1}]")

        def non_empty(j: int) -> bool:
            a = np.asarray(ds[j])
            return a.size > 0

        def search_backward(start: int):
            for i in range(start, -1, -1):
                if non_empty(i):
                    return np.asarray(ds[i])
            return None

        def search_forward(start: int):
            for i in range(start, n):
                if non_empty(i):
                    return np.asarray(ds[i])
            return None

        if self.anchor_frame_type == self.AnchorFrameType.FIRSTLAST:
            result = search_backward(local_idx)
            if result is None:
                raise ValueError("No non-empty frame found at or before local index.")
            return result

        elif self.anchor_frame_type == self.AnchorFrameType.FIRSTNEXT:
            result = search_forward(local_idx + 1)
            if result is None:
                raise ValueError("No non-empty frame found after local index.")
            return result

        elif self.anchor_frame_type == self.AnchorFrameType.FIRSTLASTTHENFIRSTNEXT:
            result = search_backward(local_idx)
            if result is not None:
                return result
            result = search_forward(local_idx + 1)
            if result is not None:
                return result
            raise ValueError("No non-empty frame found in either direction from local index.")

        else:
            raise ValueError(f"Unknown anchor_frame_type: {self.anchor_frame_type}")

    def transform(self, input: dict) -> dict:
        local_idx: int = input['local_idx']
        data_files: dict = input['data_file']
        pcd_file = Path(data_files['pcd'])
        tracking_file = pcd_file.parent.with_name("tracking_records") / pcd_file.name
        if not tracking_file.exists():
            raise FileNotFoundError(f"Tracking file {tracking_file} does not exist.")
        
        with h5py.File(tracking_file, 'r') as h5f:
            grp = h5f[self.tracker_name]
            ds = grp['track_records']
            track_centroid = self.get_solid_track_centroid(ds, local_idx)
        
        for axis in self.ignore_axis:
            track_centroid[axis] = 0
        if self.transform_matrix is not None:
            track_centroid = np.dot(self.transform_matrix, np.append(track_centroid, 1))[:3]
        input['track_centroid'] = track_centroid
        return input


@OnlineEnabled
@TRANSFORMS.register_module()
class  RelativeCoordtoTrackingCentroid(BaseTransform):
    def __init__(self,
                 discretize_resolution: Optional[Union[int, Tuple[float]]] = None,
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)

        self.discretize_resolution = (tuple([discretize_resolution] * 3) \
            if isinstance(discretize_resolution, (int, float)) else discretize_resolution) \
            if discretize_resolution is not None else None
        
    def transform(self, input: dict):
        track_centroid: np.ndarray = input['track_centroid']
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