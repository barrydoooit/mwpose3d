from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
from mmengine.config import Config

from .utils import compose_into, make_row_affine
from .base import BaseTransform, OnlineEnabled
from mwpose3d.registry import TRANSFORMS
from mwcore.tracking.api import BaseTracker
from mwcore.registry import TRACKERS


@OnlineEnabled
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
                 rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 online_mode: bool = False,
                 tracker_cfg: Union[dict, str, Path] = None):
        super().__init__(online_mode)
        self.tracker_name = tracker_name
        self.anchor_frame_type = self.AnchorFrameType(anchor_frame)
        self.ignore_axis = ignore_axis
        self.translate = np.array(translate, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.transform_matrix = self.make_transform_matrix()

        if self.online_mode:
            assert tracker_cfg is not None, "tracker_cfg must be provided in online mode"
            if isinstance(tracker_cfg, (str, Path)):
                tracker_cfg = Config.fromfile(tracker_cfg).get('tracker_cfg')
            self.tracker_cfg = tracker_cfg
            self._tracker = None
            self._last_centroid = None
    
    @property
    def tracker(self) -> "BaseTracker":
        if self._tracker is None:
            self._tracker = self._build_tracker()
        return self._tracker
    
    def _build_tracker(self) -> "BaseTracker":
        cfg = deepcopy(self.tracker_cfg)
        cfg['radar_cfg'] = dict(cfg.get('radar_cfg', {}), sensor_height=0.0, sensor_tilt=0.0)
        self._tracker = TRACKERS.build(cfg)
        return self._tracker

    def tracker_consume(self, pcd_frame: np.ndarray) -> Optional[np.ndarray]:
        if pcd_frame.shape[1] < 5:
            pcd_frame = np.pad(pcd_frame[:, :3], ((0, 0), (0, 2)), mode='constant')
        if pcd_frame.shape[1] > 5:
            pcd_frame = pcd_frame[:, :5]
        if hasattr(self.tracker, 'sort_results'):
            tracked_locations = self.tracker.consume(point_array=pcd_frame, sort_metric='snr')
        else:
            tracked_locations = self.tracker.consume(point_array=pcd_frame)
        if tracked_locations is None or len(tracked_locations) == 0:
            return None
        arr = np.asarray(tracked_locations, dtype=np.float32)
        if arr.size >= 3:
            if arr.ndim == 1:
                return arr[:3].astype(np.float32)
            elif arr.ndim == 2 and arr.shape[1] >= 3 and arr.shape[0] > 0:
                return arr[0, :3].astype(np.float32)
        return None
        
    def transform_online(self, input: dict) -> dict:
        """
        - If current frame has no result, reuse last non-empty centroid.
        - If no previous centroid exists, raise RuntimeError.
        """
        if not self.online_mode:
            raise RuntimeError("transform_online called while online_mode=False")

        if 'pcd_frames' in input:
            pcd_frames = input['pcd_frames']
        else:
            raise KeyError("Input missing 'pcd_frames' for online tracking.")

        # Reset state & (re)create tracker at the start of a sequence
        if input.get('starting_flag', False) or self._tracker is None:
            self._tracker = self._build_tracker()
            self._last_centroid = None
            for frame in pcd_frames[:-1]:
                result = self.tracker_consume(frame)
                if result is not None:
                    self._last_centroid = result.copy()
        
        if isinstance(pcd_frames[-1], (list, tuple)):
            current_pcd_frame = pcd_frames[-1][-1]
        else:
            current_pcd_frame = pcd_frames[-1]
        
        centroid = self.tracker_consume(current_pcd_frame)
        # Fallback to last known centroid if current frame is empty
        if centroid is None:
            if self._last_centroid is None:
                raise RuntimeError("No tracking result yet and no previous centroid to reuse.")
            centroid = self._last_centroid.copy()
        else:
            centroid = centroid.astype(np.float32, copy=False)

        for axis in self.ignore_axis:
            if 0 <= axis < 3:
                centroid[axis] = 0.0

        if self.transform_matrix is not None:
            centroid = (self.transform_matrix @ np.append(centroid, 1.0))[:3].astype(np.float32)

        self._last_centroid = centroid.copy()
        input['track_centroid'] = centroid
        return input
    
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
class RelativeCoordtoTrackingCentroid(BaseTransform):
    def __init__(self,
                 discretize_resolution: Optional[Union[int, Tuple[float]]] = None,
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)
        self.discretize_resolution = (tuple([discretize_resolution] * 3)
            if isinstance(discretize_resolution, (int, float)) else discretize_resolution) \
            if discretize_resolution is not None else None
        
    def transform(self, input: dict):
        track_centroid: np.ndarray = input['track_centroid'].astype(np.float32)
        if self.discretize_resolution is not None:
            for axis, resolution in enumerate(self.discretize_resolution):
                track_centroid[axis] = np.round(track_centroid[axis] / resolution) * resolution
        
        # Translation by -centroid
        t = -track_centroid[:3].astype(np.float32)

        # Point clouds
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        input['pcd_frames'] = tuple(
            np.hstack([f[:, :3] + t, f[:, 3:]]) if f.shape[1] > 3 else (f[:, :3] + t)
            for f in pcd_frames
        )

        # Skeletons (if present)
        if 'skel_frames' in input and input['skel_frames'] is not None:
            skel_frames: Tuple[np.ndarray] = input['skel_frames']
            input['skel_frames'] = tuple(
                (f.reshape(-1, 3)[:, :3] + t).reshape(-1)  # keep any extra tail? if exists, concat it:
                if (len(f) % 3) == 0 else np.concatenate([(f[: (len(f)//3)*3].reshape(-1,3) + t).ravel(), f[(len(f)//3)*3:]])
                for f in skel_frames
            )

        # Accumulate for whichever modalities are in the dict
        A = make_row_affine(R=None, t=t)
        compose_into(input, 'T_pcd',  A)
        compose_into(input, 'T_skel', A)
        return input