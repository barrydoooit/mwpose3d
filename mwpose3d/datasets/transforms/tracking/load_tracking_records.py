from collections import deque
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
from mmengine.config import Config


from ..utils import apply_frame_selection
from ..base import BaseTransform, OnlineEnabled
from mwpose3d.registry import TRANSFORMS

try:
    from mwcore.tracking.api import BaseTracker
    from mwcore.registry import TRACKERS
except ImportError:
    pass


@OnlineEnabled
@TRANSFORMS.register_module()
class LoadTrackingRecords(BaseTransform):
    class AnchorFrameType(Enum):
        FIRSTLAST = 'firstlast'
        FIRSTNEXT = 'firstnext'
        FIRSTLASTTHENFIRSTNEXT = 'firstlastthenfirstnext'
        NEARESTPERFRAME = 'nearestperframe'

    def __init__(self, 
                 tracker_name: str,
                 anchor_frame: Literal[
                     'firstlast',
                     'firstnext',
                     'firstlastthenfirstnext',
                     'nearestperframe'
                 ] = 'firstlast',
                 ignore_axis: List[int] = [2],
                 translate: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 online_mode: bool = False,
                 online_noresult_response: Literal['error', 'empty']='error',
                 tracker_cfg: Union[dict, str, Path] = None,
                 centroid_queue_len: Optional[int] = None,
                 point_cloud_clipping: Optional[dict] = None):
        """
        Args:
            centroid_queue_len: only used in NEARESTPERFRAME + online mode.
                The online result will be a tuple of length `centroid_queue_len`.
        """
        super().__init__(online_mode)
        self.tracker_name = tracker_name
        self.anchor_frame_type = self.AnchorFrameType(anchor_frame)
        self.ignore_axis = ignore_axis
        self.translate = np.array(translate, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.transform_matrix = self.make_transform_matrix()

        if self.anchor_frame_type == self.AnchorFrameType.NEARESTPERFRAME:
            self.centroid_queue_len = int(centroid_queue_len)
        self._centroid_queue: Optional[deque] = None

        self.point_cloud_clipping_dimensions: Optional[Tuple[int]] = None if point_cloud_clipping is None \
              else tuple(point_cloud_clipping.get('dimensions', None))
        self.point_cloud_clipping_threshold: Optional[Tuple[Tuple[Optional[float]]]] = None if point_cloud_clipping is None \
              else tuple(point_cloud_clipping.get('threshold', None))
        if self.online_mode:
            assert tracker_cfg is not None, "tracker_cfg must be provided in online mode"
            if isinstance(tracker_cfg, (str, Path)):
                tracker_cfg = Config.fromfile(tracker_cfg).get('tracker_cfg')
            self.tracker_cfg = tracker_cfg
            self._tracker = None
            self._last_centroid = None
            self.online_noresult_response = online_noresult_response
    
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

    def _coerce_first_xyz(self, arr_like: np.ndarray) -> np.ndarray:
        a = np.asarray(arr_like, dtype=np.float32)
        if a.size < 3:
            raise ValueError("Centroid array has fewer than 3 elements.")
        if a.ndim == 1:
            c = a[:3]
        else:
            c = a.reshape(-1)[:3]
        return c.astype(np.float32, copy=False)

    def _postprocess_centroid(self, centroid: np.ndarray) -> np.ndarray:
        c = centroid.astype(np.float32, copy=True)
        for axis in self.ignore_axis:
            if 0 <= axis < 3:
                c[axis] = 0.0
        if self.transform_matrix is not None:
            c = (self.transform_matrix @ np.append(c, 1.0))[:3].astype(np.float32)
        return c

    def _nearest_centroid(self, ds: h5py.Dataset, target_idx: int) -> np.ndarray:
        """Previous-first-then-next search around target_idx in [0, n-1]."""
        n = len(ds)
        if n == 0:
            raise ValueError("Empty tracking dataset.")

        def non_empty(j: int) -> bool:
            a = np.asarray(ds[j])
            return a.size > 0

        # clamp the search seeds to valid bounds
        start_back = min(max(target_idx, 0), n - 1)
        for i in range(start_back, -1, -1):
            if non_empty(i):
                return self._coerce_first_xyz(ds[i])

        start_fwd = min(max(target_idx + 1, 0), n)  # could be n (meaning no forward frames)
        for i in range(start_fwd, n):
            if non_empty(i):
                return self._coerce_first_xyz(ds[i])

        raise ValueError("No non-empty frame found in either direction.")

    def _clip_point_cloud(self, pcd: np.ndarray) -> np.ndarray:
        if self.point_cloud_clipping_dimensions is not None and self.point_cloud_clipping_threshold is not None:
            for dim, thresh in zip(self.point_cloud_clipping_dimensions, self.point_cloud_clipping_threshold):
                if thresh is not None:
                    if thresh[0] is None:
                        pcd = pcd[pcd[:, dim] <= thresh[1]]
                    elif thresh[1] is None:
                        pcd = pcd[pcd[:, dim] >= thresh[0]]
                    else:
                        pcd = pcd[(pcd[:, dim] >= thresh[0]) & (pcd[:, dim] <= thresh[1])]
        return pcd

    def tracker_consume(self, pcd_frame: np.ndarray) -> Optional[np.ndarray]:
        if pcd_frame.shape[1] < 5:
            pcd_frame = np.pad(pcd_frame[:, :3], ((0, 0), (0, 2)), mode='constant')
        if pcd_frame.shape[1] > 5:
            pcd_frame = pcd_frame[:, :5]
        if self.point_cloud_clipping_dimensions is not None and self.point_cloud_clipping_threshold is not None:
            pcd_frame = self._clip_point_cloud(pcd_frame)
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
          Maintain a sliding window (deque) of length `centroid_queue_len` with one centroid per frame.
          For each new frame:
            * If tracker yields a centroid, append it.
            * Else, duplicate the last appended centroid.
          If the window is not yet full after processing the provided frames, raise RuntimeError or append an empty centroid.
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
           
            if self.anchor_frame_type == self.AnchorFrameType.NEARESTPERFRAME:
                self._centroid_queue = deque(maxlen=self.centroid_queue_len)
                # Prime the queue using all provided frames (chronological).
                for frame in (pcd_frames if isinstance(pcd_frames, (list, tuple)) else [pcd_frames]):
                    c = self.tracker_consume(frame if not isinstance(frame, (list, tuple)) else frame[-1])
                    if c is None:
                        # duplicate last if exists; otherwise we can't fill yet
                        if len(self._centroid_queue) > 0:
                            self._centroid_queue.append(self._centroid_queue[-1].copy())
                        # If queue empty, skip
                    else:
                        c = self._postprocess_centroid(c.astype(np.float32, copy=False))
                        self._centroid_queue.append(c)
                        self._last_centroid = c.copy()
                if len(self._centroid_queue) < self.centroid_queue_len:
                    if self.online_noresult_response == 'error':
                        raise RuntimeError(
                            f"Centroid queue not fulfilled at start "
                            f"({len(self._centroid_queue)}/{self.centroid_queue_len}).")
                    elif self.online_noresult_response == 'empty':
                        while len(self._centroid_queue) < self.centroid_queue_len:
                            self._centroid_queue.insert(0, np.zeros((3,), dtype=np.float32))
                input['track_centroid'] = tuple(self._centroid_queue)
                apply_frame_selection(input)
                return input
            else:
                # Old modes: consume all but the last to update internal tracker state
                for frame in pcd_frames[:-1]:
                    result = self.tracker_consume(frame)
                    if result is not None:
                        self._last_centroid = result.copy()
        
        # After possible reset above, proceed per mode
        if self.anchor_frame_type == self.AnchorFrameType.NEARESTPERFRAME:
            # treat only the newest frame here (sliding window update)
            if isinstance(pcd_frames[-1], (list, tuple)):
                current_pcd_frame = pcd_frames[-1][-1]
            else:
                current_pcd_frame = pcd_frames[-1]

            c = self.tracker_consume(current_pcd_frame)
            if c is None:
                if self._centroid_queue is None or len(self._centroid_queue) == 0:
                    raise RuntimeError("No centroid available to duplicate; queue empty.")
                # duplicate last
                self._centroid_queue.append(self._centroid_queue[-1].copy())
            else:
                c = self._postprocess_centroid(c.astype(np.float32, copy=False))
                if self._centroid_queue is None:
                    self._centroid_queue = deque(maxlen=self.centroid_queue_len)
                self._centroid_queue.append(c)
                self._last_centroid = c.copy()

            if len(self._centroid_queue) < self.centroid_queue_len:
                raise RuntimeError(
                    f"Centroid queue not fulfilled "
                    f"({len(self._centroid_queue)}/{self.centroid_queue_len}).")

            input['track_centroid'] = tuple(self._centroid_queue)
            apply_frame_selection(input)
            return input

        # ------- original (single-centroid) online path for the 3 legacy modes -------
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
        """Create a transformation matrix for translation and rotation (Z-only rotation here)."""
        translation_matrix = np.eye(4, dtype=np.float32)
        translation_matrix[:3, 3] = self.translate
        
        rotation_matrix = np.eye(4, dtype=np.float32)
        # Assuming rotation is in radians and in the order of (x, y, z) but only Z is applied here.
        rotation_matrix[:3, :3] = np.array([
            [np.cos(self.rotation[2]), -np.sin(self.rotation[2]), 0],
            [np.sin(self.rotation[2]),  np.cos(self.rotation[2]), 0],
            [0,                         0,                        1]
        ], dtype=np.float32)
        
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
            if self.anchor_frame_type == self.AnchorFrameType.NEARESTPERFRAME:
                # Resolve pcd_frames and compute per-frame mapping
                pcd_frames = input.get('pcd_frames', None)
                num_frames = len(pcd_frames) if isinstance(pcd_frames, (list, tuple)) else 1

                centroids: List[np.ndarray] = []
                for j in range(num_frames):
                    # Map j-th pcd frame to dataset index (chronological window ending at local_idx)
                    target_idx = local_idx - (num_frames - 1 - j)
                    c = self._nearest_centroid(ds, target_idx)
                    c = self._postprocess_centroid(self._coerce_first_xyz(c))
                    centroids.append(c.astype(np.float32))

                input['track_centroid'] = tuple(centroids)
                return input

            track_centroid = self.get_solid_track_centroid(ds, local_idx)
        
        for axis in self.ignore_axis:
            track_centroid[axis] = 0
        if self.transform_matrix is not None:
            track_centroid = np.dot(self.transform_matrix, np.append(track_centroid, 1))[:3]
        input['track_centroid'] = track_centroid.astype(np.float32)
        apply_frame_selection(input)
        return input
