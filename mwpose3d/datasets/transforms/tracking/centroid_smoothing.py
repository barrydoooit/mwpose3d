from typing import Optional, Tuple, Union

import numpy as np

from ..base import BaseTransform, OnlineEnabled
from mwpose3d.registry import TRANSFORMS
try:
    from mwcore.utils.smoothing.savgol_filter import savgol_filter, SavGolayConfig
except ImportError:
    pass


@OnlineEnabled
@TRANSFORMS.register_module()
class SmoothingTrackingCentroid(BaseTransform):
    def __init__(self,
                 jitter_radius: float = 0.1,      # inner hold radius (jr)
                 release_scale: float = 1.5,      # rr = release_scale * jr
                 min_jitter_len: int = 2,         # min samples to treat as jitter
                 still_frames: int = 3,           # frames <= jr speed to re-enter jitter
                 window_length: int =  9, # SG window (odd)
                 polyorder: int = 2,              # SG poly order
                 online_mode: bool = False):
        super().__init__(online_mode)
        self.jitter_radius = float(jitter_radius)
        self.release_scale = float(release_scale)
        self.min_jitter_len = int(min_jitter_len)
        self.still_frames = int(still_frames)
        self.window_length = int(window_length)
        self.polyorder = int(polyorder)

    def transform(self, input: dict):
        centroids: Tuple[np.ndarray, ...] = input['track_centroid']  # (T,) of (3,)
        if not centroids:
            return input

        # Stack to (T, 3)
        arr = np.asarray(centroids, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        T = arr.shape[0]
        if T == 0:
            return input

        xy = arr[:, :2].copy()
        z = arr[:, 2].copy()

        jr = self.jitter_radius
        rr = self.release_scale * jr

        # --- 1) Detect jitter segments (start, end) using anchor + hysteresis ---
        segs = []  # list of (start, end, anchor_xy)
        in_jitter = True
        start = 0
        anchor = xy[0].copy()
        still_cnt = 0

        for t in range(1, T):
            p = xy[t]
            prev = xy[t - 1]
            if in_jitter:
                if np.linalg.norm(p - anchor) >= rr:
                    # close jitter if long enough
                    if (t - start) >= self.min_jitter_len:
                        med = np.median(xy[start:t], axis=0)
                        segs.append((start, t, med))
                    in_jitter = False
                    still_cnt = 0
                # else remain in jitter (anchor stays fixed; we’ll recompute with median on close)
            else:
                # moving: detect return to stillness by small step size
                sp = np.linalg.norm(p - prev)
                if sp <= jr:
                    still_cnt += 1
                    if still_cnt >= self.still_frames:
                        in_jitter = True
                        start = t - self.still_frames + 1
                        anchor = xy[start].copy()
                else:
                    still_cnt = 0

        # close trailing jitter
        if in_jitter and (T - start) >= self.min_jitter_len:
            med = np.median(xy[start:T], axis=0)
            segs.append((start, T, med))

        # --- 2) Collapse jitter segments to plateaus (XY only) ---
        plateau_xy = xy.copy()
        jitter_mask = np.zeros(T, dtype=bool)
        for s, e, anc in segs:
            plateau_xy[s:e] = anc
            jitter_mask[s:e] = True

        # --- 3) Piecewise Savitzky–Golay on moving spans (XY only) ---
        y_xy = plateau_xy.copy()
        idx = 0
        while idx < T:
            # skip jitter
            while idx < T and jitter_mask[idx]:
                idx += 1
            if idx >= T:
                break
            start_mv = idx
            while idx < T and not jitter_mask[idx]:
                idx += 1
            end_mv = idx  # exclusive

            span_len = end_mv - start_mv
            if span_len <= 2:
                continue

            w = min(self.window_length, span_len if (span_len % 2 == 1) else span_len - 1)
            if w < 3:
                continue
            p = min(self.polyorder, w - 1)

            local_cfg = SavGolayConfig(window_length=w, polyorder=p, mode='edge')

            span = y_xy[start_mv:end_mv]
            span[:, 0] = savgol_filter(span[:, 0], axis=0, config=local_cfg)
            span[:, 1] = savgol_filter(span[:, 1], axis=0, config=local_cfg)

            if start_mv - 1 >= 0 and jitter_mask[start_mv - 1]:
                y_xy[start_mv] = y_xy[start_mv - 1]
            if end_mv < T and jitter_mask[end_mv]:
                y_xy[end_mv - 1] = y_xy[end_mv]

        out = np.column_stack([y_xy, z])

        # Write back as a tuple of np.ndarray (shape (3,))
        input['track_centroid'] = tuple(out.astype(arr.dtype))
        return input
