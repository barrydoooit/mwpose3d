# pointing_detector.py
#
# Stateful pointing gesture detector using geometric heuristics on Kinect V2
# skeleton data. Supports both frame-by-frame (live) and batch (offline) modes.

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────
# Kinect V2 joint indices (USED_KEYPOINTS = 0..19)
# Each joint occupies cols [idx*3 : idx*3+3] = (x, y, z)
# ──────────────────────────────────────────────────────────
_J = dict(
    SPINE_BASE=0,
    SHOULDER_LEFT=4,  ELBOW_LEFT=5,  WRIST_LEFT=6,  HAND_LEFT=7,
    SHOULDER_RIGHT=8, ELBOW_RIGHT=9, WRIST_RIGHT=10, HAND_RIGHT=11,
)

_SIDES = {
    "left":  ("SHOULDER_LEFT",  "ELBOW_LEFT",  "WRIST_LEFT",  "HAND_LEFT"),
    "right": ("SHOULDER_RIGHT", "ELBOW_RIGHT", "WRIST_RIGHT", "HAND_RIGHT"),
}


def _sl(joint_idx: int) -> slice:
    """Column slice for a joint index in a flat 60-element skeleton."""
    return slice(joint_idx * 3, joint_idx * 3 + 3)


class PointingDetector:
    """
    Detects pointing gestures from Kinect V2 skeleton data.

    Works in two modes:

    * **Frame-by-frame** (live):  ``detect_frame(flat_skeleton)``
      Maintains an internal ring buffer for temporal stability.

    * **Batch** (offline):  ``detect_batch(skeleton_array)``
      Processes an entire ``(N, 60)`` array at once, fully vectorised
      except for the sliding-window variance check.

    Parameters
    ----------
    elbow_angle_threshold : float
        Min inner elbow angle (degrees) to count as extended.  Default 140.
    stability_window : int
        Number of frames for the temporal-stability sliding window.  Default 10.
    stability_max_variance : float
        Max mean spatial variance (m²) of the hand position within the
        window.  Default 0.02.
    """

    def __init__(
        self,
        elbow_angle_threshold: float = 90.0,
        stability_window: int = 5,
        stability_max_variance: float = 0.07,
    ) -> None:
        self.elbow_angle_deg = elbow_angle_threshold
        self.stability_window = stability_window
        self.stability_max_var = stability_max_variance

        # ring buffers for live mode (one per side)
        self._buffers: dict[str, deque] = {
            "left":  deque(maxlen=stability_window),
            "right": deque(maxlen=stability_window),
        }

    # ── public API ─────────────────────────────────────────

    def reset(self) -> None:
        """Clear internal state (call between sequences in live mode)."""
        for buf in self._buffers.values():
            buf.clear()

    def detect_frame(self, skeleton_flat: np.ndarray) -> bool:
        """
        Detect pointing in a single skeleton frame.

        Parameters
        ----------
        skeleton_flat : ndarray, shape (60,) or (20, 3)
            Flat Kinect V2 skeleton (20 joints × 3 xyz).

        Returns
        -------
        bool
        """
        flat = np.asarray(skeleton_flat, dtype=np.float64).ravel()
        if flat.size < 60:
            return False

        for side, (sh_name, el_name, wr_name, hd_name) in _SIDES.items():
            shoulder = flat[_sl(_J[sh_name])]
            elbow    = flat[_sl(_J[el_name])]
            wrist    = flat[_sl(_J[wr_name])]
            hand     = flat[_sl(_J[hd_name])]
            spine    = flat[_sl(_J["SPINE_BASE"])]

            if not self._check_extension(shoulder, elbow, wrist):
                self._buffers[side].clear()
                continue
            if not self._check_elevation(hand, shoulder, spine):
                self._buffers[side].clear()
                continue

            # temporal stability — only accumulate while A+B pass
            self._buffers[side].append(hand.copy())
            if self._check_stability(self._buffers[side]):
                return True

        return False

    def detect_batch(self, skel: np.ndarray) -> np.ndarray:
        """
        Detect pointing across an entire sequence.

        Parameters
        ----------
        skel : ndarray, shape (N, 60)
            Full skeleton sequence.

        Returns
        -------
        ndarray of bool, shape (N,)
        """
        n = skel.shape[0]
        flags = np.zeros(n, dtype=bool)

        for side, (sh_name, el_name, wr_name, hd_name) in _SIDES.items():
            shoulder = skel[:, _sl(_J[sh_name])]
            elbow    = skel[:, _sl(_J[el_name])]
            wrist    = skel[:, _sl(_J[wr_name])]
            hand     = skel[:, _sl(_J[hd_name])]
            spine    = skel[:, _sl(_J["SPINE_BASE"])]

            check_a = self._check_extension_batch(shoulder, elbow, wrist)
            check_b = self._check_elevation_batch(hand, shoulder, spine)
            check_c = self._check_stability_batch(hand)

            flags |= (check_a & check_b & check_c)

        return flags

    # ── geometric checks (single frame) ────────────────────

    def _check_extension(self, shoulder, elbow, wrist) -> bool:
    
        u = shoulder - elbow
        v = wrist - elbow
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        if denom < 1e-8:
            return False
        cos_theta = np.clip(np.dot(u, v) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta))) > self.elbow_angle_deg

    def _check_elevation(self, hand, shoulder, spine) -> bool:
        # hand far enough from body core (works for frontal, upward, lateral)
        hand_spine_dist = np.linalg.norm(hand - spine)
        shoulder_spine_dist = np.linalg.norm(shoulder - spine)
        return float(hand_spine_dist) > float(shoulder_spine_dist) * 0.8

    def _check_stability(self, buf: deque) -> bool:
        if len(buf) < 2:
            return False
        window = np.array(buf)
        return float(np.var(window, axis=0).mean()) < self.stability_max_var

    # ── geometric checks (batch / vectorised) ──────────────

    def _check_extension_batch(self, shoulder, elbow, wrist) -> np.ndarray:
        u = shoulder - elbow
        v = wrist - elbow
        cos_theta = np.sum(u * v, axis=1) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-8
        )
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta)) > self.elbow_angle_deg

    def _check_elevation_batch(self, hand, shoulder, spine) -> np.ndarray:
        hand_dist = np.linalg.norm(hand - spine, axis=1)
        shoulder_dist = np.linalg.norm(shoulder - spine, axis=1)
        return hand_dist > shoulder_dist * 0.8

    def _check_stability_batch(self, hand: np.ndarray) -> np.ndarray:
        n = hand.shape[0]
        result = np.zeros(n, dtype=bool)
        half_w = self.stability_window // 2
        for i in range(n):
            lo = max(0, i - half_w)
            hi = min(n, i + half_w + 1)
            if hi - lo < 2:
                continue
            result[i] = np.var(hand[lo:hi], axis=0).mean() < self.stability_max_var
        return result
