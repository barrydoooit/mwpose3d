from pathlib import Path
from typing import List, Literal, Tuple, Union

import h5py
import numpy as np

from mwpose3d.datasets.transforms.loading import LoadMultiFrameFromH5
from mwpose3d.datasets.transforms.utils import apply_frame_selection

from .base import BaseTransform, OnlineEnabled
from mwpose3d.registry import TRANSFORMS



@OnlineEnabled
@TRANSFORMS.register_module()
class SequenceClip(BaseTransform):
    def __init__(self,
                 mode: Literal['random', 'first', 'last'],
                 sequence_length: int,
                 online_mode: bool = False
                 ):
        super().__init__(online_mode)
        self.mode = mode
        assert self.mode in ['random', 'first', 'last']
        self.sequence_length = sequence_length
    
    def transform(self, input: dict):
        pcd_frames: List[np.ndarray] = input[LoadMultiFrameFromH5.PCD_FRAMES]
        original_length = len(pcd_frames)
        assert original_length >= self.sequence_length, f"Input point cloud sequence length {original_length} is less than the required {self.sequence_length}."

        if self.mode == 'random':
            selected_frame_indices = np.random.randint(0, original_length, self.sequence_length)
        elif self.mode == 'first':
            selected_frame_indices = np.arange(self.sequence_length)
        elif self.mode == 'last':
            selected_frame_indices = np.arange(original_length - self.sequence_length, original_length)
        else:
            raise ValueError(f'Invalid mode {self.mode} for SequenceClip.')

        apply_frame_selection(input, selected_frame_indices)
        return input



@OnlineEnabled
@TRANSFORMS.register_module()
class StackPointCloudFrames(BaseTransform):
    def __init__(self, stack_size: int, *, inject_index: bool = False, keep_structure: bool = False, online_mode: bool = False):
        super().__init__(online_mode)
        self.stack_size = stack_size
        self.inject_index = inject_index
        self.keep_structure = keep_structure
    
    def transform(self, input: dict):
        if self.stack_size <= 1:
            return input
        pcd_frames: List[np.ndarray] = input[LoadMultiFrameFromH5.PCD_FRAMES]
        original_length = len(pcd_frames)
        stacked_pcd_frames: List[np.ndarray] = []
        keep_indices = list(range(self.stack_size - 1, original_length))
        apply_frame_selection(input, keep_indices, skip_keys=[LoadMultiFrameFromH5.PCD_FRAMES])
        for i in range(self.stack_size - 1, len(pcd_frames)):
            window = pcd_frames[i - self.stack_size + 1:i + 1] 
            if not self.keep_structure:
                pts = np.concatenate(window, axis=0)  # shape = (sum N_j, 3)
            else:
                pts = np.stack(window, axis=0)

            if self.inject_index:
                lengths = [f.shape[0] for f in window]
                times = np.linspace(0.0, 1.0, num=self.stack_size)
                idx_col = np.repeat(times, lengths).reshape(-1, 1)  # shape = (sum N_j, 1)
                pts = np.concatenate((pts, idx_col), axis=1)        # shape = (sum N_j, 4)

            stacked_pcd_frames.append(pts)
        
        input[LoadMultiFrameFromH5.PCD_FRAMES] = stacked_pcd_frames
        return input

@OnlineEnabled
@TRANSFORMS.register_module()
class DensityFilter(BaseTransform):
    def __init__(self,
                 mode: Literal['threshold', 'lowest_k_percent'] = 'threshold',
                 min_points: int = 5,
                 k_percent: float = 10.0,
                 min_num_frames: int = 1,
                 online_mode: bool = False):
        super().__init__(online_mode)
        self.mode = mode
        self.min_points = int(min_points)
        self.k_percent = float(k_percent)
        self.min_num_frames = int(min_num_frames)
        if not (0.0 <= self.k_percent <= 100.0):
            raise ValueError("k_percent must be in [0, 100].")

    def _counts(self, pcd_frames: Tuple[np.ndarray]) -> List[int]:
        return [int(f.shape[0]) for f in pcd_frames]

    def _keep_indices(self, counts: List[int]) -> List[int]:
        n = len(counts)
        if self.mode == 'threshold':
            keep = [i for i, c in enumerate(counts) if c >= self.min_points]
        elif self.mode == 'lowest_k_percent':
            remove_k = int(np.floor(self.k_percent / 100.0 * n))
            remove_k = max(0, min(remove_k, n))
            if remove_k == 0:
                keep = list(range(n))
            elif remove_k >= n:
                keep = []
            else:
                order = np.argsort(counts, kind='stable')
                remove = set(order[:remove_k].tolist())
                keep = [i for i in range(n) if i not in remove]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return keep
    
    def transform(self, input: dict) -> dict:
        pcd_frames: Tuple[np.ndarray] = input[LoadMultiFrameFromH5.PCD_FRAMES]

        counts = self._counts(pcd_frames)
        keep = self._keep_indices(counts)

        if len(keep) < self.min_num_frames:
            raise ValueError(f"DensityFilter would keep {len(keep)} frames (< min_num_frames={self.min_num_frames}).")

        apply_frame_selection(input, keep)
        return input
                 
