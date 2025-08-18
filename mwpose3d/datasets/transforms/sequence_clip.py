from pathlib import Path
from typing import List, Literal

import h5py
import numpy as np

from .base import BaseTransform, OnlineEnabled
from mwpose3d.registry import TRANSFORMS



@OnlineEnabled
@TRANSFORMS.register_module()
class SequenceClip(BaseTransform):
    def __init__(self,
                 mode: Literal['random', 'first', 'last'],
                 sequence_length: int
                 ):
        super().__init__()
        self.mode = mode
        assert self.mode in ['random', 'first', 'last']
        self.sequence_length = sequence_length
    
    def transform(self, input: dict):
        pcd_frames: List[np.ndarray] = input['pcd_frames']
        original_length = len(pcd_frames)
        assert original_length >= self.sequence_length
        
        if self.mode == 'random':
            selected_frame_indices = np.random.randint(0, original_length, self.sequence_length)
        elif self.mode == 'first':
            selected_frame_indices = np.arange(self.sequence_length)
        elif self.mode == 'last':
            selected_frame_indices = np.arange(original_length - self.sequence_length, original_length)
        else:
            raise ValueError(f'Invalid mode {self.mode} for SequenceClip.')

        pcd_frames = [pcd_frames[i] for i in selected_frame_indices]
        input['pcd_frames'] = pcd_frames

        if 'skel_frames' in input:
            skel_frames: List[np.ndarray] = input['skel_frames']
            assert original_length == len(skel_frames)
            skel_frames = [skel_frames[i] for i in selected_frame_indices]
            input['skel_frames'] = skel_frames
        return input



@OnlineEnabled
@TRANSFORMS.register_module()
class StackPointCloudFrames(BaseTransform):
    def __init__(self, stack_size: int, *, inject_index: bool = False, keep_structure: bool = False, ):
        super().__init__()
        self.stack_size = stack_size
        self.inject_index = inject_index
        self.keep_structure = keep_structure
    
    def transform(self, input: dict):
        if self.stack_size <= 1:
            return input
        pcd_frames: List[np.ndarray] = input['pcd_frames']
        original_length = len(pcd_frames)
        stacked_pcd_frames: List[np.ndarray] = []
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
        
        input['pcd_frames'] = stacked_pcd_frames

        for key, value in input.items():
            if (isinstance(value, list) or isinstance(value, tuple)) and len(value) == original_length:
                input[key] = value[self.stack_size - 1:]
        
        return input
