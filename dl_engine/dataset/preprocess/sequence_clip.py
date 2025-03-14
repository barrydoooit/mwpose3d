from pathlib import Path
from typing import List, Literal

import h5py
import numpy as np

from .base import TRANSFORM, BaseTransform


@TRANSFORM.register_module()
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
        skel_frames: List[np.ndarray] = input['skel_frames']
        
        assert len(pcd_frames) == len(skel_frames)
        assert len(pcd_frames) >= self.sequence_length
        
        if self.mode == 'random':
            selected_frame_indices = np.random.randint(0, len(pcd_frames), self.sequence_length)
        elif self.mode == 'first':
            selected_frame_indices = np.arange(self.sequence_length)
        elif self.mode == 'last':
            selected_frame_indices = np.arange(len(pcd_frames) - self.sequence_length, len(pcd_frames))
        else:
            raise ValueError(f'Invalid mode {self.mode} for SequenceClip.')

        pcd_frames = [pcd_frames[i] for i in selected_frame_indices]
        skel_frames = [skel_frames[i] for i in selected_frame_indices]
        
        input['pcd_frames'] = pcd_frames
        input['skel_frames'] = skel_frames
        
        return input
        
        