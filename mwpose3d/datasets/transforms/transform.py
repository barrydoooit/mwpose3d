from typing import Literal, Tuple

import numpy as np
from .utils import compose_into, make_row_affine
import mwpose3d.utils.kinect_toolkits as kntk
from .base import BaseTransform
from mwpose3d.registry import TRANSFORMS



@TRANSFORMS.register_module()
class RandomTransform(BaseTransform):
    def __init__(self,
                 transform_prob: float = 0.5,
                 sigma_xyz: Tuple[float, float, float] = (0.15, 0.15, 0.05),
                 max_d_xyz: Tuple[float, float, float] = (0.5, 0.5, 0.5)
                 ):
        super().__init__(online_mode=False)
        self.transform_prob = transform_prob
        self.sigma_xyz = sigma_xyz
        self.max_d_xyz = max_d_xyz
    
    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        skel_frames: Tuple[np.ndarray] = input['skel_frames']

        # if np.random.rand() >= self.transform_prob:
        #     return input
        
        global_shift = np.array([
            np.clip(np.random.normal(0, self.sigma_xyz[0]), -self.max_d_xyz[0], self.max_d_xyz[0]),
            np.clip(np.random.normal(0, self.sigma_xyz[1]), -self.max_d_xyz[1], self.max_d_xyz[1]),
            np.clip(np.random.normal(0, self.sigma_xyz[2]), -self.max_d_xyz[2], self.max_d_xyz[2]),
        ], dtype=np.float32)

        input['pcd_frames'] = tuple(
            np.hstack([f[:, :3] + global_shift, f[:, 3:]]) if f.shape[1] > 3 else (f[:, :3] + global_shift)
            for f in pcd_frames
        )
        input['skel_frames'] = tuple(
            np.concatenate([(f[: (len(f)//3)*3].reshape(-1, 3) + global_shift).ravel(), f[(len(f)//3)*3:]])
            for f in skel_frames
        )

        # Accumulate: same A for all frames in each modality
        A = make_row_affine(R=None, t=global_shift)
        compose_into(input, 'T_pcd',  A, n=len(input['pcd_frames']))
        compose_into(input, 'T_skel', A, n=len(input['skel_frames']))
        return input

@TRANSFORMS.register_module()
class SequenceReverse(BaseTransform):
    def __init__(self, reverse_prob: float = 0.3, velocity_idx: int = 3):
        super().__init__(online_mode=False)
        self.reverse_prob = reverse_prob
        self.velocity_idx = velocity_idx

    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        skel_frames: Tuple[np.ndarray] = input['skel_frames']

        to_reverse = np.random.rand() < self.reverse_prob
        if not to_reverse:
            return input
        
        input['pcd_frames'] = pcd_frames[::-1]
        input['skel_frames'] = skel_frames[::-1]
        
        for frame in input['pcd_frames']:
            frame[:, self.velocity_idx] *= -1
            
        return input

@TRANSFORMS.register_module()
class RandomFrameDrop(BaseTransform):
    def __init__(self, drop_prob: float, max_drop: int, min_frame_len: int=None):
        super().__init__(online_mode=False)
        self.drop_prob = drop_prob
        self.max_drop = max_drop
        self.min_frame_len = min_frame_len
    
    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        skel_frames: Tuple[np.ndarray] = input['skel_frames']

        to_drop = np.random.rand() < self.drop_prob
        if not to_drop:
            return input
        
        drop_count = np.random.randint(1, self.max_drop + 1)
        if self.min_frame_len is None:
            self.min_frame_len = input.get('target_num_frames', 0)
        if len(pcd_frames) - drop_count < self.min_frame_len:
            drop_count = max(0, len(pcd_frames) - self.min_frame_len)
        if drop_count == 0:
            return input
        drop_indices = np.random.choice(len(pcd_frames), drop_count, replace=False)
        
        pcd_frames = [frame for i, frame in enumerate(pcd_frames) if i not in drop_indices]
        skel_frames = [frame for i, frame in enumerate(skel_frames) if i not in drop_indices]
        
        input['pcd_frames'] = pcd_frames
        input['skel_frames'] = skel_frames
        
        return input