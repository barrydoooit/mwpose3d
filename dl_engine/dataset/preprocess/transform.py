from typing import Literal, Tuple

import numpy as np
import kinect_toolkits as kntk
from .base import TRANSFORM, BaseTransform


@TRANSFORM.register_module()
class RandomFlip(BaseTransform):
    def __init__(self,
                 flip_prob: float = 0.5,
                 duplicate_prob: float = 0.0):
        super().__init__(online_mode=False)
        self.flip_prob = flip_prob
        self.duplicate_prob = duplicate_prob

    def invert_x_axis(self, pcd_frames: Tuple[np.ndarray], skel_frames: Tuple[np.ndarray]):
        for frame in pcd_frames:
            frame[:, 0] = -frame[:, 0]
        for frame in skel_frames:
            keypoints = frame[:len(frame) // 3 * 3].reshape(-1, 3)
            keypoints[:, 0] = -keypoints[:, 0]
            frame[:len(frame) // 3 * 3] = keypoints.flatten()
        return pcd_frames, skel_frames
    
    def mirror_yz_plane(self, pcd_frames: Tuple[np.ndarray], skel_frames: Tuple[np.ndarray], half: Literal[-1, 1]):
        # NOTE: should be called after the skeleton is transformed into radar coordinates
        new_pcd_frames = []
        for frame in pcd_frames:
            pos_mask = frame[:, 0] * half > 0
            pos_points = frame[pos_mask].copy()
            
            mirrored_points = pos_points.copy()
            mirrored_points[:, 0] = -mirrored_points[:, 0]
            new_frame = np.concatenate([frame, mirrored_points], axis=0)
            new_pcd_frames.append(new_frame)
        
        new_skel_frames = []
        assert len(skel_frames[0]) == len(kntk.USED_KEYPOINTS) * 3 + 1
        left_indices = [t.value for t in kntk.LEFT_KEYPOINTS] # positive half
        right_indices = [t.value for t in kntk.RIGHT_KEYPOINTS] # negative half
        indices_to_mirror = left_indices if half > 0 else right_indices
        indices_to_remove = right_indices if half > 0 else left_indices
        for frame in skel_frames:
            keypoints = frame[:len(frame) // 3 * 3].reshape(-1, 3)
            for i, (indice_to_mirror, indice_to_remove) in enumerate(zip(indices_to_mirror, indices_to_remove)):
                to_mirror = keypoints[indice_to_mirror].copy()
                to_mirror[0] = -to_mirror[0]
                keypoints[indice_to_remove] = to_mirror
            frame[:len(frame) // 3 * 3] = keypoints.flatten()
            new_skel_frames.append(frame)
        return new_pcd_frames, new_skel_frames
        
    def transform(self, input: dict):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        skel_frames: Tuple[np.ndarray] = input['skel_frames']

        to_flip = np.random.rand() < self.flip_prob
        if not to_flip:
            return input
        
        to_duplicate = np.random.rand() < self.duplicate_prob
        if not to_duplicate:
            input['pcd_frames'], input['skel_frames'] = self.invert_x_axis(pcd_frames, skel_frames)
            return input
        
        half = 1 if np.random.rand() < 0.5 else -1
        input['pcd_frames'], input['skel_frames'] = self.mirror_yz_plane(pcd_frames, skel_frames, half)
        return input

@TRANSFORM.register_module()
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

        to_transform = np.random.rand() < self.transform_prob
        if not to_transform:
            return input
        
        global_shift = np.array([
            max(-self.max_d_xyz[0], min(self.max_d_xyz[0], np.random.normal(0, self.sigma_xyz[0]))),
            max(-self.max_d_xyz[1], min(self.max_d_xyz[1], np.random.normal(0, self.sigma_xyz[1]))),
            max(-self.max_d_xyz[2], min(self.max_d_xyz[2], np.random.normal(0, self.sigma_xyz[2])))
        ])
        for frame in pcd_frames:
            frame[:, :3] += global_shift
        
        for frame in skel_frames:
            keypoints = frame[:len(frame) // 3 * 3].reshape(-1, 3)
            keypoints += global_shift
            frame[:len(frame) // 3 * 3] = keypoints.flatten()
        return input
    
@TRANSFORM.register_module()
class RandomScale(BaseTransform):
    def __init__(self,
                 scale_prob: float = 0.5,
                 scale_range_x: Tuple[float, float] = (1.0, 1.0),
                 scale_range_y: Tuple[float, float] = (1.0, 1.0),
                 scale_range_z: Tuple[float, float] = (1.0, 1.0)
                 ):
        super().__init__(online_mode=False)
        self.scale_prob = scale_prob
        self.scale_factors = np.array([
            np.random.uniform(scale_range_x[0], scale_range_x[1]),
            np.random.uniform(scale_range_y[0], scale_range_y[1]),
            np.random.uniform(scale_range_z[0], scale_range_z[1])
        ])
    
    
    def transform(self, input):
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        skel_frames: Tuple[np.ndarray] = input['skel_frames']

        to_scale = np.random.rand() < self.scale_prob
        if not to_scale:
            return input
        
        for frame in pcd_frames:
            frame[:, :3] *= self.scale_factors
        
        for frame in skel_frames:
            keypoints = frame[:len(frame) // 3 * 3].reshape(-1, 3)
            keypoints *= self.scale_factors
            frame[:len(frame) // 3 * 3] = keypoints.flatten()
        return input