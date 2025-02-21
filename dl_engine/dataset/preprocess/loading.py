from pathlib import Path

import h5py
import numpy as np

from .base import TRANSFORM, BaseTransform


@TRANSFORM.register_module()
class LoadSingleFrameFromH5(BaseTransform):
    def __init__(self,
                 load_pcd_dim: int
                 ):
        self.load_pcd_dim = load_pcd_dim
    
    def transform(self, input: dict):
        local_idx: int = input['local_idx']
        file_path: Path = input['file_path']
        
        with h5py.File(file_path, 'r') as f:
            grp = f['pcd']
            index = grp['index'][:]
            
            if local_idx < 0 or local_idx >= index[-1]:
                raise IndexError(f'local_idx {local_idx} is out of bounds [0, {index[-1]})')
            start = index[local_idx]
            end = index[local_idx + 1]
            
            pcd_data = grp['data'][start:end]
            pcd_data = pcd_data[:, :self.load_pcd_dim]
            
            skel_data = f['skel'][local_idx]
            input['skel_frames'] = (skel_data,)
            input['pcd_frames'] = (pcd_data,)
        return input
    
@TRANSFORM.register_module()
class LoadMultiFrameFromH5(BaseTransform):
    def __init__(self,
                 load_pcd_dim: int,
                 num_frames: int
                 ):
        self.load_pcd_dim = load_pcd_dim
        self.num_frames = num_frames
    
    def transform(self, input: dict):
        local_idx: int = input['local_idx']
        file_path: Path = input['file_path']
        
        pcd_frames = []
        
        with h5py.File(file_path, 'r') as f:
            grp = f['pcd']
            index = grp['index'][:]
            
            if local_idx < 0 or local_idx >= index[-1]:
                raise IndexError(f'local_idx {local_idx} is out of bounds [0, {index[-1]})')
            
            for i in range(self.num_frames):
                cur_idx = local_idx - i
                if cur_idx < 0:
                    pcd_frames.insert(0, np.zeros((0, self.load_pcd_dim), dtype=np.float32))
                else:
                    start = index[cur_idx]
                    end = index[cur_idx + 1]
                    pcd_data = grp['data'][start:end]
                    pcd_data = pcd_data[:, :self.load_pcd_dim]
                    pcd_frames.insert(0, pcd_data)
            
            skel_data = f['skel'][local_idx]
            input['skel_frames'] = (skel_data,)
            input['pcd_frames'] = tuple(pcd_frames)
        return input