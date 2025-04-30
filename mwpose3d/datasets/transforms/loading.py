from pathlib import Path

import h5py
import numpy as np

from .base import BaseTransform
from mwpose3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadSingleFrameFromH5(BaseTransform):
    POINTCLOUD_MODALITY_KEY = 'pcd'
    SKELETON_MODALITY_KEY = 'skel'
    def __init__(self,
                 load_pcd_dim: int
                 ):
        super().__init__()
        self.load_pcd_dim = load_pcd_dim
    
    def transform(self, input: dict):
        local_idx: int = input['local_idx']
        data_files: dict = input['data_file']
        
        pcd_file = data_files[self.POINTCLOUD_MODALITY_KEY]
        skel_file = data_files[self.SKELETON_MODALITY_KEY]

        with h5py.File(pcd_file, 'r') as f:
            grp = f['pcd']
            index = grp['index'][:]
            
            if local_idx < 0 or local_idx >= index[-1]:
                raise IndexError(f'local_idx {local_idx} is out of bounds [0, {index[-1]})')
            start = index[local_idx]
            end = index[local_idx + 1]
            
            pcd_data = grp['data'][start:end]
            pcd_data = pcd_data[:, :self.load_pcd_dim]

        with h5py.File(skel_file, 'r') as f:          
            skel_data = f['skel'][local_idx]
            input['skel_frames'] = (skel_data,)
            input['pcd_frames'] = (pcd_data,)
        return input
    
@TRANSFORMS.register_module()
class LoadMultiFrameFromH5(BaseTransform):
    POINTCLOUD_MODALITY_KEY = 'pcd'
    SKELETON_MODALITY_KEY = 'skel'
    def __init__(self,
                 load_pcd_dim: int,
                 num_frames: int,
                 backup_frames: int = 0,
                 ):
        super().__init__()
        self.backup_frames = backup_frames
        self.load_pcd_dim = load_pcd_dim
        self.num_frames = num_frames
            
    def transform(self, input: dict):
        local_idx: int = input['local_idx']
        data_files: dict = input['data_file']
        pcd_file = data_files[self.POINTCLOUD_MODALITY_KEY]
        skel_file = data_files[self.SKELETON_MODALITY_KEY]
        
        pcd_frames = []
        skel_frames = []
        frame_idx = None

        with h5py.File(pcd_file, 'r') as f:
            grp = f['pcd']
            index = grp['index'][:]
            if 'frame_idx' in grp:
                frame_idx = grp['frame_idx'][:]

            if local_idx < 0 or local_idx >= index[-1]:
                raise IndexError(f'local_idx {local_idx} is out of bounds [0, {index[-1]})')
            
            for i in range(self.num_frames):
                cur_idx = local_idx - i
                if cur_idx < 0:
                    pcd_frames.insert(0, np.zeros((0, self.load_pcd_dim), dtype=np.float32))
                else:
                    start = index[cur_idx]
                    try:
                        end = index[cur_idx + 1]
                    except IndexError:
                        print(index.shape)
                        print(f"IndexError: cur_idx {cur_idx} is out of bounds for index array. i = {i}, local_idx = {local_idx}")
                        raise RuntimeError()
                    pcd_data = grp['data'][start:end]
                    pcd_data = pcd_data[:, :self.load_pcd_dim]
                    pcd_frames.insert(0, pcd_data)
        
            last_backup_idx = cur_idx - 1
            for i in range(self.backup_frames):
                cur_idx = last_backup_idx - i
                if cur_idx < 0:
                    pcd_frames.insert(0, np.zeros((0, self.load_pcd_dim), dtype=np.float32))
                else:
                    start = index[cur_idx]
                    end = index[cur_idx + 1]
                    pcd_data = grp['data'][start:end]
                    pcd_data = pcd_data[:, :self.load_pcd_dim]
                    pcd_frames.insert(0, pcd_data)
            input['pcd_frames'] = tuple(pcd_frames)
        
        with h5py.File(pcd_file, 'r') as pf, h5py.File(skel_file, 'r') as sf:
            # load optional frame‐index map from your pcd HDF5
            frame_idx = pf['pcd'].get('frame_idx', None)
            skel_sequence = sf['skel']

            def to_skel_idx(i):
                # if frame_idx exists, map pcd‐frame i → original skel frame
                return int(frame_idx[i] - 1) if frame_idx is not None else i

            skel_frames = []
            # main frames
            for i in range(self.num_frames):
                idx = to_skel_idx(local_idx - i)
                skel_frames.insert(0, skel_sequence[idx])
            # backup frames
            for i in range(self.backup_frames):
                idx = to_skel_idx(last_backup_idx - i)
                skel_data = skel_sequence[idx]
                assert skel_data.shape[0] == 17 * 3
                skel_frames.insert(0, skel_data)

            input['skel_frames'] = tuple(skel_frames)

        input['target_num_frames'] = self.num_frames
        return input