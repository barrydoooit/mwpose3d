from pathlib import Path
import pickle
from typing import TYPE_CHECKING

import numpy as np

from dl_engine.dataset.preprocess.base import TRANSFORM, BaseTransform

from .base import DATASETS



@DATASETS.register_module()
class MotionDataset:
    def __init__(self,
                 info_path: str,
                 data_root: str,
                 pipeline: list,
                 ):
        self.info_path = Path(info_path)
        self.data_root = Path(data_root)   
        assert self.info_path.exists(), f'{self.info_path} does not exist.'
        assert self.data_root.exists(), f'{self.data_root} does not exist.'
        
        print(self.info_path)
        with open(self.info_path, 'rb') as f:
            self.info = pickle.load(f)
        
        self.cum_frames, self.file_names = self._build_global_idx_to_file_table()
        self.total_frames = self.cum_frames[-1] if self.cum_frames.size > 0 else 0
        
        self.pipeline: list['BaseTransform'] = self._build_pipeline(pipeline)
        print("MotionDataset initialized with total_frames:", self.total_frames)
        
    def _build_pipeline(self, pipeline: list):
        if isinstance(pipeline[0], dict):
            return [TRANSFORM.build(p) for p in pipeline]
        else:
            return pipeline
    
    def _build_global_idx_to_file_table(self):
        cum_list = []
        file_list = []
        total_frames = 0
        for file_name, meta in self.info.items():
            frame_count = meta['frame_count']
            total_frames += frame_count
            cum_list.append(total_frames)
            file_list.append(file_name)
        cum_frames = np.array(cum_list, dtype=np.int64)
        file_names = np.array(file_list)
        return cum_frames, file_names
    
    def get_file_by_global_idx(self, global_idx: int):
        if global_idx < 0 or global_idx >= self.total_frames:
            raise IndexError(f"global_idx {global_idx} is out of bounds [0, {self.total_frames})")
        
        file_idx = np.searchsorted(self.cum_frames, global_idx, side='right')
        file_name = self.file_names[file_idx]
        
        if file_idx == 0:
            local_idx = global_idx
        else:
            local_idx = global_idx - self.cum_frames[file_idx - 1]
        
        return file_name, local_idx
    
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx) -> dict:
        file_name, local_idx = self.get_file_by_global_idx(idx)
        sample =  {'global_idx': idx, 
                'local_idx': local_idx,
                'file_path': self.data_root / file_name,}
        for transform in self.pipeline:
            sample = transform(sample)
        return sample