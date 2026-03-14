from copy import copy, deepcopy
from pathlib import Path
import pickle
from typing import TYPE_CHECKING
from mmengine.dataset import Compose
from mmengine.fileio import join_path
import numpy as np

from mwpose3d.datasets.transforms.base import TRANSFORMS, BaseTransform

from mwpose3d.registry import DATASETS



@DATASETS.register_module()
class MotionDataset:
    def __init__(self,
                 info_path: str,
                 data_root: str,
                 data_prefix: dict, # e.g. {'pcd': 'mmwave_filtered', 'skel': 'skeleton'}
                 pipeline: list,
                 sequence_length: int = 1,
                 allow_pad_sequence: bool = False,
                 max_sequences: int = None
                 ):
        self.info_path = Path(info_path)
        self.data_root = Path(data_root)
        self.data_prefix = copy(data_prefix)
        assert self.info_path.exists(), f'{self.info_path} does not exist.'
        assert self.data_root.exists(), f'{self.data_root} does not exist.'
        
        self.sequence_length = sequence_length
        self.allow_pad_sequence = allow_pad_sequence
        
        with open(self.info_path, 'rb') as f:
            self.info: list = pickle.load(f)

        if max_sequences is not None:
            self.info = self.info[:max_sequences]

        self.cum_frames, self.file_names = self._build_global_idx_to_file_table()
        self.total_frames = self.cum_frames[-1] if self.cum_frames.size > 0 else 0
        
        self.pipeline: Compose = Compose(pipeline) # NOTE: This will only work when default range is moved to mwpose3d from mmengine (like in the init of runner class)
        print(f"MotionDataset initialized: {len(self.info)} sequences, {self.total_frames} total frames")

    @staticmethod
    def _resolve_prefix_root_dir(prefix: str, modality: str = ''):
        return prefix.split('.')[0]

    @staticmethod
    def _resolve_path(prefix: str, modality: str, info_single: dict):
        path_key = f"{MotionDataset._resolve_prefix_root_dir(prefix, modality)}_path"
        path_info = info_single.get(path_key, None)
        if path_info is None:
            raise KeyError(f"Key '{path_key}' is necessary but not found in info_single. Available keys: {list(info_single.keys())}")
        if isinstance(path_info, str):
            return path_info
        if isinstance(path_info, dict):
            node = path_info
            for key in prefix.split('.')[1:]:
                node = node[key]
            return node
        raise TypeError(f"Unsupported type for path_info under key '{path_key}': {type(path_info)}. Expected str or dict.")

    def _build_global_idx_to_file_table(self):
        cum_list = []
        file_lists = {modality: [] for modality in list(self.data_prefix.keys())} # {'pcd': [], 'skel': []}
        total_frames = 0
        for info_single in self.info:
            frame_count = info_single['frame_count']
            if isinstance(frame_count, dict):
                # print(frame_count)
                frame_count = min(frame_count.values())
            if self.sequence_length > 1 and not self.allow_pad_sequence:
                valid_frames = max(frame_count - (self.sequence_length - 1), 0)
            else:
                valid_frames = frame_count
            total_frames += valid_frames
            cum_list.append(total_frames)
            for modality in list(self.data_prefix.keys()):
                prefix = self.data_prefix[modality]
                prefix_root_dir = self._resolve_prefix_root_dir(prefix, modality)
                file_lists[modality].append(join_path(
                    self.data_root,
                    prefix_root_dir,
                    self._resolve_path(prefix, modality, info_single),
                ))
        cum_frames = np.array(cum_list, dtype=np.int64)
        file_names = {modality: np.array(file_lists[modality]) for modality in list(self.data_prefix.keys())}
        return cum_frames, file_names
    
    def get_info_by_global_idx(self, global_idx: int):
        if global_idx < 0 or global_idx >= self.total_frames:
            raise IndexError(f"global_idx {global_idx} is out of bounds [0, {self.total_frames})")
        
        file_idx = np.searchsorted(self.cum_frames, global_idx, side='right')
        info: dict = deepcopy(self.info[file_idx])

        if file_idx == 0:
            valid_local_idx = global_idx
        else:
            valid_local_idx = global_idx - self.cum_frames[file_idx - 1]
        
        if self.sequence_length > 1 and not self.allow_pad_sequence:
            local_idx = valid_local_idx + (self.sequence_length - 1)
        else:
            local_idx = valid_local_idx
        
        if "snippets" in info:
            assert len(info["snippets"]) == 1, "Only one snippet is supported for now."
            snippet = info["snippets"][0]
            local_idx += snippet[0]
        

        info.update({
            "global_idx": global_idx,
            "local_idx": local_idx,
            "data_file": {f'{modality}': self.file_names[modality][file_idx] for modality in list(self.data_prefix.keys())}
        })

        return info
    
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx) -> dict:
        sample = self.pipeline(self.get_info_by_global_idx(idx))
        return sample
