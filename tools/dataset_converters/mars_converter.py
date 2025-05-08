import os
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from typing import Dict, List, Optional, Tuple, Literal



def camel_to_snake(name: str) -> str:
    # Split CamelCase to snake_case
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return snake

class MarsDatasetConverter:
    def __init__(self, 
                 input_root: Path, 
                 output_root: Path,
                 use_processed_features: bool = False,
                 use_official_split: bool = False,
                 clip_size: Optional[int] = 512,
                 split_ratios: dict = dict(train=0.7, val=0.15, test=0.15)):
        self.input_root = input_root
        self.output_root = output_root
        self.out_mmwave = self.output_root / 'mmwave'
        self.out_skeleton = self.output_root / 'skeleton'
        self.out_mmwave.mkdir(parents=True, exist_ok=True)
        self.out_skeleton.mkdir(parents=True, exist_ok=True)
        
        self.use_processed_features = use_processed_features
        self.use_official_split = use_official_split or use_processed_features
        if use_official_split:
            self.clip_size = None
            self.split_ratios = None
            self.splits = ['train', 'val', 'test']
        else:
            self.clip_size = clip_size
            self.split_ratios = split_ratios
            self.splits = list(split_ratios.keys())
        self.records: Dict[str,List[dict]] = {}

    @staticmethod
    def load_radar(csv_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
        radar_df = pd.read_csv(csv_path)
        radar_df = radar_df.rename(columns={
            'Frame #': 'seq',
            'X': 'x', 'Y': 'y', 'Z': 'z',
            'Doppler': 'vel', 'Intensity': 'snr',
            'Abs Time': 'ts'
        })[['seq', 'x', 'y', 'z', 'vel', 'snr', 'ts']]
        grouped = radar_df.groupby('seq', sort=True)
        frame_ids = sorted(radar_df['seq'].unique())
        pcd_pts, pcd_idx = [], [0]
        for fid in frame_ids:
            arr = grouped.get_group(fid)[['x', 'y', 'z', 'vel', 'snr']].to_numpy(np.float32)
            pcd_pts.append(arr)
            pcd_idx.append(pcd_idx[-1] + arr.shape[0])
        pcd_data = np.concatenate(pcd_pts, axis=0)
        pcd_idx = np.array(pcd_idx, dtype=np.int64)
        pcd_cols = ['x', 'y', 'z', 'vel', 'snr']#, 'seq', 'ts']
        
        return pcd_data, pcd_idx, pcd_cols
    
    @staticmethod
    def load_skel(csv_path: str) -> Tuple[np.ndarray, list]:
        skel_df = pd.read_csv(csv_path)
        skel_cols = []
        for c in skel_df.columns:
            joint, axis = c.split('_')
            skel_cols.append(f"{camel_to_snake(joint)}_{axis.lower()}")
        skel_df.columns = skel_cols
        skel_data = skel_df.to_numpy(np.float32)
        num_frames = skel_data.shape[0]
        num_joints = len(skel_cols) // 3
        skel_data = skel_data.reshape(num_frames, 3, num_joints).transpose(0, 2, 1).reshape(num_frames, 3 * num_joints)
        return skel_data, skel_cols
    
    def load_data(self, subject_dir: Path, suffix: str = 'all'):
        pcd_data, pcd_idx, pcd_cols = MarsDatasetConverter.load_radar(subject_dir / f'radar_data_{suffix}.csv')
        skel_data, skel_cols = MarsDatasetConverter.load_skel(subject_dir / f'kinect_data_{suffix}.csv')
        return pcd_data, pcd_idx, pcd_cols, skel_data, skel_cols

    @staticmethod
    def update_info_file(dataset_h5_file: Path, info_path: Path) -> Tuple[str, dict]:
        info_all = {}
        if info_path.exists():
            with open(info_path, 'rb') as f:
                info_all = pickle.load(f)
        
        file_key = dataset_h5_file.name
        if file_key in info_all:
            print(f"File key {file_key} already exists in info file. Overwriting is not allowed.")
            return file_key, info_all[file_key]

        with h5py.File(dataset_h5_file, 'r') as h5f:
            frame_count = int(h5f['skel'].shape[0])
        info_all[file_key] = {'frame_count': frame_count}

        with open(info_path, 'wb') as f:
            pickle.dump(info_all, f)

        return file_key, info_all[file_key]
    
    def write_official_clip(self,
                            pcd_data: np.ndarray,
                            pcd_idx: np.ndarray,
                            pcd_cols: list,
                            skel_data: np.ndarray,
                            skel_cols: list,
                            subject: str,
                            split: str,):
        # write full sequence for official split
        fname = f"{subject}_{split}.h5"
        # skeleton
        skel_path = self.out_skeleton / fname
        with h5py.File(skel_path, 'w') as h5f:
            ds = h5f.create_dataset('skel', data=skel_data)
            ds.attrs['columns'] = np.array(skel_cols, dtype='S')
        # mmwave
        mmw_path = self.out_mmwave / fname
        with h5py.File(mmw_path, 'w') as h5f:
            grp = h5f.create_group('pcd')
            ds_data = grp.create_dataset('data', data=pcd_data)
            ds_data.attrs['columns'] = np.array(pcd_cols, dtype='S')
            # full data is one segment, index all counts equal to points per frame omitted
            # assume pcd_idx is already per frame idx
            grp.create_dataset('index', data=pcd_idx)
        # record
        self.records[split].append({
            'subject': subject,
            'frame_count': skel_data.shape[0],
            'mmwave_path': mmw_path.name,
            'skeleton_path': skel_path.name,
        })
        
    def write_single_clip(self, 
                    pcd_data: np.ndarray,
                    pcd_idx: np.ndarray,
                    pcd_cols: list,
                    skel_data: np.ndarray,
                    skel_cols: list,
                    subject: str,
                    start: int,
                    count: int,
                    split: str):
        fname = f"{subject}_clip{start//self.clip_size}.h5"
        skel_path = self.out_skeleton / fname
        with h5py.File(skel_path, 'w') as h5f:
            skel_slice = skel_data[start:start+count]
            ds_skel = h5f.create_dataset('skel', data=skel_slice)
            ds_skel.attrs['columns'] = np.array(skel_cols, dtype='S')
        
        mmw_path = self.out_mmwave / fname
        with h5py.File(mmw_path, 'w') as h5f:
            idx_slice = pcd_idx[start:start + count + 1]
            pts = pcd_data[idx_slice[0]: idx_slice[-1]]
            grp_pcd = h5f.create_group('pcd')
            ds_pcd = grp_pcd.create_dataset('data', data=pts)
            ds_pcd.attrs['columns'] = np.array(pcd_cols, dtype='S')

            rel_counts = np.diff(idx_slice)
            rel_idx = np.concatenate(([0], np.cumsum(rel_counts)))
            grp_pcd.create_dataset('index', data=rel_idx)
        
        self.records[split].append({
            'subject': subject,
            'frame_count': count,
            'mmwave_path': mmw_path.name,
            'skeleton_path': skel_path.name,
        })
    
    def process_single(self, subject_dir):
        subject = subject_dir.name
        if self.use_official_split:
            ts_dir = subject_dir / 'timesplit'
            for split in self.splits:
                split_fullname = dict({s: s for s in self.splits}, val='validate')[split]
                pcd_data, pcd_idx, pcd_cols, skel_data, skel_cols  = self.load_data(ts_dir, suffix=split_fullname)
                self.write_official_clip(pcd_data, pcd_idx, pcd_cols, skel_data, skel_cols, subject, split)
        else:
            pcd_data, pcd_idx, pcd_cols, skel_data, skel_cols = self.load_data(subject_dir)
            total_frames = skel_data.shape[0]
            num_clips = total_frames // self.clip_size
            rem = total_frames - num_clips * self.clip_size
            
            clip_indices = list(range(num_clips + (1 if rem > 0 else 0)))
            n_val = int(self.split_ratios['val'] * len(clip_indices))
            n_test = int(self.split_ratios['test'] * len(clip_indices))
            
            perm = np.random.permutation(clip_indices)
            val_set = set(perm[:n_val])
            test_set = set(perm[n_val:n_val + n_test])
            
            for i in clip_indices:
                start = i * self.clip_size
                count = self.clip_size if i < num_clips else rem
                if i in val_set:
                    split = 'val'
                elif i in test_set:
                    split = 'test'
                else:
                    split = 'train'
                self.write_single_clip(pcd_data, pcd_idx, pcd_cols, skel_data, skel_cols, subject, start, count, split)
    
    def process_features(self):
        for split in self.splits:
            split_fullname = dict({s: s for s in self.splits}, val='validate')[split]
            featuremap = np.load(self.input_root / f'featuremap_{split_fullname}.npy')
            N, H, W, D = featuremap.shape
            featuremap = featuremap.reshape(N, H * W, D).reshape(N * H * W, D).astype(np.float32)
            
            frame_idx = np.arange(0, (N + 1) * H * W, H * W, dtype=np.int64)
            mmw_path = self.out_mmwave / f"{split}.h5"
            mmw_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(mmw_path, 'w') as h5f:
                grp = h5f.create_group('pcd')
                ds = grp.create_dataset('data', data=featuremap)
                ds.attrs['columns'] = np.array(['x', 'y', 'z', 'vel', 'snr'], dtype='S')
                grp.create_dataset('index', data=frame_idx)
            
            skel_data = np.load(self.input_root / f'labels_{split_fullname}.npy')
            N, D = skel_data.shape
            skel_data = skel_data.reshape(N, 3, -1).transpose(0, 2, 1).reshape(N, -1)
            skel_data_padded = np.zeros((N, 3 * 21), dtype=np.float32)
            orig_ptr = 0
            for j in range(21):
                if j in [7, 11]:
                    continue
                skel_data_padded[:, 3 * j:3 * j + 3] = skel_data[:, orig_ptr*3:orig_ptr*3 + 3]
                orig_ptr += 1
            cols = []
            for axis in ('x', 'y', 'z'):
                for j in range(21):
                    cols.append(f"joint{j}_{axis}")
            
            skel_path = self.out_skeleton / f"{split}.h5"
            skel_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(skel_path, 'w') as h5f:
                ds = h5f.create_dataset('skel', data=skel_data_padded)
                ds.attrs['columns'] = np.array(cols, dtype='S')
        
            self.records[split].append({
                'frame_count': skel_data_padded.shape[0],
                'mmwave_path': mmw_path.name,
                'skeleton_path': skel_path.name,
            })
            
    def process_all(self):
        self.records = {split: [] for split in self.splits}
        if self.use_processed_features:
            self.process_features()
        else:
            for subject_dir in sorted(p for p in self.input_root.iterdir() if p.is_dir()):
                if not subject_dir.name.startswith('subject'):
                    continue
                self.process_single(subject_dir)
        
        for split, rec in self.records.items():
            with open(self.output_root / f"info_{split}.pkl", 'wb') as f:
                pickle.dump(rec, f)