import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import h5py
from tqdm import tqdm



class MilipointDatasetConverter:
    _kp18_names = [
        'NOSE', 'NECK', 'RIGHT_SHOULDER', 'RIGHT_ELBOW',
        'RIGHT_WRIST', 'LEFT_SHOULDER', 'LEFT_ELBOW',
        'LEFT_WRIST', 'RIGHT_HIP', 'RIGHT_KNEE',
        'RIGHT_ANKLE', 'LEFT_HIP', 'LEFT_KNEE',
        'LEFT_ANKLE', 'RIGHT_EYE', 'LEFT_EYE',
        'RIGHT_EAR', 'LEFT_EAR'
    ]
    # We want 9 keypoints (same order as MMRKeypointData.kp9_names + head = mean of head points)
    _kp9_names = [
        'RIGHT_SHOULDER', 'RIGHT_ELBOW',
        'LEFT_SHOULDER',  'LEFT_ELBOW',
        'RIGHT_HIP',      'RIGHT_KNEE',
        'LEFT_HIP',       'LEFT_KNEE',
        'HEAD'
    ]
    _head_names = ['NOSE', 'RIGHT_EYE', 'LEFT_EYE', 'RIGHT_EAR', 'LEFT_EAR']

    def __init__(self,
                 input_root: Union[str, Path],
                 output_root: Union[str, Path],
                 seed: int = 20,
                 partitions = {'train': 0.8, 'val': 0.1, 'test': 0.1},
                 num_keypoints: int = 18):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)

        self.out_mmwave = self.output_root / 'mmwave'
        self.out_skeleton = self.output_root / 'skeleton'
        self.out_mmwave.mkdir(parents=True, exist_ok=True)
        self.out_skeleton.mkdir(parents=True, exist_ok=True)

        total = sum(partitions.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'Split ratios must sum to 1, got {total}')
        self.partitions = partitions

        assert num_keypoints in [9, 18], "num_keypoints must be either 9 or 18"
        self.num_keypoints = num_keypoints

        self.seed = seed
        random.seed(seed)

        self.records: Dict[str, List[dict]] = {'train': [], 'val': [], 'test': []}

    def _collect_raw_pkls(self) -> List[Path]:
        """
        Return a sorted list of all the .pkl files directly under input_root,
        excluding any non-.pkl (e.g. action_label.py).
        """
        all_pkls = [p for p in sorted(self.input_root.iterdir()) if p.suffix == ".pkl"]
        if not all_pkls:
            raise FileNotFoundError(f"No .pkl files found in {self.input_root}")
        return all_pkls

    def _shuffle_and_split_files(self, pkls: List[Path]) -> Dict[str, List[Path]]:
        random.shuffle(pkls)
        n = len(pkls)
        n_train = int(n * self.partitions['train'])
        n_val = int(n * self.partitions['val'])
        train_files = pkls[:n_train]
        val_files = pkls[n_train:n_train + n_val]
        test_files = pkls[n_train + n_val:]
        return {'train': train_files, 'val': val_files, 'test': test_files}
    
    def _transform_keypoints(self, data_list: List[dict]) -> List[dict]:
        if self.num_keypoints == 18:
            return data_list
        
        kp9_idx = [self._kp18_names.index(kp) for kp in self._kp9_names]
        head_idx = [self._kp18_names.index(kp) for kp in self._head_names]

        for data in data_list:
            kpts18 = data['y']
            kpts8 = kpts18[kp9_idx, :]
            head_pts = kpts18[head_idx, :]
            head_mean = head_pts.mean(axis=0).reshape(1, 3)
            kpts9 = np.concatenate([kpts8, head_mean], axis=0)
            data['y'] = kpts9
        return data_list
    
    def _shuffle_and_split(self, data_list: List[dict]) -> Dict[str, List[dict]]:
        random.shuffle(data_list)
        total = len(data_list)
        train_end = int(total * self.partitions['train'])
        val_end = train_end + int(total * self.partitions['val'])

        return {
            'train': data_list[:train_end],
            'val': data_list[train_end:val_end],
            'test': data_list[val_end:]
        }
    
    def _build_pcd(self,
                   pcd_frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a list of point-cloud arrays (one per frame), build:
          - pcd_data: (total_points, 3) concatenated across frames.
          - pcd_idx:  (num_frames + 1,) array of cumulative start indices.
        """
        idx_list = [0]
        pts_accum: List[np.ndarray] = []
        for pts in pcd_frames:
            pts32 = pts.astype(np.float32)
            pts_accum.append(pts32)
            idx_list.append(idx_list[-1] + pts32.shape[0])
        if pts_accum:
            pcd_data = np.concatenate(pts_accum, axis=0)
        else:
            pcd_data = np.zeros((0, 3), dtype=np.float32)
        pcd_idx = np.array(idx_list, dtype=np.int64)
        return pcd_data, pcd_idx
    
    def _make_skel_columns(self) -> List[bytes]:
        cols: List[str] = []
        for j in range(self.num_keypoints):
            for axis in ('x', 'y', 'z'):
                cols.append(f"kp{j}_{axis}")
        return [c.encode('utf-8') for c in cols]
    
    def _build_skel(self,
                    skel_frames: List[np.ndarray]) -> Tuple[np.ndarray, List[bytes]]:
        """
        Given a list of skeleton arrays (one per frame), each (num_keypoints, 3),
        stack them into (num_frames, num_keypoints*3). Also return columns.
        """
        if not skel_frames:
            return np.zeros((0, 0), dtype=np.float32), []
        
        skel_flat: List[np.ndarray] = []
        for joints in skel_frames:
            flat = joints.astype(np.float32).reshape(-1)
            skel_flat.append(flat)
        skel_data = np.stack(skel_flat, axis=0)
        skel_cols = self._make_skel_columns()
        return skel_data, skel_cols

    def _write_sequence_h5(self,
                         file_stem: str,
                         frames_x: List[np.ndarray],
                         frames_y: List[np.ndarray]) -> Tuple[str, str]:
        fname_pcd = f"{file_stem}.h5"
        fname_skel = f"{file_stem}.h5"

        pcd_data, pcd_idx = self._build_pcd(frames_x)
        mmw_path = self.out_mmwave / fname_pcd
        with h5py.File(mmw_path, 'w') as h5f:
            grp = h5f.create_group('pcd')
            ds = grp.create_dataset('data', data=pcd_data)
            ds.attrs['columns'] = np.array(['x', 'y', 'z'], dtype='S')
            grp.create_dataset('index', data=pcd_idx)
        
        skel_data, skel_cols = self._build_skel(frames_y)
        skel_path = self.out_skeleton / fname_skel
        with h5py.File(skel_path, 'w') as h5f:
            ds_skel = h5f.create_dataset('skel', data=skel_data)
            ds_skel.attrs['columns'] = np.array(skel_cols, dtype='S')
        
        return fname_pcd, fname_skel
    
    def process_split(self, split: str, pkl_list: List[Path]) -> None:
        progress_bar = tqdm(pkl_list, desc=f"Processing {split} split", unit="file")
        for pkl_path in progress_bar:
            file_stem = pkl_path.stem
            progress_bar.set_description(f"Processing {split} - {file_stem}")
            with open(pkl_path, 'rb') as f:
                data_list = pickle.load(f)
            data_list = self._transform_keypoints(data_list)

            frames_x = [data['x'] for data in data_list]
            frames_y = [data['y'] for data in data_list]

            fname_pcd, fname_skel = self._write_sequence_h5(file_stem, frames_x, frames_y)

            rec = {
                'sequence_id': file_stem,
                'frame_count': len(frames_x),
                'mmwave_path': fname_pcd,
                'skeleton_path': fname_skel,
            }
            self.records[split].append(rec)
    
    def process_all(self) -> None:
        all_pkls = self._collect_raw_pkls()
        split_files = self._shuffle_and_split_files(all_pkls)
        for split in self.partitions.keys():
            self.process_split(split, split_files[split])

            info_path = self.output_root / f'info_{split}.pkl'
            with open(info_path, "wb") as f:
                pickle.dump(self.records[split], f)
        
