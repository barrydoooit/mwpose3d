from pathlib import Path
import pickle
import uuid

import h5py
import numpy as np
from .episode import Episode


class ToHdf5:
    def __init__(self,
                 alligned_episode: 'Episode',
                 output_dir: Path):
        self.alligned_episode = alligned_episode
        self.output_dir = output_dir

    def check_na(self):
        pcd_nok, skel_nok = self.alligned_episode.check_na()
        assert not pcd_nok, 'PCD data contains NA values.'
        assert not skel_nok, 'Skeleton data contains NA values.'
    
    def save(self):
        self.check_na()
        
        # Check if 'seq' column is ascending from 0
        frame_ids = self.alligned_episode.pcd_df['seq'].unique()
        assert (np.diff(frame_ids) == 1).all() and frame_ids[0] == 0, ValueError("'seq' column is not ascending from 0.")

        (self.output_dir / 'h5').mkdir(parents=True, exist_ok=True)
        hdf5_path = self.output_dir / 'h5' / f'{self.alligned_episode.episode_name}.h5'
        pcd_df = self.alligned_episode.pcd_df
        
        all_points = []
        frame_indices = [0]
        grouped = pcd_df.groupby('seq', sort=True)
        for fid in frame_ids:
            group = grouped.get_group(fid)
            group_values = group.to_numpy().astype(np.float64)
            all_points.append(group_values)
            frame_indices.append(frame_indices[-1] + group_values.shape[0])
        pcd_all = np.concatenate(all_points, axis=0)
        frame_indices = np.array(frame_indices)
        
        
        skel_df = self.alligned_episode.skel_df
        skel_all = skel_df.to_numpy(np.float64)
        
        with h5py.File(hdf5_path, 'w') as h5file:
            grp_pcd = h5file.create_group('pcd')
            ds_data = grp_pcd.create_dataset('data', data=pcd_all)
            ds_index = grp_pcd.create_dataset('index', data=frame_indices)
            ds_data.attrs['columns'] = np.array(pcd_df.columns, dtype='S')
            ds_skel = h5file.create_dataset('skel', data=skel_all)
            ds_skel.attrs['columns'] = np.array(skel_df.columns, dtype='S')
        self.update_info_file(hdf5_path, self.output_dir / 'info_all.pkl')
        
    def update_info_file(self, dataset_h5_file: Path, info_pkl_path: Path) -> dict:
        info_all = {}
        if info_pkl_path.exists():
            with open(info_pkl_path, 'rb') as f:
                info_all = pickle.load(f)
        
        file_key = dataset_h5_file.name
        if file_key in info_all:
            print(f"File key {file_key} already exists in info file. Overwriting is not allowed for now.")
            return
        existing_tokens = {info['token'] for info in info_all.values() if 'token' in info}
        new_token = uuid.uuid4().hex
        while new_token in existing_tokens:
            new_token = uuid.uuid4().hex
        info_all[file_key] = {'token': new_token}
        
        with h5py.File(dataset_h5_file, 'r') as h5file:
            frame_count = h5file['skel'].shape[0]
        
        info_all[file_key]['frame_count'] = frame_count
        with open(info_pkl_path, 'wb') as f:
            pickle.dump(info_all, f)