from pathlib import Path
import pickle
import uuid

import h5py
import numpy as np
from .episode import Episode


class ToHdf5:
    def __init__(self,
                 alligned_episode: 'Episode',
                 output_dir: Path,
                 pointcloud_subdir: str = 'default',
                 mmwave_path_as_dict: bool = True):
        self.alligned_episode = alligned_episode
        self.output_dir = output_dir
        self.mmwave_path_as_dict = bool(mmwave_path_as_dict)
        subdir = (pointcloud_subdir or '').strip()
        if self.mmwave_path_as_dict:
            self.pointcloud_subdir = subdir or 'default'
        else:
            # Flat mode: save directly under mmwave/pointcloud.
            self.pointcloud_subdir = subdir

    def _mmwave_rel_path(self, file_key: str) -> str:
        if self.pointcloud_subdir:
            return f'pointcloud/{self.pointcloud_subdir}/{file_key}.h5'
        return f'pointcloud/{file_key}.h5'

    @staticmethod
    def _extract_mmwave_pointcloud_map(mmwave_value) -> dict:
        if isinstance(mmwave_value, str):
            parts = Path(mmwave_value).parts
            if 'pointcloud' in parts:
                i = parts.index('pointcloud')
                if len(parts) >= i + 3:
                    variant = parts[i + 1]
                    return {variant: mmwave_value}
            return {'default': mmwave_value}
        if not isinstance(mmwave_value, dict):
            return {}
        pointcloud = mmwave_value.get('pointcloud', None)
        if isinstance(pointcloud, dict):
            return dict(pointcloud)
        return {}

    def check_na(self):
        pcd_nok, skel_nok = self.alligned_episode.check_na()
        assert not pcd_nok, 'PCD data contains NA values.'
        assert not skel_nok, 'Skeleton data contains NA values.'
    
    def save(self, info_suffices=['all']):
        self.check_na()
        
        # Check if 'seq' column is ascending from 0
        frame_ids = self.alligned_episode.pcd_df['seq'].unique()
        assert (np.diff(frame_ids) == 1).all() and frame_ids[0] == 0, ValueError("'seq' column is not ascending from 0.")
        
        mmwave_dir = self.output_dir / 'mmwave' / 'pointcloud'
        if self.pointcloud_subdir:
            mmwave_dir = mmwave_dir / self.pointcloud_subdir
        skeleton_dir = self.output_dir / 'skeleton'
        mmwave_dir.mkdir(parents=True, exist_ok=True)
        skeleton_dir.mkdir(parents=True, exist_ok=True)
        out_mmwave = mmwave_dir / f'{self.alligned_episode.episode_name}.h5'
        out_skeleton = skeleton_dir / f'{self.alligned_episode.episode_name}.h5'
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
        
        with h5py.File(out_mmwave, 'w') as h5file:
            grp_pcd = h5file.create_group('pcd')
            ds_data = grp_pcd.create_dataset('data', data=pcd_all)
            ds_index = grp_pcd.create_dataset('index', data=frame_indices)
            ds_data.attrs['columns'] = np.array(pcd_df.columns, dtype='S')
        
        with h5py.File(out_skeleton, 'w') as h5file:
            ds_skel = h5file.create_dataset('skel', data=skel_all)
            ds_skel.attrs['columns'] = np.array(skel_df.columns, dtype='S')

        file_key, file_info = self.update_info_file(self.output_dir / f'info_all.pkl')
        for suffix in info_suffices:
            if suffix == 'all':
                continue
            info_path = self.output_dir / f'info_{suffix}.pkl'
            if info_path.exists():
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
            else:
                infos = []
            with open(self.output_dir / f'info_{suffix}.pkl', 'wb') as f:
                pickle.dump(infos + [file_info], f)
        
    def update_info_file(self, info_pkl_path: Path) -> dict:
        info_all = []
        if info_pkl_path.exists():
            with open(info_pkl_path, 'rb') as f:
                info_all = pickle.load(f)
        
        file_key = self.alligned_episode.episode_name
        frame_count = self.alligned_episode.pcd_df['seq'].nunique()
        assert frame_count == self.alligned_episode.episode_length, f"Frame count mismatch: {frame_count} vs {self.alligned_episode.episode_length}"
        meta = self.alligned_episode.pcd_meta
        if meta is None: meta = {}
        new_info_entry = dict(
            meta,
            frame_count=frame_count,
            id=file_key,
            skeleton_path=f'{file_key}.h5',
        )
        old_entry = None
        for info in info_all:
            if info['id'] == file_key:
                old_entry = info
                break

        mmwave_rel_path = self._mmwave_rel_path(file_key)
        if self.mmwave_path_as_dict:
            old_mmwave = None
            if old_entry is not None:
                old_mmwave = old_entry.get('mmwave_path', old_entry.get('mmwave', None))
            pointcloud_map = self._extract_mmwave_pointcloud_map(old_mmwave)
            variant = self.pointcloud_subdir or 'default'
            pointcloud_map[variant] = mmwave_rel_path
            mmwave_value = {'pointcloud': pointcloud_map}
            new_info_entry['mmwave_path'] = mmwave_value
            new_info_entry.pop('mmwave', None)
        else:
            new_info_entry['mmwave_path'] = mmwave_rel_path
            new_info_entry.pop('mmwave', None)
        meta_data = self.alligned_episode.pcd_meta
        if meta_data is not None:
            new_info_entry = dict(new_info_entry, **meta_data)
        for info in info_all:
            if info['id'] == file_key:
                info.clear()
                info.update(new_info_entry)
                break
        else:
            info_all.append(new_info_entry)
                
        with open(info_pkl_path, 'wb') as f:
            pickle.dump(info_all, f)
        return file_key,  new_info_entry
