from pathlib import Path
import pickle
import uuid

import h5py
import numpy as np
import pandas as pd
from .episode import Episode


class ToHdf5:
    def __init__(self,
                 alligned_episode: 'Episode',
                 output_dir: Path,
                 pointcloud_subdir: str = 'default',
                 mmwave_path_as_dict: bool = True,
                 missing_frame_strategy: str = 'duplicate'):
        self.alligned_episode = alligned_episode
        self.output_dir = output_dir
        self.mmwave_path_as_dict = bool(mmwave_path_as_dict)
        if missing_frame_strategy not in {'duplicate', 'remove'}:
            raise ValueError(
                "missing_frame_strategy must be one of {'duplicate', 'remove'}."
            )
        self.missing_frame_strategy = missing_frame_strategy
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

    def _prepare_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        pcd_df = self.alligned_episode.pcd_df.copy()
        skel_df = self.alligned_episode.skel_df.copy()

        frame_ids = pcd_df['seq'].unique()
        if len(frame_ids) == 0:
            raise ValueError('PCD data is empty.')

        contiguous = (np.diff(frame_ids) == 1).all() and frame_ids[0] == 0
        if contiguous:
            return pcd_df, skel_df

        if self.missing_frame_strategy == 'remove':
            return self._remove_missing_frames_from_sequence(pcd_df, skel_df)
        return self._duplicate_previous_frame_for_missing_sequences(pcd_df, skel_df)

    def _remove_missing_frames_from_sequence(
        self,
        pcd_df: pd.DataFrame,
        skel_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame_ids = pcd_df['seq'].unique()
        seq_map = {old_seq: new_seq for new_seq, old_seq in enumerate(frame_ids)}
        pcd_df.loc[:, 'seq'] = pcd_df['seq'].map(seq_map).astype(np.int64)
        return pcd_df, skel_df

    def _duplicate_previous_frame_for_missing_sequences(
        self,
        pcd_df: pd.DataFrame,
        skel_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame_ids = pcd_df['seq'].unique()
        grouped = {int(fid): group.copy() for fid, group in pcd_df.groupby('seq', sort=False)}
        expected_frame_ids = np.arange(0, int(frame_ids[-1]) + 1, dtype=np.int64)

        seq_to_position = {int(fid): idx for idx, fid in enumerate(frame_ids)}
        can_expand_skeleton = len(skel_df) == len(frame_ids)
        if can_expand_skeleton:
            skel_rows = [row.copy() for _, row in skel_df.iterrows()]
            repaired_skel_rows = []

        repaired_groups = []
        last_group = None
        last_skel_row = None

        for fid in expected_frame_ids:
            fid = int(fid)
            group = grouped.get(fid)
            if group is not None:
                repaired_groups.append(group)
                last_group = group
                if can_expand_skeleton:
                    last_skel_row = skel_rows[seq_to_position[fid]]
                    repaired_skel_rows.append(last_skel_row.copy())
                continue

            if last_group is None:
                next_existing_seq = int(frame_ids[0])
                template_group = grouped[next_existing_seq].copy()
                template_skel_row = skel_rows[0].copy() if can_expand_skeleton else None
            else:
                template_group = last_group.copy()
                template_skel_row = last_skel_row.copy() if can_expand_skeleton else None

            duplicated_group = template_group.copy()
            duplicated_group.loc[:, 'seq'] = fid
            duplicated_group.loc[:, 'ts'] = self._estimate_missing_timestamp(
                frame_ids=frame_ids,
                pcd_df=pcd_df,
                missing_seq=fid,
            )
            repaired_groups.append(duplicated_group)

            if can_expand_skeleton and template_skel_row is not None:
                repaired_skel_rows.append(template_skel_row.copy())

        repaired_pcd_df = pd.concat(repaired_groups, ignore_index=True)
        repaired_pcd_df.loc[:, 'seq'] = repaired_pcd_df['seq'].astype(np.int64)
        repaired_pcd_df.loc[:, 'ts'] = repaired_pcd_df['ts'].astype(np.int64)

        if can_expand_skeleton:
            repaired_skel_df = pd.DataFrame(repaired_skel_rows).reset_index(drop=True)
        else:
            repaired_skel_df = skel_df
        return repaired_pcd_df, repaired_skel_df

    @staticmethod
    def _estimate_missing_timestamp(
        frame_ids: np.ndarray,
        pcd_df: pd.DataFrame,
        missing_seq: int,
    ) -> np.int64:
        seq_to_ts = (
            pcd_df[['seq', 'ts']]
            .drop_duplicates(subset='seq')
            .set_index('seq')['ts']
            .astype(np.int64)
        )
        previous_ids = frame_ids[frame_ids < missing_seq]
        next_ids = frame_ids[frame_ids > missing_seq]

        if len(previous_ids) and len(next_ids):
            prev_seq = int(previous_ids[-1])
            next_seq = int(next_ids[0])
            prev_ts = int(seq_to_ts.loc[prev_seq])
            next_ts = int(seq_to_ts.loc[next_seq])
            gap = next_seq - prev_seq
            if gap > 0:
                step = (next_ts - prev_ts) / gap
                return np.int64(round(prev_ts + step * (missing_seq - prev_seq)))

        if len(previous_ids):
            prev_seq = int(previous_ids[-1])
            prev_ts = int(seq_to_ts.loc[prev_seq])
            if len(previous_ids) >= 2:
                prev_prev_seq = int(previous_ids[-2])
                prev_prev_ts = int(seq_to_ts.loc[prev_prev_seq])
                step = (prev_ts - prev_prev_ts) / max(prev_seq - prev_prev_seq, 1)
            else:
                step = 1
            return np.int64(round(prev_ts + step * (missing_seq - prev_seq)))

        next_seq = int(next_ids[0])
        next_ts = int(seq_to_ts.loc[next_seq])
        if len(next_ids) >= 2:
            next_next_seq = int(next_ids[1])
            next_next_ts = int(seq_to_ts.loc[next_next_seq])
            step = (next_next_ts - next_ts) / max(next_next_seq - next_seq, 1)
        else:
            step = 1
        return np.int64(round(next_ts - step * (next_seq - missing_seq)))
    
    def save(self, info_suffices=['all']):
        self.check_na()
        pcd_df, skel_df = self._prepare_dataframes()
        frame_ids = pcd_df['seq'].unique()
        
        mmwave_dir = self.output_dir / 'mmwave' / 'pointcloud'
        if self.pointcloud_subdir:
            mmwave_dir = mmwave_dir / self.pointcloud_subdir
        skeleton_dir = self.output_dir / 'skeleton'
        mmwave_dir.mkdir(parents=True, exist_ok=True)
        skeleton_dir.mkdir(parents=True, exist_ok=True)
        out_mmwave = mmwave_dir / f'{self.alligned_episode.episode_name}.h5'
        out_skeleton = skeleton_dir / f'{self.alligned_episode.episode_name}.h5'
        
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
        
        skel_all = skel_df.to_numpy(np.float64)
        
        with h5py.File(out_mmwave, 'w') as h5file:
            grp_pcd = h5file.create_group('pcd')
            ds_data = grp_pcd.create_dataset('data', data=pcd_all)
            ds_index = grp_pcd.create_dataset('index', data=frame_indices)
            ds_data.attrs['columns'] = np.array(pcd_df.columns, dtype='S')
        
        with h5py.File(out_skeleton, 'w') as h5file:
            ds_skel = h5file.create_dataset('skel', data=skel_all)
            ds_skel.attrs['columns'] = np.array(skel_df.columns, dtype='S')

        file_key, file_info = self.update_info_file(
            self.output_dir / f'info_all.pkl',
            frame_count=int(len(frame_ids)),
        )
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
        
    def update_info_file(self, info_pkl_path: Path, frame_count: int) -> dict:
        info_all = []
        if info_pkl_path.exists():
            with open(info_pkl_path, 'rb') as f:
                info_all = pickle.load(f)
        
        file_key = self.alligned_episode.episode_name
        if frame_count != self.alligned_episode.episode_length:
            print(
                f"Warning: frame count changed during export for {file_key}: "
                f"{self.alligned_episode.episode_length} -> {frame_count} "
                f"(strategy={self.missing_frame_strategy})"
            )
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
