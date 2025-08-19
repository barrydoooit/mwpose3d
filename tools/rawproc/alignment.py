from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd



if TYPE_CHECKING:
    from .episode import Episode

class AlignTraces:
    def __init__(self,
                 episode: 'Episode',
                 use_interp_skel: bool = True,
                 skeleton_ts_type: Literal['real_ts', 'unix_ms'] = 'unix_ms', # real_ts is calculated in calibrator as start unix_ms + timestamp
                 skeleton_ts_offset_ms: int = 60, # default value for kinect-based collection
                 ): 
        self.use_interp_skel = use_interp_skel
        self.episode = episode
        self.skeleton_ts_type = skeleton_ts_type
        self.output_pcd_df = None
        self.output_skel_df = None
        self.skeleton_ts_offset_ms = skeleton_ts_offset_ms
        
    def align(self):
        if self.use_interp_skel:
            last_pcd_ts = self.interpolate_aligned_skel()
        else:
            last_pcd_ts = self.fetch_aligned_skel()
        if last_pcd_ts is not None:
            self.output_pcd_df = self.episode.clip_pcd_by(col='ts', end_seq_inclusive=last_pcd_ts - 1)
        else:
            self.output_pcd_df = self.episode.pcd_df
        if not self.check_aligned_length():
            raise ValueError(f'Aligned data length mismatch: {len(self.output_skel_df)} != {len(self.output_pcd_df)}')
        
    def fetch_aligned_skel(self):
        df_skel = self.episode.skel_df
        df_pcd = self.episode.pcd_df
        
        aligned_skels = []
        df_skel[self.skeleton_ts_type] += self.skeleton_ts_offset_ms
        unique_radar_ts = np.sort(df_pcd['ts'].unique())
        used_idx = np.full(len(df_skel), False)
        for pcd_ts in unique_radar_ts:
            available_skel_idx = np.where(~used_idx)[0]
            if len(available_skel_idx) == 0:
                self.output_skel_df = pd.DataFrame(aligned_skels).reset_index(drop=True)
                return pcd_ts
            next_available_skel = df_skel.loc[available_skel_idx].copy()
            next_available_skel['ts_diff'] = np.abs(next_available_skel[self.skeleton_ts_type] - pcd_ts).abs()
            closest_skel_idx = next_available_skel['ts_diff'].idxmin()
            used_idx[closest_skel_idx] = True
            row = df_skel.loc[closest_skel_idx].copy()
            row['pcd_ts'] = int(pcd_ts)
            aligned_skels.append(row)
        
        self.output_skel_df = pd.DataFrame(aligned_skels).reset_index(drop=True)
        return None
    
    def interpolate_aligned_skel(self):
        df_skel = self.episode.skel_df
        df_pcd = self.episode.pcd_df
        
        aligned_skels = []
        
        unique_radar_ts = np.sort(df_pcd['ts'].unique())
        skel_ts_values = df_skel[self.skeleton_ts_type].values
        
        for pcd_ts in  unique_radar_ts:
            idx = np.searchsorted(skel_ts_values, pcd_ts, side='left')
            if idx == 0:
                row = df_skel.iloc[0].copy()
                row['pcd_ts'] = round(pcd_ts)
            elif idx >= len(df_skel):
                break # NOTE: No more skel data to assign
            else:
                lower_idx = idx - 1
                upper_idx = idx
                lower_row = df_skel.iloc[lower_idx]
                upper_row = df_skel.iloc[upper_idx]
                
                t0 = lower_row[self.skeleton_ts_type]
                t1 = upper_row[self.skeleton_ts_type]
                if t0 == t1:
                    weight = 0.0
                else:
                    weight = (pcd_ts - t0) / (t1 - t0)
                interp_data = {}
                for col in df_skel.columns:
                    if col in ['real_ts', 'timestamp', 'unix_ms']:
                        continue
                    if pd.api.types.is_numeric_dtype(df_skel[col]):
                        interp_data[col] = lower_row[col] * (1 - weight) + upper_row[col] * weight
                    else:
                        interp_data[col] = lower_row[col]
                interp_data['pcd_ts'] = round(pcd_ts)
                row = pd.Series(interp_data)
            aligned_skels.append(row)
        else:
            pcd_ts = None
        self.output_skel_df = pd.DataFrame(aligned_skels).reset_index(drop=True)
        self.output_skel_df['pcd_ts'] = self.output_skel_df['pcd_ts'].astype(pd.Int64Dtype())
        return pcd_ts

    def check_aligned_length(self):
        assert self.output_skel_df is not None, 'Skeleton data not aligned.'
        assert self.output_pcd_df is not None, 'PCD data not aligned.'
        return len(self.output_skel_df) == self.output_pcd_df['ts'].nunique()

    def make_episode(self, episode_name: str) -> 'Episode':
        assert self.check_aligned_length(), 'Alligned data length mismatch.'
        length = len(self.output_skel_df)
        from .episode import Episode
        new_episode = Episode(episode_name, length)
        new_episode.skel_df = self.output_skel_df
        new_episode.pcd_df = self.output_pcd_df
        new_episode.pcd_meta = self.episode.pcd_meta
        new_episode.clean_skel_df()
        return new_episode