import json
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

from dl_engine.tools.rawproc.alignment import AlignTraces
from dl_engine.tools.rawproc.time_calib import TimeCalibrator

from . import load_utils

class Episode:
    def __init__(self,
                 episode_name: int,
                 episode_length: int):
        self.episode_name = episode_name
        self.episode_length = episode_length
        
        self._pcd_df: Optional[pd.DataFrame] = None
        self._pcd_meta: Optional[dict] = None
        self._skel_df: Optional[pd.DataFrame] = None
        
    def load_pcd(self, raw_data_dir: Path, allow_missing: bool = False):
        raw_data_file = raw_data_dir / f'{self.episode_name}.json'
        if not raw_data_file.exists() and not allow_missing:
            raise FileNotFoundError(f'{raw_data_file} does not exist.')
        if not raw_data_file.exists():
            return None
        self.pcd_df = load_utils.load_radar_schema_json_to_df(raw_data_file)
         
    def load_pcd_meta(self, meta_data_dir: Path, allow_missing: bool = False):
        meta_data_file = meta_data_dir / f'{self.episode_name}.json'
        if not meta_data_file.exists() and not allow_missing:
            raise FileNotFoundError(f'{meta_data_file} does not exist.')
        if not meta_data_file.exists():
            return None
        with open(meta_data_file, 'r') as f:
            self.pcd_meta = json.load(f)
    
    def load_skeleton(self, raw_data_dir: Path, allow_missing: bool = False):
        raw_data_file = raw_data_dir / f'{self.episode_name}.csv'
        if not raw_data_file.exists() and not allow_missing:
            raise FileNotFoundError(f'{raw_data_file} does not exist.')
        if not raw_data_file.exists():
            return None
        self.skel_df = load_utils.load_raw_skeleton_csv_to_df(raw_data_file)
        
    def calibrate_time(self):
        calibrator = TimeCalibrator(self)
        calibrator.calib()
        return calibrator.make_episode()
    
    def align_traces(self, use_interp_skel: bool = True):
        aligner = AlignTraces(self, use_interp_skel)
        aligner.align()
        return aligner.make_episode(self.episode_name)
    
    def clip_skel_by_ts(self, 
                  start_ts_inclusive: Optional[Union[int, float]] = None,
                  end_ts_inclusive: Optional[Union[int, float]] = None,
                  ts_col: Literal['timestamp', 'unix_ms'] = 'unix_ms'
                  ):
        if start_ts_inclusive is None and end_ts_inclusive is None:
            return
        if start_ts_inclusive is None:
            start_ts_inclusive = self.skel_df[ts_col].iloc[0]
        if end_ts_inclusive is None:
            end_ts_inclusive = self.skel_df[ts_col].iloc[-1]
        new_kel_df = self.skel_df[(self.skel_df[ts_col] >= start_ts_inclusive) & (self.skel_df[ts_col] <= end_ts_inclusive)].copy()
        return new_kel_df
    
    def clip_pcd_by(self, 
                        col: str,
                        start_seq_inclusive: Optional[int] = None,
                        end_seq_inclusive: Optional[int] = None):
        if start_seq_inclusive is None and end_seq_inclusive is None:
            return
        if start_seq_inclusive is None:
            start_seq_inclusive = self.pcd_df[col].iloc[0]
        if end_seq_inclusive is None:
            end_seq_inclusive = self.pcd_df[col].iloc[-1]
        new_pcd_df = self.pcd_df[(self.pcd_df[col] >= start_seq_inclusive) & (self.pcd_df[col] <= end_seq_inclusive)]
        return new_pcd_df
    
    def clean_skel_df(self):
        if self.skel_df is not None:
            self.skel_df.drop(columns=['timestamp', 'unix_ms', 'real_ts'], errors='ignore', inplace=True)

    def check_na(self):
        return self.pcd_df.isna().values.any(), self.skel_df.isna().values.any()
    ###############
        
    @property
    def pcd_df(self):
        return self._pcd_df
    
    @pcd_df.setter
    def pcd_df(self, pcd_df):
        assert self.pcd_df is None, 'pcd data already loaded.'
        self._pcd_df = pcd_df
    
    @property
    def pcd_meta(self):
        return self._pcd_meta
    
    @pcd_meta.setter
    def pcd_meta(self, pcd_meta):
        assert self.pcd_meta is None, 'pcd meta data already loaded.'
        self._pcd_meta = pcd_meta
    
    @property
    def skel_df(self):
        return self._skel_df
    
    @skel_df.setter
    def skel_df(self, skel_df):
        assert self.skel_df is None, 'skeleton data already loaded.'
        self._skel_df = skel_df