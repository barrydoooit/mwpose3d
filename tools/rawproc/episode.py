import json
from pathlib import Path
from typing import Any, Literal, Optional, Union
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from mwpose3d.utils.pointcloud_toolkits.structures import SimplePoint5D
from tools.rawproc.alignment import AlignTraces
from tools.rawproc.time_calib_manual import CalibrateTimeWindow
from mwpose3d.utils.kinect_toolkits.kinectData import Skeleton

from mwcore.radario.readers.offlineReaders.raw_bin_reader import RawBinReader
from . import load_utils

class Episode:
    def __init__(self,
                 episode_name: 'str',
                 episode_length: Optional[int] = None):
        self.episode_name = episode_name
        self.episode_length = episode_length
        
        self._pcd_df: Optional[pd.DataFrame] = None
        self._pcd_meta: Optional[dict] = None
        self._skel_df: Optional[pd.DataFrame] = None
    
    def get_pcd_frame_by(self, col: str, value: Any, encapsulate: bool = False):
        df =  self.pcd_df[self.pcd_df[col] == value]
        if not encapsulate:
            return df
        pcd_array = np.asarray(df[['x', 'y', 'z', 'vel', 'snr']])
        return [SimplePoint5D.from_numpy(point) for point in pcd_array]
    
    def get_skel_frame_by(self, col: str, value: Any, encapsulate: bool = False):
        df = self.skel_df[self.skel_df[col] == value]
        if not encapsulate:
            return df
        return Skeleton.from_dataframe(df.iloc[0])
    
    def convert_skel_timestamp_to_unix(self, start_unix_ts: Union[float, int] = None):
        if start_unix_ts is None:
            start_unix_ts = self.skel_df['unix_ms'].iloc[0]
        self.skel_df['timestamp'] = start_unix_ts  + (self.skel_df['timestamp'] - self.skel_df['timestamp'].iloc[0])
        self.SKEL_TS_BASE = start_unix_ts
        logger.warning(msg=f'Skeleton df of {self.episode_name} has changed timestamp to unix timestamp.')
    
    def get_episode_min_max_ts(self):
        if not hasattr(self, 'SKEL_TS_BASE'):
            self.convert_skel_timestamp_to_unix()
        return min(self.skel_df['timestamp'].iloc[0], self.pcd_df['ts'].iloc[0]), max(self.skel_df['timestamp'].iloc[-1], self.pcd_df['ts'].iloc[-1])
        
    def load_pcd(self, raw_data_dir: Path, allow_missing: bool = False):
        raw_data_file = raw_data_dir / f'{self.episode_name}.json'
        if not raw_data_file.exists() and not allow_missing:
            raise FileNotFoundError(f'{raw_data_file} does not exist.')
        if not raw_data_file.exists():
            return None
        self.pcd_df = load_utils.load_radar_schema_json_to_df(raw_data_file)

    def load_pcd_bin(self, bin_path: Path):
        if not bin_path.exists():
            raise FileNotFoundError(f'{bin_path} does not exist.')
        
        reader = RawBinReader(str(bin_path), has_timestamp=True)
        
        rows = []
        try:
            while True:
                data = reader.read()
                if data is None:
                    break
                
                ts, pcd = data
                assert pcd.shape[1] == 6, f'pcd shape is {pcd.shape}, expected (6, N)'

                pcd_t = pcd.T
                for point in pcd_t:
                    rows.append({
                        'seq': reader.current_frame_idx, # 0 ind
                        'ts': int(ts * 1000), 
                            'x': point[0],
                            'y': point[1],
                            'z': point[2],
                            'vel': point[3],
                            'snr': point[4],
                        })
        finally:
            reader.close()
            
        if rows:
            self.pcd_df = pd.DataFrame(rows)
            self.pcd_df['ts'] = self.pcd_df['ts'].astype(pd.Int64Dtype())
        else:
             # Create empty DF with correct columns
            self.pcd_df = pd.DataFrame(columns=['seq', 'ts', 'x', 'y', 'z', 'vel', 'snr'])

         
    def load_pcd_meta(self, meta_data_dir: Path, allow_missing: bool = False):
        meta_data_file = meta_data_dir / f'{self.episode_name}.json'
        if not meta_data_file.exists() and not allow_missing:
            if meta_data_file.with_suffix('.meta.json').exists():
                meta_data_file = meta_data_file.with_suffix('.meta.json')
            else:
                raise FileNotFoundError(f'{meta_data_file} does not exist.')
        with open(meta_data_file, 'r') as f:
            self.pcd_meta = json.load(f)
        if 'calib_frames' in self.pcd_meta:
            calib_frames = self.pcd_meta['calib_frames']
            self.episode_length = int(''.join(filter(str.isdigit, self.episode_name.split('_')[-1]))) - calib_frames
        else:
            self.episode_length = int(''.join(filter(str.isdigit, self.episode_name.split('_')[-1])))
        
    def load_skeleton(self, raw_data_dir: Path, allow_missing: bool = False):
        raw_data_file = raw_data_dir / f'{self.episode_name}.csv'
        if not raw_data_file.exists() and not allow_missing:
            raise FileNotFoundError(f'{raw_data_file} does not exist.')
        if not raw_data_file.exists():
            return None
        self.skel_df = load_utils.load_framed_skeleton_csv_to_df(raw_data_file)
        
    def calibrate_time(self, manual: bool = False):
        if not manual:
            raise NotImplementedError("Automatic time calibration is removed.")
        else:
            calibrator_gui = CalibrateTimeWindow(self)
            calibrator_gui.wait_window()
            return calibrator_gui.result_offset_ms
            
    def align_traces(self,
                     use_interp_skel: bool = True,
                     skeleton_ts_type: Literal['real_ts', 'unix_ms'] = 'unix_ms',
                     skeleton_ts_offset_ms: int = 60):
        aligner = AlignTraces(
            self,
            use_interp_skel=use_interp_skel,
            skeleton_ts_type=skeleton_ts_type,
            skeleton_ts_offset_ms=skeleton_ts_offset_ms,
        )
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
