from typing import List, Literal, TYPE_CHECKING, Optional, Tuple
import pandas as pd

import mwpose3d.utils.kinect_toolkits as kntk

if TYPE_CHECKING:
    from .episode import Episode

def dnet_ts_to_unix(ts: float) -> int:
    return round(ts)

distance_func = {
    'euclidean': lambda x, y, z: (x**2 + y**2 + z**2)**0.5,
    'manhattan': lambda x, y, z: abs(x) + abs(y) + abs(z)
}

def find_stationary_periods_of_skeleton_df(skel_df: pd.DataFrame,
                                           stationary_threshold_m: float = 0.08,
                                           distance: Literal['euclidean', 'manhattan'] = 'manhattan',
                                           duration_threshold_ms: int = 2000,
                                           keypoints: list = [7, 11]) -> List[dict]:
    distance_func_to_use = distance_func[distance]
    stat_periods = []
    i = 0
    n = len(skel_df)
    
    while i < n:
        # 以当前帧作为静止候选参考帧
        ref_row = skel_df.iloc[i]
        start_ts = ref_row['timestamp']
        start_unix_ms = ref_row['unix_ms']
        j = i + 1
        
        # 标记当前帧开始的静止段
        while j < n:
            current_row = skel_df.iloc[j]
            diff_sum = 0.0
            for kp in keypoints:
                kp_str = kntk.KeypointType(kp).name.lower()
                dx = current_row[f"{kp_str}_x"] - ref_row[f"{kp_str}_x"]
                dy = current_row[f"{kp_str}_y"] - ref_row[f"{kp_str}_y"]
                dz = current_row[f"{kp_str}_z"] - ref_row[f"{kp_str}_z"]
                diff_sum += distance_func_to_use(dx, dy, dz)
            # 如果差异超过阈值，则结束当前静止段的判断
            if diff_sum >= stationary_threshold_m:
                break
            j += 1
        
        # 如果从 i 到 j-1 的时间持续超过阈值，则记录静止段
        end_row = skel_df.iloc[j - 1]
        duration = end_row['timestamp'] - start_ts
        if duration >= duration_threshold_ms:
            stat_periods.append({
                'start_ts': start_ts,
                'end_ts': end_row['timestamp'],
                'start_unix_ms': start_unix_ms,
                'end_unix_ms': end_row['unix_ms'],
                'duration': duration
            })
        # 将 i 移动到 j 处，继续寻找下一段静止状态
        i = j
    
    return stat_periods

class TimeCalibrator:
    def __init__(self,
                 episode: 'Episode',
                 convert_ts_to_unix: bool = True):
        self.convert_ts_to_unix = convert_ts_to_unix
        self.episode = episode
        pcd_meta = episode.pcd_meta
        assert pcd_meta is not None, 'pcd meta data not loaded.'
        self.radar_still_start_ts: int = pcd_meta['still_start_ts']
        self.radar_still_end_ts: int = pcd_meta['still_end_ts']
        self.radar_calib_frames: int = pcd_meta['calib_frames']
        
        self.output_pcd_df = None
        self.output_skel_df = None
        
    def calib(self, motion_start_ts: Optional[Tuple] = None, skel_ts_col: str = 'unix_ms'):
        if motion_start_ts is not None:
            radar_motion_start_ts, kinect_motion_start_ts = motion_start_ts
        else:
            radar_motion_start_ts, kinect_motion_start_ts = self.locate_motion_start_ts()

        print(f"Radar Motion Start TS: {radar_motion_start_ts}, kinect motion start ts: {kinect_motion_start_ts}")
        self.output_skel_df = self.episode.clip_skel_by_ts(start_ts_inclusive=kinect_motion_start_ts,
                                     ts_col=skel_ts_col)
        self.output_pcd_df = self.episode.clip_pcd_by(col='ts', start_seq_inclusive=radar_motion_start_ts)
        self.output_pcd_df.loc[:, 'seq'] = self.output_pcd_df['seq'] - self.output_pcd_df['seq'].iloc[0]
        self.shift_skel_df(self.output_skel_df, shift_ts_ms=radar_motion_start_ts - kinect_motion_start_ts)
        self.give_skel_real_ts(self.output_skel_df, base_ts=radar_motion_start_ts)
    
    def give_skel_real_ts(self, skel_df, base_ts: int):
        first_ts = int(round(skel_df['timestamp'].iloc[0]))
        skel_df['real_ts'] = base_ts + (skel_df['timestamp'].round().astype(int) - first_ts)
        
    def shift_skel_df(self, skel_df, shift_ts_ms: int):
        skel_df['timestamp'] = skel_df['timestamp'] + shift_ts_ms
        skel_df['unix_ms'] = skel_df['unix_ms'] + shift_ts_ms
        
    def locate_motion_start_ts(self):
        skel_df = self.episode.skel_df
        assert skel_df is not None, 'skeleton data not loaded.'
        stationary_periods = find_stationary_periods_of_skeleton_df(skel_df)
        
        calib_period = stationary_periods[-1]
        # print(f"Try Calibration Period:"
        #     f"<start>{round(dnet_ts_to_unix(calib_period['start_ts']) / 1000, 2)}s "
        #     f"<end>{round(dnet_ts_to_unix(calib_period['end_ts']) / 1000, 2)}s "
        #     f"<start_unix>{calib_period['start_unix_ms']} "
        #     f"<end_unix>{calib_period['end_unix_ms']} "
        #     f"<duration>{round(calib_period['duration'] / 1000, 2)}s ")
        radar_motion_start_ts = self.get_radar_motion_start_ts()
        kinect_motion_start_ts = calib_period['end_unix_ms']
        return radar_motion_start_ts, kinect_motion_start_ts
    
    def get_radar_motion_start_ts(self):
        motion_start_frame_idx = self.episode.pcd_meta.get('calib_frames')
        pcd_df = self.episode.pcd_df
        motion_start_ts = pcd_df.loc[pcd_df['seq'] == motion_start_frame_idx, 'ts'].iloc[0]
        return motion_start_ts

    def make_episode(self):
        from .episode import Episode
        episode = Episode(episode_name=self.episode.episode_name,
                          episode_length=self.output_pcd_df['ts'].nunique())
        # assert episode.episode_length == self.episode.episode_length, 'episode length mismatch.'
        episode.pcd_df = self.output_pcd_df
        episode.skel_df = self.output_skel_df
        episode.pcd_meta = self.episode.pcd_meta
        
        return episode