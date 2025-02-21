import json
from pathlib import Path
from typing import List

import pandas as pd

import kinect_toolkits as kntk


def load_radar_schema_json_to_df(json_path: Path) -> pd.DataFrame:
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frame_keys = data['frame_keys'] # ["seq", "ts", "points"]
    point_keys = data['point_keys'] # ["x", "y", "z", "vel", "snr"]
    pcd_frames = data['frames']
    
    rows = []
    for frame in pcd_frames:
        frame_dict = dict(zip(frame_keys, frame))
        seq, ts, points = frame_dict['seq'], frame_dict['ts'], frame_dict['points']
        for point in points:
            point_dict = dict(zip(point_keys, point))
            point_dict.update({'seq': seq, 'ts': int(ts)})
            rows.append(point_dict)
    df =  pd.DataFrame(rows)
    # df['ts'] = df['ts'].astype(pd.Int64Dtype())
    return df

def clean_raw_skeleton_df(raw_skeleton_df: pd.DataFrame) -> pd.DataFrame:
    raw_skeleton_df.rename(columns=lambda x: x.strip('#').strip(), inplace=True)
    cleaned_df = raw_skeleton_df.groupby('timestamp').filter(lambda group: len(group) == len(kntk.kinectData.ALL_KEYPOINTS))
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df

def load_raw_skeleton_csv_to_df(csv_path: Path, parse: bool = True) -> pd.DataFrame:
    raw_skeleton_df = clean_raw_skeleton_df(pd.read_csv(csv_path))
    if not parse:
        return raw_skeleton_df
    
    skeleton_list: List[kntk.Skeleton] = kntk.process_record(raw_skeleton_df)
    rows = []
    for skeleton in skeleton_list:
        ts = skeleton.timestamp
        row = {'timestamp': ts, 'unix_ms': skeleton.unix_ms}
        for kp_type, kp in skeleton.keypoints.items():
            kp_type_str = kp_type.name.lower()
            row.update({f'{kp_type_str}_x': kp.x,
                        f'{kp_type_str}_y': kp.y,
                        f'{kp_type_str}_z': kp.z})
        rows.append(row)
    return pd.DataFrame(rows)