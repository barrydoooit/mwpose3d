import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PySide6.QtWidgets import QApplication

# ensures can run from mwpose3d/ root
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mwpose3d.visualization.pcd_offline import PointCloudOfflineVisualizerSK
from mwpose3d.utils.kinect_toolkits.kinectData import USED_KEYPOINTS, KeypointType

def get_skeleton_columns():
    """Helper to get column names in order [joint1.x, joint1.y, joint1.z, joint2.x, ...]"""
    cols = []
    for kp_val in USED_KEYPOINTS:
        kp_name = KeypointType(kp_val).name.lower()
        cols.extend([f"{kp_name}.x", f"{kp_name}.y", f"{kp_name}.z"])
    return cols

def main():
    data_dir = Path(r"E:\Projects\mwpose3d\tmpraw")
    episode_name = "radar_bin"
    
    pcd_path = data_dir / f"{episode_name}_pcd.pkl"
    skel_path = data_dir / f"{episode_name}_skel.pkl"

    if not pcd_path.exists() or not skel_path.exists():
        print(f"Error: Could not find files in {data_dir}")
        print(f"Looking for: {pcd_path.name} and {skel_path.name}")
        return

    print(f"Loading data from {data_dir}...")
    pcd_df = pd.read_pickle(pcd_path)
    skel_df = pd.read_pickle(skel_path)

    print(f"Loaded {len(pcd_df)} points and {len(skel_df)} skeleton frames.")


    print("Preparing point clouds...")
    pcd_df = pcd_df.sort_values('ts')
    

    pcd_frames = []
    for ts, group in pcd_df.groupby('ts', sort=False):
        xyz = group[['x', 'y', 'z']].values
        pcd_frames.append(xyz)

    print("Preparing skeletons...")
    skel_cols = get_skeleton_columns()
    
    valid_cols = [c for c in skel_cols if c in skel_df.columns]
    skel_frames = skel_df[valid_cols].values
    skel_frames_list = [frame for frame in skel_frames]

    print("Launching visualizer...")
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    assert len(pcd_frames) == len(skel_frames_list), "PCD and skeleton frames length mismatch"

    visualizer = PointCloudOfflineVisualizerSK(
        point_clouds=pcd_frames,
        skeletons=skel_frames_list,
        total_frames=len(pcd_frames),
        play_fps=10, 
        tracking_mode='dot'
    )
    
    visualizer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()