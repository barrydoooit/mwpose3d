import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PySide6.QtWidgets import QApplication

# Add project root to sys.path
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
    # 1. Configuration
    # Adjust these paths to match your data
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

    # 2. Prepare Point Clouds
    print("Preparing point clouds...")
    # Ensure sorted by timestamp
    pcd_df = pcd_df.sort_values('ts')
    
    # Group by timestamp to get one frame per step
    # The visualizer expects a list of numpy arrays (N, 3)
    pcd_frames = []
    # Note: Using 'ts' to group because it's the alignment key
    for ts, group in pcd_df.groupby('ts', sort=False):
        xyz = group[['x', 'y', 'z']].values
        pcd_frames.append(xyz)

    # 3. Prepare Skeletons
    print("Preparing skeletons...")
    skel_cols = get_skeleton_columns()
    
    # Filter columns that actually exist in the dataframe
    valid_cols = [c for c in skel_cols if c in skel_df.columns]
    
    # Extract values as a list of flat arrays (one row per frame)
    skel_frames = skel_df[valid_cols].values
    # Convert to list of arrays for the visualizer
    skel_frames_list = [frame for frame in skel_frames]

    # 4. Launch Visualizer
    print("Launching visualizer...")
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    # Ensure equal length for playback (alignment should guarantee this, but safety first)
    min_len = min(len(pcd_frames), len(skel_frames_list))
    pcd_frames = pcd_frames[:min_len]
    skel_frames_list = skel_frames_list[:min_len]

    visualizer = PointCloudOfflineVisualizerSK(
        point_clouds=pcd_frames,
        skeletons=skel_frames_list,
        total_frames=min_len,
        play_fps=10,  # Adjust playback speed here
        tracking_mode='dot' # Optional: 'dot' or 'bbox' if you had tracking data
    )
    
    visualizer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()