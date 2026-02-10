import pandas as pd
from pathlib import Path

# Config
output_dir = Path(r"E:\Projects\mwpose3d\tmpraw")
episode_name = "radar_bin" # Based on your log

pcd_path = output_dir / f"{episode_name}_pcd.pkl"
skel_path = output_dir / f"{episode_name}_skel.pkl"

if not pcd_path.exists() or not skel_path.exists():
    print(f"Files not found in {output_dir}")
    exit()

# 1. Inspect Radar Data (Point Cloud)
print(f"Loading {pcd_path}...")
pcd_df = pd.read_pickle(pcd_path)
print(f"PCD Shape: {pcd_df.shape} (rows = total points)")
print(pcd_df[['seq', 'ts', 'x', 'y', 'z']].head())

# 2. Inspect Skeleton Data
print(f"\nLoading {skel_path}...")
skel_df = pd.read_pickle(skel_path)
print(f"Skeleton Shape: {skel_df.shape} (rows = aligned frames)")
print(skel_df[['pcd_ts']].head())

# 3. Verify Alignment Logic
# Every unique radar timestamp should have exactly one corresponding skeleton frame
unique_radar_ts_count = pcd_df['ts'].nunique()
aligned_skel_count = len(skel_df)

print(f"\n--- Verification Results ---")
print(f"Unique Radar Timestamps: {unique_radar_ts_count}")
print(f"Aligned Skeleton Frames: {aligned_skel_count}")

if unique_radar_ts_count == aligned_skel_count:
    print("✅ SUCCESS: Alignment is consistent.")
else:
    print(f"❌ MISMATCH: Expected {unique_radar_ts_count} frames but got {aligned_skel_count} skeleton entries.")