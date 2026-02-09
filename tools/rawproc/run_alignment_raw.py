
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tools.rawproc.episode import Episode
from tools.rawproc.alignment import AlignTraces
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Align raw radar data (.bin) with Kinect data (.csv)")
    parser.add_argument("--radar-bin", type=str, required=True, help="Path to radar .bin file")
    parser.add_argument("--kinect-csv", type=str, required=True, help="Path to Kinect .csv file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save aligned output")
    parser.add_argument("--episode-name", type=str, help="Episode name (optional, defaults to filename stem)")
    
    args = parser.parse_args()
    
    radar_path = Path(args.radar_bin)
    kinect_path = Path(args.kinect_csv)
    output_dir = Path(args.output_dir)
    
    if not radar_path.exists():
        logger.error(f"Radar file not found: {radar_path}")
        return
    if not kinect_path.exists():
        logger.error(f"Kinect file not found: {kinect_path}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    episode_name = args.episode_name or radar_path.stem
    logger.info(f"Processing episode: {episode_name}")
    
    # 1. Initialize Episode
    episode = Episode(episode_name)
    
    # 2. Load Radar Data
    logger.info("Loading radar data...")
    try:
        episode.load_pcd_bin(radar_path)
        logger.info(f"Loaded {len(episode.pcd_df)} radar points.")
    except Exception as e:
        logger.error(f"Failed to load radar data: {e}")
        return

    # 3. Load Kinect Data
    logger.info("Loading Kinect data...")
    # Manually load kinect data since Episode.load_skeleton expects a directory structure
    # and specific naming convention if using load_skeleton.
    # But we can just set episode.skel_df directly.
    from tools.rawproc import load_utils
    try:
        # We use load_raw_skeleton_csv_to_df for raw kinect data
        # Note: existing load_utils.load_framed_skeleton_csv_to_df is for already framed data?
        # Let's check load_utils.py again.
        # load_raw_skeleton_csv_to_df calls process_record which does skeleton parsing.
        # Ideally we want the raw data to be parsed into the format Episode expects.
        # Episode usually expects "framed_skeleton" (one row per frame, clean columns).
        # load_raw_skeleton_csv_to_df returns a DataFrame with 'unix_ms', 'timestamp', and 'joint.x/y/z' columns.
        
        episode.skel_df = load_utils.load_raw_skeleton_csv_to_df(kinect_path)
        logger.info(f"Loaded {len(episode.skel_df)} skeleton frames.")
    except Exception as e:
        logger.error(f"Failed to load Kinect data: {e}")
        return
        
    # 4. Align
    logger.info("Aligning traces...")
    try:
        # AlignTraces uses 'unix_ms' by default and 60ms offset.
        # Ensure our data has 'unix_ms'. load_raw_skeleton_csv_to_df provides it.
        # radar pcd_df 'ts' should be in ms (we converted seconds to ms in load_pcd_bin).
        
        aligner = AlignTraces(episode)
        aligner.align()
        
        # Check results
        aligned_episode = aligner.make_episode(episode_name)
        logger.info(f"Alignment complete. Aligned length: {len(aligned_episode.pcd_df)}")
        
        # 5. Save Output
        # Save as pickle or whatever format is used.
        # Episode doesn't have a save method shown in the snippet, 
        # but we can save the dataframes.
        
        output_pcd_path = output_dir / f"{episode_name}_pcd.pkl"
        output_skel_path = output_dir / f"{episode_name}_skel.pkl"
        
        aligned_episode.pcd_df.to_pickle(output_pcd_path)
        aligned_episode.skel_df.to_pickle(output_skel_path)
        
        logger.info(f"Saved aligned data to {output_dir}")
        
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
