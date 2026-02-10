
import argparse
import sys
from pathlib import Path
import logging

# Add project root to sys.path to ensure imports work correctly
# This allows running the script directly from any directory
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]  # tools/rawproc/run_alignment_raw.py -> tools/rawproc -> tools -> root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from tools.rawproc.episode import Episode
from tools.rawproc.alignment import AlignTraces

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Align raw radar data (.bin) with Kinect data (.csv)")

    parser.add_argument("--radar-bin", type=str, help="Path to radar .bin file")
    parser.add_argument("--kinect-csv", type=str, help="Path to Kinect .csv file")
    parser.add_argument("--dir", type=str, help="Base directory containing 'raw' and 'kinect' subfolders. Overrides --radar-bin/--kinect-csv")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save aligned output")
    parser.add_argument("--episode-name", type=str, help="Episode name (optional, defaults to filename stem)")
    
    args = parser.parse_args()
    
    # Logic to resolve paths
    radar_path = None
    kinect_path = None
    
    if args.dir:
        base_dir = Path(args.dir)
        raw_dir = base_dir / 'raw'
        kinect_dir = base_dir / 'kinect'
        
        if not raw_dir.exists():
            # Fallback for "radar_bin" folder naming if "raw" doesn't exist?
            # User specifically said "raw/", sticking to it strictly for now.
            logger.error(f"Raw directory not found: {raw_dir}")
            return
        if not kinect_dir.exists():
            logger.error(f"Kinect directory not found: {kinect_dir}")
            return
            
        # Find files
        # We'll take the first one found or error if multiple/none
        radar_files = list(raw_dir.glob("*.bin"))
        kinect_files = list(kinect_dir.glob("*.csv"))
        
        if len(radar_files) == 0:
            logger.error(f"No .bin files found in {raw_dir}")
            return
        if len(kinect_files) == 0:
            logger.error(f"No .csv files found in {kinect_dir}")
            return
            
        if len(radar_files) > 1 or len(kinect_files) > 1:
            logger.warning(f"Multiple files found in {args.dir}. Using the first pair found.")
            # Optional: Implement matching logic here if needed (e.g. by timestamp)
            # For now, simplistic approach:
            
        radar_path = radar_files[0]
        kinect_path = kinect_files[0]
        
        logger.info(f"Auto-detected inputs:\nRadar: {radar_path}\nKinect: {kinect_path}")
        
    else:
        if not args.radar_bin or not args.kinect_csv:
            parser.error("If --dir is not specified, --radar-bin and --kinect-csv are required.")
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
