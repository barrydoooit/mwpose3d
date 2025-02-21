import argparse
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from dl_engine.tools.rawproc.episode import Episode
from dl_engine.tools.rawproc.hdf5_dumper import ToHdf5


def main():
    parser = argparse.ArgumentParser(description='Create data.')
    parser.add_argument('--raw-dir', type=str, help='Data directory.')
    parser.add_argument('--output-dir', type=str, help='Output directory.')
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    
    example_episode_name = "20250219_114617-114702_235"
    example_episode = Episode(episode_name=example_episode_name, episode_length=200)
    example_episode.load_pcd(raw_dir / 'radar')
    example_episode.load_pcd_meta(raw_dir / 'radar' / 'meta')
    example_episode.load_skeleton(raw_dir / 'kinect')
    
    calibrated_episode = example_episode.calibrate_time()

    aligned_episode = calibrated_episode.align_traces(use_interp_skel=True)
    ToHdf5(alligned_episode=aligned_episode, output_dir=output_dir).save()
if __name__ == '__main__':
    main()