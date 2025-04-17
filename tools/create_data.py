import argparse
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from mwpose3d.tools.rawproc.data_manager.core import DataProcessorDelegate, DataProcessorGUI
from mwpose3d.tools.rawproc.episode import Episode
from mwpose3d.tools.rawproc.hdf5_dumper import ToHdf5
import traceback


def main():
    parser = argparse.ArgumentParser(description='Create data.')
    parser.add_argument('--gui', action='store_true', help='Use GUI to select files.', default=False)
    parser.add_argument('--raw-dir', type=str, help='Data directory.', default='./data/raw')
    parser.add_argument('--output-dir', type=str, help='Output directory.', default='./data/experimental/neat')
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    if args.gui:
        print('Using GUI')
        try:
            delegate = DataProcessorDelegate(args.raw_dir, args.output_dir)
            gui = DataProcessorGUI(delegate)
            gui.mainloop()
        except KeyboardInterrupt:
            print('Interrupted')
        except Exception as e:
            traceback.print_exc()
        finally:
            return
    else:
        example_episode_name = "20250227_122654-122851_647"
        example_episode = Episode(episode_name=example_episode_name, episode_length=600)
        example_episode.load_pcd(raw_dir / 'radar')
        example_episode.load_pcd_meta(raw_dir / 'radar' / 'meta')
        example_episode.load_skeleton(raw_dir / 'kinect')
        print(example_episode.pcd_df.columns)
        print(example_episode.skel_df.columns)
        # calibrated_episode = example_episode.calibrate_time()

        # aligned_episode = calibrated_episode.align_traces(use_interp_skel=True)
        
        # final_pcd = aligned_episode.pcd_df
        pcd = example_episode.pcd_df
        print(pcd.columns)
        groups = pcd.groupby('ts')
        group_sizes = groups.size()
        print(group_sizes.value_counts())
        # ToHdf5(alligned_episode=aligned_episode, output_dir=output_dir).save()
if __name__ == '__main__':
    main()