import argparse
import sys
import os
import debugpy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.dataset_converters.generate_tracking import TrackingRecordGenerator


def main():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument('dataset', help='name of the dataset')
    parser.add_argument('--trec', action='store_true', default=False, help='generate tracking records')
    parser.add_argument('--tcfg', type=str, help='tracker config file path')
    parser.add_argument('--pcd-prefix', type=str, default='mmwave', help='prefix for point cloud data folder')
    parser.add_argument('--vis', action='store_true', default=False, help='visualize the dataset')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], help='dataset splits to process')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        debugpy.listen(5678)
        print('Waiting for debugger attach')
        debugpy.wait_for_client()
    
    if args.trec:
        trec_grt = TrackingRecordGenerator(
            dataset=args.dataset,
            tracker_cfg_f=args.tcfg,
            data_prefix=dict(pcd=args.pcd_prefix),
            splits=args.splits,
            vis_mode=args.vis
        )

        trec_grt.generate()


if __name__ == '__main__':
    main()