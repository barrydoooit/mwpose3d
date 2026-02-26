import argparse
import sys
import os
import debugpy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.dataset_converters.generate_tracking_v2 import TrackingRecordGeneratorV2
from tools.dataset_converters.generate_pointing_annotation import PointingGestureAnnotator


def main():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument('dataset', help='name of the dataset')
    parser.add_argument('--trec', action='store_true', default=False, help='generate tracking records')
    parser.add_argument('--hpe-cfg', type=str, help='hpe config file path')
    parser.add_argument('--pcd-prefix', type=str, default='mmwave', help='prefix for point cloud data folder')
    parser.add_argument('--vis', action='store_true', default=False, help='visualize the dataset')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], help='dataset splits to process')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    # Pointing gesture annotation
    parser.add_argument('--pointing', action='store_true', default=False, help='generate pointing gesture annotations')
    parser.add_argument('--pointing-angle', type=float, default=160.0, help='min elbow angle (degrees) for pointing detection')
    parser.add_argument('--pointing-window', type=int, default=15, help='temporal stability window size (frames)')
    parser.add_argument('--pointing-var', type=float, default=0.005, help='max spatial variance for temporal stability')

    args = parser.parse_args()
    if args.debug:
        debugpy.listen(5678)
        print('Waiting for debugger attach')
        debugpy.wait_for_client()
    
    if args.trec:
        trec_grt = TrackingRecordGeneratorV2(
            dataset=args.dataset,
            hpe_cfg_f=args.hpe_cfg,
            data_prefix=dict(pcd=args.pcd_prefix),
            splits=args.splits,
        )
        if args.vis:
            trec_grt.visualize()
        else:
            trec_grt.generate()

    if args.pointing:
        pointing_ann = PointingGestureAnnotator(
            hpe_cfg_f=args.hpe_cfg,
            elbow_angle_threshold=args.pointing_angle,
            stability_window=args.pointing_window,
            stability_max_variance=args.pointing_var,
        )
        pointing_ann.generate()


if __name__ == '__main__':
    main()