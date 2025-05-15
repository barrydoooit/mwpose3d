import argparse
from copy import deepcopy
import logging
import os
import os.path as osp
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..'))

import debugpy
from mmengine.config import Config, DictAction

from mwpose3d.runner.runner import Runner



def parse_args():
    parser = argparse.ArgumentParser(description='Inspect a dataset')
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--work-dir', help='the dir to save logs and models', default=None)
    parser.add_argument('--vis', action='store_true', help='visualize the dataset', default=False)
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    
    return parser.parse_args()

def main():
    args = parse_args()
    if args.debug:
        debugpy.listen(5678)
        print('Waiting for debugger attach')
        debugpy.wait_for_client()
    
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    runner = Runner.from_cfg(cfg)
    assert isinstance(cfg.train_dataloader, dict)
    

    # Check that the required dataloaders exist in the config
    required_dataloader_keys = dict(
        train='train_dataloader',
        # val='val_dataloader',
        # test='test_dataloader'
    )
    existing_dataloader_keys = required_dataloader_keys.copy()
    for key, dataloader_key in required_dataloader_keys.items():
        if getattr(cfg, dataloader_key, None) is None:
            existing_dataloader_keys.pop(key)
    
    for label, dataloader_key in existing_dataloader_keys.items():
        dataloader = getattr(cfg, dataloader_key, None)
        if dataloader is not None:
            print(f'Inspecting {label} dataloader...')
            inspect(runner, dataloader, args.vis)
            
def inspect(runner, dataloader: Config, vis: bool = False):
    dataloader_new = deepcopy(dataloader.to_dict())
    for transform in dataloader['dataset']['pipeline']:
        if transform['type'] in ['PointDuplicator', 'PointPadding', 'PointSortAndClip', 'RandomTransform', 'NormalizePointAttr', 'SkeletonCoordNormalization',
                                 'SkeletonCoordinateTransform', 'PointCloudCoordinateTransform']:
            dataloader_new['dataset']['pipeline'].remove(transform)
    dataloader_new.update(dict(batch_size=1, num_workers=0, shuffle=False))
    dataloader = runner.build_dataloader(dataloader_new)
    
    if not vis:
        # --- ORIGINAL VOL / SNR setup ---
        indice_to_check = (3, 4)
        names = ('VOL', 'SNR')
        pcd_values = [[], []]

        # --- NEW: XYZ setup ---
        coord_indices = (0, 1, 2)
        coord_names = ('x', 'y', 'z')
        coord_values = [[], [], []]

        skel_values = []

        total = len(dataloader) if hasattr(dataloader, '__len__') else None
        for idx, data_batch in tqdm(enumerate(dataloader),
                                    desc='Processing data', total=total):
            # last point-cloud frame in this batch
            pcd_frame = data_batch['pcd_frames'][-1][0]  # shape (N_points, channels)

            # accumulate VOL & SNR
            for i, ind in enumerate(indice_to_check):
                pcd_values[i].append(pcd_frame[:, ind])

            # accumulate XYZ coords
            for i, ind in enumerate(coord_indices):
                coord_values[i].append(pcd_frame[:, ind])

            # accumulate skeletons as before
            skel_frame = data_batch['skel_frames'][-1][0]
            skel_values.append(skel_frame)

        # --- ORIGINAL stats for VOL & SNR ---
        for i, name in enumerate(names):
            arr = np.concatenate(pcd_values[i], axis=0)
            print(f'{name} shape: {arr.shape}')
            print(f'{name} mean : {arr.mean():.4f}')
            print(f'{name} std  : {arr.std():.4f}')
        print('─' * 50)
        # --- NEW: percentiles for XYZ ---
        perc = [0, 1, 2, 5, 10, 90, 95, 98, 99, 100]
        labels = ['min', '1%', '2%', '5%', '10%', '90%', '95%', '98%', '99%', 'max']

        for i, name in enumerate(coord_names):
            arr = np.concatenate(coord_values[i], axis=0)
            pvals = np.percentile(arr, perc)

            # print header
            header = '  '.join(f'{lab:>5}' for lab in labels)
            values = '  '.join(f'{val:>5.4f}' for val in pvals)

            print(f"Axis '{name}':")
            print(header)
            print(values)
            print('─' * 50)
        
        skel_all = np.stack(skel_values, axis=0)
        skel_means = skel_all.mean(axis=0)
        skel_stds = skel_all.std(axis=0)

        skel_means_axis = skel_means.reshape(-1, 3).mean(axis=0)
        print(f'Skeleton mean: x={skel_means_axis[0]:.2f}, y={skel_means_axis[1]:.2f}, z={skel_means_axis[2]:.2f}')
        # print('Skeleton mean:', [round(x, 2) for x in skel_means])
        # print('Skeleton std:', [round(x, 2) for x in skel_stds])
    
    if vis:
        from PySide2.QtWidgets import QApplication
        from mwpose3d.visualization import PointCloudOfflineVisualizerSK

        def pcd_generator():
            for idx, data_batch in enumerate(dataloader):
                assert len(data_batch['pcd_frames'][-1]) == 1
                print(f'pcd size: {data_batch["pcd_frames"][-1][0].shape}')
                yield data_batch['pcd_frames'][-1][0]
        
        def skel_generator():
            for idx, data_batch in enumerate(dataloader):
                yield data_batch['skel_frames'][-1][0]
        
        app = QApplication(sys.argv)
        total = len(dataloader) if hasattr(dataloader, '__len__') else None
        visualizer = PointCloudOfflineVisualizerSK(
            pcd_generator(),
            skel_generator(),
            total_frames=total,
            play_fps=60,
        )
        visualizer.show()
        app.exec_()
if __name__ == '__main__':
    main()