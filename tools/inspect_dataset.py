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
        indice_to_check = (3, 4,)
        names = ('VOL', 'SNR',)
        pcd_values = [[], []]
        skel_values = []
        total = len(dataloader) if hasattr(dataloader, '__len__') else None
        for idx, data_batch in tqdm(enumerate(dataloader), desc='Processing data', total=total):
            pcd_frame_list = data_batch['pcd_frames']
            for i, (ind, name) in enumerate(zip(indice_to_check, names)):
                pcd_values[i].append(pcd_frame_list[-1][0][:, ind])
            
            skel_frame_list = data_batch['skel_frames']
            skel_batch = skel_frame_list[-1][0]
            skel_values.append(skel_batch)
            
        for i, name in enumerate(names):
            pcd_values[i] = np.concatenate(pcd_values[i], axis=0)
            print(f'{name} shape: {pcd_values[i].shape}')
            print(f'{name} mean: {pcd_values[i].mean()}')
            print(f'{name} std: {pcd_values[i].std()}')
        print('--------------------------------------------------')

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