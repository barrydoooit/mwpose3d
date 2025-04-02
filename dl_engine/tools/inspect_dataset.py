import argparse
import logging
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../..'))

import debugpy
from mmengine.config import Config, DictAction

from dl_engine.runner.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description='Inspect a dataset')
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--work-dir', help='the dir to save logs and models', default=None)
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
        val='val_dataloader',
        test='test_dataloader'
    )
    existing_dataloader_keys = required_dataloader_keys.copy()
    for key, dataloader_key in required_dataloader_keys.items():
        if getattr(cfg, dataloader_key, None) is None:
            existing_dataloader_keys.pop(key)
    
    for label, dataloader_key in existing_dataloader_keys.items():
        dataloader = getattr(cfg, dataloader_key, None)
        if dataloader is not None:
            print(f'Inspecting {label} dataloader...')
            inspect(runner, dataloader)
            
def inspect(runner, dataloader: Config):
    # dataloader = dataloader.to_dict()
    for transform in dataloader['dataset']['pipeline']:
        if transform['type'] == 'PointDuplicator':
            dataloader['dataset']['pipeline'].remove(transform)
    dataloader.update(dict(batch_size=1, shuffle=False))
    dataloader = runner.build_dataloader(dataloader)
    
    indice_to_check = (3, 4,)
    names = ('VOL', 'SNR',)
    values = [[], []]
    for idx, data_batch in enumerate(dataloader):
        pcd_frame_list = data_batch['pcd_frames']
        for i, (ind, name) in enumerate(zip(indice_to_check, names)):
            values[i].append(pcd_frame_list[-1][0][:, ind])

    for i, name in enumerate(names):
        values[i] = np.concatenate(values[i], axis=0)
        print(f'{name} shape: {values[i].shape}')
        print(f'{name} mean: {values[i].mean()}')
        print(f'{name} std: {values[i].std()}')
    print('--------------------------------------------------')
if __name__ == '__main__':
    main()