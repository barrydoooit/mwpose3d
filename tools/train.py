import argparse
import logging
import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..'))

import debugpy
from mmengine.config import Config, DictAction

from mwpose3d.runner.runner import Runner

def parse_args(argv: list[str] = None):
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    if argv:
        return parser.parse_args(argv)
    return parser.parse_args()

def main(argv: list[str] = None):
    args = parse_args(argv)
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
    runner.train()

if __name__ == '__main__':
    main()