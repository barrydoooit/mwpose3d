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
                                 ]:#'SkeletonCoordinateTransform', 'PointCloudCoordinateTransform']:
            dataloader_new['dataset']['pipeline'].remove(transform)
    dataloader_new.update(dict(batch_size=1, num_workers=0, shuffle=False))
    dataloader = runner.build_dataloader(dataloader_new)
    
    if not vis:
        # --- ORIGINAL VOL / SNR setup ---
        indice_to_check = (3, 4)
        names = ('VOL', 'SNR')
        pcd_values = [[], []]
        pcd_count = []
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
            try:
                # accumulate VOL & SNR
                for i, ind in enumerate(indice_to_check):
                    pcd_values[i].append(pcd_frame[:, ind])
                pcd_count.append(pcd_frame.shape[0])
            except IndexError as e:
                pass

            # accumulate XYZ coords
            for i, ind in enumerate(coord_indices):
                coord_values[i].append(pcd_frame[:, ind])

            # accumulate skeletons as before
            skel_frame = data_batch['skel_frames'][-1][0]
            skel_values.append(skel_frame)

        # --- ORIGINAL stats for VOL & SNR ---
        try:
            for i, name in enumerate(names):
                arr = np.concatenate(pcd_values[i], axis=0)
                print(f'{name} shape: {arr.shape}')
                print(f'{name} mean : {arr.mean():.4f}')
                print(f'{name} std  : {arr.std():.4f}')
            print(f'Average number of points per frame: {np.mean(pcd_count):.2f}')
            print('─' * 50)
        except Exception as e:
            logging.warning(f"VOL/SNR not available")
        # --- NEW: percentiles for XYZ ---
        perc = [0, 1, 2, 5, 10, 90, 95, 98, 99, 100]
        labels = ['min', '1%', '2%', '5%', '10%', '90%', '95%', '98%', '99%', 'max']

        for i, name in enumerate(coord_names):
            arr = np.concatenate(coord_values[i], axis=0)
            pvals = np.percentile(arr, perc)
            print(f'Axis {name} mean : {arr.mean():.4f}')
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
        import h5py

        # Pre-load pointing flags from all skeleton H5 files in the dataset
        pointing_flags_cache = {}  # str -> np.ndarray or None
        data_root = dataloader.dataset.data_root
        skel_dir = data_root / 'skeleton' if hasattr(data_root, '__truediv__') else None
        if skel_dir and skel_dir.is_dir():
            for h5f in skel_dir.glob('*.h5'):
                try:
                    with h5py.File(h5f, 'r') as f:
                        if 'pointing_gesture' in f:
                            pointing_flags_cache[str(h5f)] = f['pointing_gesture'][:]
                except Exception:
                    pass

        def _is_pointing(data_batch):
            """Check pointing flag for the current batch item."""
            data_file = data_batch.get('data_file', {})
            skel_path = data_file.get('skel', None)
            if isinstance(skel_path, (list, tuple)):
                skel_path = skel_path[0]
            if skel_path is None:
                return False
            flags = pointing_flags_cache.get(str(skel_path), None)
            if flags is None:
                return False
            local_idx = data_batch.get('local_idx', 0)
            if isinstance(local_idx, (list, tuple)):
                local_idx = local_idx[-1]
            return bool(flags[local_idx]) if local_idx < len(flags) else False

        # Collect all data in a single pass (no separate generator per modality)
        pcd_list, skel_list, pointing_list = [], [], []
        total = len(dataloader) if hasattr(dataloader, '__len__') else None
        for idx, data_batch in tqdm(enumerate(dataloader), desc='Loading frames', total=total):
            pcd_list.append(data_batch['pcd_frames'][-1][0])
            skel_list.append(data_batch['skel_frames'][-1][0])
            pointing_list.append(_is_pointing(data_batch))

        from PySide6.QtWidgets import QApplication
        app = QApplication(sys.argv)
        from mwpose3d.visualization import PointCloudOfflineVisualizerSK

        class _PointingVisualizer(PointCloudOfflineVisualizerSK):
            """Thin subclass that sets per-frame colors based on pointing flags."""
            def __init__(self, pcds, skels, flags, **kwargs):
                self._pointing_flags = flags
                super().__init__(pcds, skels, **kwargs)

            def update_display(self):
                is_pointing = (self.current_frame < len(self._pointing_flags)
                               and self._pointing_flags[self.current_frame])
                self.pcd_color = (1, 1, 0, 1) if is_pointing else None
                self.skel_color = (0, 0, 1, 1) if is_pointing else (0, 1, 0, 1)
                super().update_display()

        visualizer = _PointingVisualizer(
            pcd_list, skel_list, pointing_list,
            total_frames=len(pcd_list),
            play_fps=60,
        )
        visualizer.show()
        app.exec_()
if __name__ == '__main__':
    main()