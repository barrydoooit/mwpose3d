from pathlib import Path
from typing import TYPE_CHECKING
import h5py
from mmengine.config import Config
import numpy as np
import sys
from PySide2.QtWidgets import QApplication
from tqdm import tqdm

from mwpose3d.runner import Runner
from mwcore.registry import TRACKERS

from torch.utils.data import DataLoader

from mwpose3d.visualization.pcd_offline import PointCloudOfflineVisualizer

if TYPE_CHECKING:
    from mwcore.tracking.api.base import BaseTracker



class TrackingRecordGenerator:
    def __init__(self, 
                 dataset: str, 
                 tracker_cfg_f: str, 
                 data_prefix: dict[str, str] = dict(pcd='mmwave'), 
                 splits: list[str] = ['train', 'val', 'test'],
                 vis_mode: bool = False):
        self.tracker_cfg_f = tracker_cfg_f
        self.dataset = dataset
        self.data_prefix = data_prefix
        self.splits = splits
        self.vis_mode = vis_mode
    
    def make_tracker(self, tracker_cfg_f: str) -> "BaseTracker":
        tracker_cfg = Config.fromfile(tracker_cfg_f).get('tracker_cfg')
        tracker_cfg['radar_cfg'] = dict(
            sensor_height=0.0,
            sensor_tilt=0.0
        )
        tracker: "BaseTracker" = TRACKERS.build(tracker_cfg)
        return tracker
    
    def make_dataloader(self,
            dataset_type: str,
            data_root: str,
            info_path: str,
            data_prefix: dict,
            pcd_dim: int):
        return Runner.build_dataloader(
            dict(
                batch_size=1,
                num_workers=1,
                shuffle=False,
                dataset=dict(
                    type=dataset_type,
                    data_root=data_root,
                    info_path=info_path,
                    data_prefix=data_prefix,
                    sequence_length=1,
                    pipeline=[
                        dict(
                            type='mwpose3d.LoadMultiFrameFromH5',
                            load_pcd_dim=pcd_dim,
                            num_frames=1,
                            backup_frames=0,
                            empty_frame_op='error',
                            with_skeleton=False
                        ),
                    ],
                )
            )
        )

    def generate(self):
        for split in self.splits:
            data_root = f'data/{self.dataset}'
            info_path = f'{data_root}/info_{split}.pkl'
            dataloader = self.make_dataloader(
                dataset_type='MotionDataset',
                data_root=data_root,
                info_path=info_path,
                data_prefix=self.data_prefix,
                pcd_dim=5,
            )
            if self.vis_mode:
                self.visualize_tracking_records_single(dataloader)
            else:
                self.generate_tracking_records_single(dataloader)

    def generate_tracking_records_single(self, dataloader: DataLoader):
        tracker = None

        current_track_file = None
        current_tracker_type = None
        collected_flats: list[np.ndarray] = []

        pcd_prefix = self.data_prefix['pcd']
        pcd_path_key = f'{pcd_prefix}_path'

        def _flush_sequence():
            nonlocal current_track_file, current_tracker_type, collected_flats
            if current_track_file is None or current_tracker_type is None:
                collected_flats = []
                return
            with h5py.File(current_track_file, 'a') as h5f:
                if current_tracker_type in h5f:
                    del h5f[current_tracker_type]
                grp = h5f.create_group(current_tracker_type)
                vlen_dtype = h5py.vlen_dtype(np.dtype('float32'))
                ds = grp.create_dataset(
                    "track_records",
                    data=np.array(collected_flats, dtype=object),
                    dtype=vlen_dtype,
                )
                ds.attrs['columns'] = np.array(["x", "y", "z"], dtype='S')
            collected_flats = []

        for data in tqdm(dataloader, desc="Generating tracking records"):
            pcd_frame: np.ndarray = data['pcd_frames'][-1][0]
            first_frame: bool = data['starting_flag'][0]
            if first_frame:
                _flush_sequence()
                tracker = self.make_tracker(self.tracker_cfg_f)
                pcd_file_name = data[pcd_path_key][-1]
                data_root = Path(f'data/{self.dataset}')
                track_dir = data_root / 'tracking_records'
                track_dir.mkdir(parents=True, exist_ok=True)
                current_track_file = track_dir / pcd_file_name
                current_tracker_type = tracker.__class__.__name__
                    
            tracked_locations = tracker.consume(point_array=pcd_frame)
            if tracked_locations is None or len(tracked_locations) == 0:
                locs3flat = np.zeros((0,), dtype=np.float32)
      
            else:
                arr = np.asarray(tracked_locations, dtype=np.float32)
                if arr.size == 0:
                    locs3flat = np.zeros((0,), dtype=np.float32)
                elif arr.ndim == 1 and arr.size >= 3:
                    locs3flat = arr[:3].reshape(-1).astype(np.float32)
                elif arr.ndim == 2 and arr.shape[1] >= 3:
                    locs3flat = arr[:, :3].reshape(-1).astype(np.float32)
                else:
                    locs3flat = np.zeros((0,), dtype=np.float32)
            
            collected_flats.append(locs3flat)
        
        _flush_sequence()

            

    def visualize_tracking_records_single(self, dataloader: DataLoader):

        def pcd_generator():
            for data in dataloader:
                pcd_frame: np.ndarray = data['pcd_frames'][-1][0]
                yield pcd_frame
        
        def trk_generator():
            tracker = None
            for data in dataloader:
                pcd_frame: np.ndarray = data['pcd_frames'][-1][0]
                first_frame: bool = data['starting_flag'][0]
                if first_frame:
                    tracker = self.make_tracker(self.tracker_cfg_f)
                    print("New Sequence, initializing tracker.")
                tracked_locations = tracker.consume(point_array=pcd_frame)
                yield tracked_locations
        total = len(dataloader) if hasattr(dataloader, '__len__') else None
        app = QApplication(sys.argv)
        visualizer = PointCloudOfflineVisualizer(
            point_clouds=pcd_generator(),
            tracking_data=trk_generator(),
            total_frames=total,
            play_fps=10,
            tracking_mode='dot'
        )
        visualizer.show()
        app.exec_()