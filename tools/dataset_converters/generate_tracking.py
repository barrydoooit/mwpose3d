from typing import TYPE_CHECKING
from mmengine.config import Config
import numpy as np
from mwpose3d.runner import Runner
from mwcore.registry import TRACKERS

from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from mwcore.tracking.api.base import BaseTracker



class TrackingRecordGenerator:
    def __init__(self, dataset: str, tracker_cfg_f: str, data_prefix: dict[str, str] = dict(pcd='mmwave'), splits: list[str] = ['train', 'val', 'test']):
        self.tracker_cfg_f = tracker_cfg_f
        self.dataset = dataset
        self.data_prefix = data_prefix
        self.splits = splits
    
    def make_tracker(self, tracker_cfg_f: str) -> "BaseTracker":
        tracker_cfg = Config.fromfile(tracker_cfg_f).get('tracker_cfg')
        tracker_cfg['radar_cfg'] = dict(
            sensor_height=1.7,
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
            self.generate_tracking_records_single(dataloader)

    def generate_tracking_records_single(self, dataloader: DataLoader):
        tracker = None
        for data in dataloader:
            pcd_frame: np.ndarray = data['pcd_frames'][0][0]
            first_frame: bool = data['starting_flag']
            if first_frame:
                tracker = self.make_tracker(self.tracker_cfg_f)
            tracked_locations = tracker.consume(point_array=pcd_frame)
            print(f'Tracked locations: {tracked_locations}')