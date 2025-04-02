from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import torch
from mmengine.device import get_device

from apps.common.pcd.pc_buffer import PointCloudBuffer
from dl_engine.dataset.preprocess import WITH_ONLINE_FUNCTIONALITY, TRANSFORM, BaseTransform
from dl_engine.model.base import MODELS

if TYPE_CHECKING:
    from apps.common.pcd.pointCloud import SimplePointCloud5D




class InferenceEngine:
    def __init__(self,
                 model: dict,
                 pipeline: List[dict],
                 load_from: str,
                 keypoints_involved: List[int],
                 ):
        self.model: torch.nn.Module = MODELS.build(model)
        self.model.to(get_device())
        self.pipeline = self._make_pipeline(pipeline)
        self.model.load_state_dict(torch.load(load_from, map_location=torch.device(get_device())))
        self.model.eval()
        self.output: dict = None
        self.keypoints_involved = keypoints_involved
        
    def _make_pipeline(self, pipeline: List[dict]) -> Tuple[BaseTransform, ...]:
        pipeline = [
            transform for transform in pipeline if transform['type'] in WITH_ONLINE_FUNCTIONALITY
        ]
        pipeline = [TRANSFORM.build(dict(transform, online_mode=True)) for transform in pipeline]
        return tuple(pipeline)
    
    def _preprocess(self, point_cloud: 'SimplePointCloud5D') -> dict:
        input_dict = {
            'pcd_frames': (np.asarray(point_cloud.serialize(compact=True)),),
            'num_recent_frames': 1
            }
        for transform in self.pipeline:
            input = transform(input_dict)
        processed_pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        input['pcd_frames'] = tuple([
            np.expand_dims(pcd_frame, axis=0) for pcd_frame in processed_pcd_frames
        ]) # make a batch dimension
        return input
    
    def infer(self, point_cloud: 'SimplePointCloud5D'):
        data_batch_dict = self._preprocess(point_cloud)
        if self.output is not None:
            data_batch_dict.update(dict(self.output))
        data_batch_dict.pop('tensor', None)
        batch_inputs, data_samples = self.model.pack_input_online(data_batch_dict)
        with torch.no_grad():
            self.output = self.model(batch_inputs, data_samples, mode='predict')
        return self.output
    
    @classmethod
    def from_cfg(cls, config: dict):
        return cls(
            model=config['model'],
            pipeline=config['test_pipeline'],
            load_from=config['load_from'],
            keypoints_involved=config['keypoint_involved']
        )