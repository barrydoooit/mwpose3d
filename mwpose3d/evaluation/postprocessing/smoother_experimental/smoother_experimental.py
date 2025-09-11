from collections import deque
from typing import Sequence
import numpy as np
import torch

from mwpose3d.datasets.transforms.base import OnlineEnabled
from .cuda_pose_smoother import (
    PoseSmoother, PoseSmootherConfig,
    GaussianEMASmoother, GaussianEMASmootherConfig
)
from ..base import BasePostProcessing
from ..base import POSTPROCESSING


@OnlineEnabled
@POSTPROCESSING.register_module()
class ExperimentalSmoother(BasePostProcessing):
    def __init__(self, 
                 keypoints_involved: Sequence[int], 
                 idx_spine_shoulder_elbow_wrist_lr: Sequence[int],
                 online_mode: bool = False, 
                 smoother_kwargs: dict = dict(),
                 smoother_type: str = "lock"  # "lock" (original) or "gaussian_ema"
                 ):
        super().__init__(online_mode)

        # Choose the smoother family
        if smoother_type == "gaussian_ema":
            self.config = GaussianEMASmootherConfig(dt=1/18)
            for k, v in smoother_kwargs.items():
                setattr(self.config, k, v)
            self.smoother = GaussianEMASmoother(N_joints=len(keypoints_involved), cfg=self.config)
        else:
            self.config = PoseSmootherConfig(dt=1/18)
            for k, v in smoother_kwargs.items():
                setattr(self.config, k, v)
            self.smoother = PoseSmoother(N_joints=len(keypoints_involved), cfg=self.config)

        mapped = []
        for ki, key_point in enumerate(['spine', 'mid', 'neck', 'shoulder_L', 'shoulder_R', 'elbow_L', 'elbow_R', 'wrist_L', 'wrist_R']):
            assert idx_spine_shoulder_elbow_wrist_lr[ki] in keypoints_involved
            indice_in_frame = keypoints_involved.index(idx_spine_shoulder_elbow_wrist_lr[ki])
            mapped.append(indice_in_frame)

        self.smoother.set_indices(*mapped)

    def transform(self, data_batch_dict, datasample):
        orig = datasample.pred
        try:
            datasample.pred = self.smoother.filter(orig)
            return data_batch_dict, datasample
        except Exception as e:
            print(f"Error occurred while smoothing: {e}")
            return data_batch_dict, datasample
