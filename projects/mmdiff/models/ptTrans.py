# NEW: mmcoarse.py

from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmengine.device import get_device

from mwpose3d.datasets.skel_data_sample import SkeletonDataSample
from mwpose3d.models.base import BaseSkeletonEstimModel
from mwpose3d.registry import MODELS


@MODELS.register_module()
class PtTransPredictor(BaseSkeletonEstimModel):
    def _check_model_validty(self):
        if self.model_feat is not None:
            dummy = torch.rand(self.input_shape, dtype=torch.float32, device=get_device())
            self.model_feat.to(get_device())
            output = self.model_feat(dummy)
            for i in range(2):
                if tuple(output[i].size()) != tuple(self.output_shape[i]):
                    raise RuntimeError(
                        f"Desired [{i}] output shape of self.model_feat is {self.output_shape[i]}, "
                        f"received {list(output[i].size())}"
                    )
            print("self.model_feat validity passes.")

    def __init__(self,
                 point_cloud_size: int,
                 keypoints_involved: List[int],
                 feature_extractor: dict,
                 seq_frames: int,
                 radar_input_c: int,
                 input_shape: Tuple[int, int, int, int],     # (B, T, N, C)
                 output_shape: Tuple[Tuple[int, int, int],   # (B, J, 3)
                                     Tuple[int, int, int]],  # (B, J, F)   F = mid_dim of extractor
                 gt_norm_joint: Optional[int] = None,
                 test_cfg: dict = dict()):
        super().__init__()
        self.point_cloud_size = point_cloud_size
        self.keypoints_involved = keypoints_involved
        self.num_joints = len(keypoints_involved)

        self.model_feat = MODELS.build(feature_extractor)
        self.input_shape = deepcopy(input_shape)
        self.output_shape = deepcopy(output_shape)
        self.radar_input_c = radar_input_c
        assert self.input_shape[3] == radar_input_c

        self._check_model_validty()

        self.seq_frames = seq_frames
        # joint to normalize GT against (optional)
        self.joint2idx = {j: i for i, j in enumerate(keypoints_involved)}
        self.gt_norm_joint = self.joint2idx[gt_norm_joint] if gt_norm_joint is not None else None

        self.criterion = nn.MSELoss()
        self.test_cfg = test_cfg or {}

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[SkeletonDataSample]] = None,
                mode: str = 'tensor'):
        if mode in ['loss', 'loss-pretrain', 'loss-train']:
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            return self._forward(inputs, data_samples)

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def _extract_current(self, pcd: torch.Tensor):
        pcd_seq = pcd[:, -self.seq_frames:, :, :]
        joints_predict, joint_emb = self.model_feat(pcd_seq)
        return joints_predict, joint_emb

    def _maybe_denorm(self, skel_frame: torch.Tensor) -> torch.Tensor:
        if self.gt_norm_joint is None:
            return skel_frame
        skel_frame = skel_frame.reshape(-1, 3)
        reference = skel_frame[self.gt_norm_joint].clone()
        skel_frame = skel_frame + reference
        skel_frame[self.gt_norm_joint] = reference
        return skel_frame.flatten()

    def loss(self, batch_inputs, data_samples, **kwargs):
        preds, _ = self._extract_current(batch_inputs['final_pcd_tensor'])  # (B, J, 3)
        gt = torch.stack([ds.gt[-1] for ds in data_samples], dim=0)         # (B, J, 3)
        preds = preds.to(get_device())
        loss = self.criterion(preds, gt)
        return loss

    @torch.no_grad()
    def predict(self, batch_inputs, data_samples):
        preds, feats = self._extract_current(batch_inputs['final_pcd_tensor'])
        for b, ds in enumerate(data_samples):
            ds.pred = preds[b].flatten()
            if ds.gt is not None:
                ds.gt = ds.gt[-1, ...].flatten()
        return dict(coarse_pr_list=[preds])   # keep mmDiff-style key

    def pack_input(self, data_batch_dict: dict, training: bool = True):
        pcd_frame_list: List[Tuple[np.ndarray]] = data_batch_dict['pcd_frames']  # F x B x N x C
        batch_size = len(pcd_frame_list[0])
        frame_len = len(pcd_frame_list)

        final_pcd_frame = np.zeros((frame_len, batch_size, self.point_cloud_size, self.radar_input_c), dtype=np.float32)
        for frame_seq, batched_frames in enumerate(pcd_frame_list):
            for batch_idx, pcd_frame in enumerate(batched_frames):
                final_pcd_frame[frame_seq, batch_idx] = pcd_frame[:self.point_cloud_size]
        final_pcd_tensor = torch.from_numpy(final_pcd_frame).float().to(get_device())
        final_pcd_tensor = final_pcd_tensor.permute(1, 0, 2, 3).contiguous()  # (B, F, N, C)

        try:
            skel_frame_list: List[Tuple[np.ndarray]] = data_batch_dict['skel_frames']
            skel_frame_tensors = [
                torch.tensor(np.stack(frame_batches, axis=0), dtype=torch.float32, device=get_device())
                for frame_batches in list(zip(*skel_frame_list))
            ]
            data_sample_list = [
                SkeletonDataSample(gt=skel_tensor[:, :(skel_tensor.shape[1] // 3) * 3]
                                   .reshape(skel_tensor.shape[0], -1, 3))
                for skel_tensor in skel_frame_tensors
            ]
        except KeyError:
            data_sample_list = [SkeletonDataSample(gt=None) for _ in range(batch_size)]

        batch_inputs = dict(final_pcd_tensor=final_pcd_tensor)
        data_batch_dict.update(batch_inputs)
        return batch_inputs, data_sample_list
