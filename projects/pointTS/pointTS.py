import sys
import warnings
from typing import List, Literal, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.device import get_device

from mwpose3d.datasets.skel_data_sample import SkeletonDataSample
from mwpose3d.models.base import BaseSkeletonEstimModel
from mwpose3d.registry import MODELS



@MODELS.register_module()
class PointTSPredictor(BaseSkeletonEstimModel):
    def __init__(self,
                 backbone_cfg: dict,
                 global_feat_dim: int,
                 transformer_cfg: dict,
                 keypoints_involved: List[int],
                 point_cloud_size_per_frame: int,
                 stacked_frames: int,
                 input_channels: int,
                 train_cfg: dict = dict(),
                 test_cfg: dict = dict()
                 ):
        super().__init__()
        self.backbone = MODELS.build(backbone_cfg)
        d_model = global_feat_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_cfg['nhead'],
            dim_feedforward=transformer_cfg['dim_feedforward'],
            dropout=transformer_cfg['dropout'],
            activation=transformer_cfg['activation']
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_cfg['num_layers'],
        )
        self.agg = transformer_cfg.get('agg', 'last')
        self.keypoints_involved = keypoints_involved
        self.fc_out = nn.Linear(d_model, len(keypoints_involved) * 3)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.point_cloud_size = point_cloud_size_per_frame * stacked_frames
        self.input_channels = input_channels

        
    def loss(self, batch_inputs, data_samples):
        tensor = tuple(self._forward(batch_inputs, data_samples).values())[0]
        gt = torch.stack([data_sample.gt for data_sample in data_samples], dim=0)
        loss_dict = dict()
        loss_dict['loss_l2'] = F.mse_loss(tensor, gt)
        return sum(loss_dict.values())/ len(loss_dict)
    
    def predict(self, batch_inputs, data_samples):
        output = self._forward(batch_inputs, data_samples)
        tensor = tuple(output.values())[0]
        for b, data_sample in enumerate(data_samples):
            data_sample.pred = tensor[b]
        return output

    def _forward(self, batch_inputs, data_samples):
        final_pcd_tensor = batch_inputs['final_pcd_tensor']
        B, T, N, C = final_pcd_tensor.size()
        feats = []
        for t in range(T):
            f_t = self.backbone((final_pcd_tensor[:, t, :, :]))
            feats.append(f_t)
        feats = torch.stack(feats, dim=1)
        y = self.transformer(feats)
        if self.agg == 'mean':
            y_agged = y.mean(dim=1)
        elif self.agg == 'last':
            y_agged = y[:, -1, :]
        else:
            raise ValueError(f"Unsupported aggregation method: {self.agg}")
        out = self.fc_out(y_agged)
        # x_out = out.view(B, len(self.keypoints_involved), 3)
        return dict(
            tensor=out,
        )
    
    def pack_input(self, data_batch_dict: dict):
        pcd_frame_list: List[Tuple[np.ndarray]] = data_batch_dict['pcd_frames'] # F x B x N x C
        batch_size = len(pcd_frame_list[0])
        frame_len = len(pcd_frame_list)
        final_pcd_frame = np.zeros((frame_len, batch_size, self.point_cloud_size, self.input_channels
                                   ), dtype=np.float32)
        
        for frame_seq, batched_frames in enumerate(pcd_frame_list):
            for batch_idx, pcd_frame in enumerate(batched_frames):
                final_pcd_frame[frame_seq, batch_idx] = pcd_frame[:self.point_cloud_size]
                
        final_pcd_tensor = torch.from_numpy(final_pcd_frame).float().to(get_device())
        final_pcd_tensor = final_pcd_tensor.permute(1, 0, 2, 3).contiguous() # B x F x N x C
        
        try:
            skel_frame_list: List[Tuple[np.ndarray]] = data_batch_dict['skel_frames']
            
            last_skel_frame = torch.from_numpy(np.stack(skel_frame_list[-1], axis=0)).float().to(get_device())
            data_sample_list = [
                SkeletonDataSample(gt=last_skel_frame_tensor[:(last_skel_frame_tensor.shape[0] // 3) * 3])
                for last_skel_frame_tensor in last_skel_frame
            ]
        except KeyError:
            data_sample_list = [SkeletonDataSample(gt=None) for _ in range(batch_size)]
        data_batch_dict["final_pcd_tensor"] = final_pcd_tensor
        return data_batch_dict, data_sample_list