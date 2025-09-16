import math
from typing import List, Literal, Tuple, TYPE_CHECKING
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.device import get_device
from mmcv.ops.voxelize import Voxelization
from mwpose3d.datasets.skel_data_sample import SkeletonDataSample
from mwpose3d.models.base import BaseSkeletonEstimModel

from mwpose3d.registry import MODELS
try:
    from mwpose3d.models.utils.sdtw_cuda import SoftDTW
except Exception as e:
    warnings.warn(f"SoftDTW is not available due to: {e}. Training with SoftDTW will trigger error")
    
if TYPE_CHECKING:
    from mwpose3d.models.middle_encoders.sparse_encoder import SparseEncoder



@MODELS.register_module()
class RadHARCNNBiLSTM(BaseSkeletonEstimModel):
    def __init__(self,
                 num_frames: int,
                 voxel_size: list[float],
                 point_cloud_range: list[float], # [x_min, y_min, z_min, x_max, y_max, z_max]
                 moddle_encoder: dict,
                 backbone: dict,
                 lstm_cfg: dict,
                 head_channels: Tuple[int,] = (1024, 512, 256),
                 max_num_points: int = 128,
                 max_voxels: int = 20000,
                 voxelize_reduce: bool = True,
                 keypoints_involved: List[int] = list(range(0, 21)),
                 criterion: Literal['MSELoss', 'CrossEntropyLoss', 'sdtw'] = 'sdtw',
                 train_cfg: dict = dict(splits=2),
                 test_cfg: dict = dict(cache_feats=True, splits=1),
                 ):
        super().__init__()
        self.middle_encoder: 'SparseEncoder' = MODELS.build(moddle_encoder)
        self.backbone = MODELS.build(backbone)

        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.voxelize_reduce = voxelize_reduce
        self.voxel_layer = Voxelization(
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels
        )

        self.num_frames = num_frames
        self.num_joints = len(keypoints_involved)
        self.lstm_cfg = lstm_cfg

        if self.lstm_cfg.pop('learnable_init_state', False):
            self.h0 = nn.Parameter(torch.zeros(self.lstm_cfg['num_layers'] * (2 if self.lstm_cfg.get('bidirectional', False) else 1), 1, self.lstm_cfg['hidden_size']))
            self.c0 = nn.Parameter(torch.zeros(self.lstm_cfg['num_layers'] * (2 if self.lstm_cfg.get('bidirectional', False) else 1), 1, self.lstm_cfg['hidden_size']))
            self.learnable_init_state = True
        else:
            self.h0 = None
            self.c0 = None
            self.learnable_init_state = False
        
        self.lstm = nn.LSTM(**self.lstm_cfg)

        self.joints_head = [nn.Dropout(0.3)]
        head_channels = [*head_channels, self.num_joints * 3]
        for i in range(len(head_channels) - 1):
            self.joints_head.append(nn.Linear(head_channels[i], head_channels[i + 1]))
            if i != len(head_channels) - 2:
                self.joints_head.append(nn.ReLU())
                self.joints_head.append(nn.Dropout(0.1))
        self.joints_head = nn.Sequential(*self.joints_head)

        self.sdtw = False
        if criterion == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif criterion == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        elif criterion == 'sdtw':
            self.criterion = [SoftDTW(gamma=0.1), nn.MSELoss()]
            self.sdtw = True

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @torch.no_grad()
    def voxelize(self, points_list):
        """
        points_list: List[Tensor] of length B, each (N_i, C_in)
        Returns:
            feats: (M, C_in) concatenated
            coords: (M, 4) with batch idx padded
            sizes (optional): num points per voxel
        """
        feats, coords, sizes = [], [], []
        for b_idx, pts in enumerate(points_list):
            ret = self.voxel_layer(pts)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=b_idx))
            if n is not None:
                sizes.append(n)
        
        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()
        return feats, coords, sizes
        
    def loss(self, batch_inputs, data_samples):
        tensor = tuple(self._forward(batch_inputs, data_samples).values())[0]
        gt = torch.stack([data_sample.gt for data_sample in data_samples], dim=0)
        if not isinstance(self.criterion, list):
            criterion = self.criterion
            loss = criterion(tensor, gt)
        else:
            B, T = tensor.shape[:2]
            mse_loss = self.criterion[1](tensor, gt)
            tensor = tensor.reshape(B, T, -1)
            gt = gt.reshape(B, T, -1)
            loss = 0.01 * self.criterion[0](tensor, gt).mean() + mse_loss
        return loss
    
    def predict(self, batch_inputs, data_samples):
        output = self._forward(batch_inputs, data_samples)
        tensor = tuple(output.values())[0]
        for b, data_sample in enumerate(data_samples):
            data_sample.pred = tensor[b]
            data_sample.pred = data_sample.pred[-1, :]
            if data_sample.gt is not None:
                data_sample.gt = data_sample.gt[-1, :]
        return output

    def _forward(self, batch_inputs: dict, data_samples: List[SkeletonDataSample]):
        points_seq = batch_inputs['points']
        T = len(points_seq)
        B = len(points_seq[0])
        if not self.training \
            and self.test_cfg.get('cache_feats', False)\
            and 'cached_feats' in batch_inputs:
            cached_feats = batch_inputs.get('cached_feats')
            last_pts = points_seq[-1]
            feats, coords, _ = self.voxelize(last_pts)
            spatial = self.middle_encoder(feats, coords, B)
            seq_feats = self.backbone(spatial)[-1].view(B, -1).unsqueeze(1)
            seq_feats = torch.cat((cached_feats, seq_feats), dim=1)
            
        else:
            split = self.train_cfg.get('splits', 1) if self.training else self.test_cfg.get('splits', 1)
            base_size, r = divmod(T, split)
            sizes = [base_size + (1 if i < r else 0) for i in range(split)]
            seq_feats_list = []
            idx = 0

            for sz in sizes:
                flat_pc = [
                    points_seq[t][b]
                    for b in range(B)
                    for t in range(idx, idx + sz)
                ]

                feats, coords, _ = self.voxelize(flat_pc)
                spatial = self.middle_encoder(feats, coords, B * sz)
                seq_feats = self.backbone(spatial)[-1].view(B, sz, -1)
                seq_feats_list.append(seq_feats)
                idx += sz
            seq_feats = torch.cat(seq_feats_list, dim=1)
        # print(seq_feats[:, :, 0].flatten())
        h0 = batch_inputs.get('h0', self.h0.expand(-1, B, -1).contiguous())
        c0 = batch_inputs.get('c0', self.c0.expand(-1, B, -1).contiguous())
        lstm_out, (hn, cn) = self.lstm(seq_feats, (h0, c0))
        flat = lstm_out.reshape(B * T, -1)
        logits = self.joints_head(flat)
        logits = logits.view(B, T, self.num_joints * 3)
        return dict(
            tensor=logits,
            hn=hn,
            cn=cn,
            prev_feats=seq_feats,  # for caching
        )

    def pack_input(self, data_batch_dict: dict, training: bool = True):
        pcd_frame_list: List[Tuple[np.ndarray]] = data_batch_dict['pcd_frames'] # F x B x N x C
        T = len(pcd_frame_list)
        B = len(pcd_frame_list[0])
        points = []
        for s in range(T):
            sample_frames = []
            for b in range(B):
                pcd_frame = pcd_frame_list[s][b]
                tensor_pts = torch.from_numpy(pcd_frame).float().to(get_device())
                sample_frames.append(tensor_pts)
            points.append(sample_frames)
        
        try:
            skel_frame_list: List[Tuple[np.ndarray]] = data_batch_dict['skel_frames']
            skel_frame_tensors = [
            torch.tensor(np.stack(frame_batches, axis=0), dtype=torch.float32, device=get_device()) \
                for frame_batches in list(zip(*skel_frame_list))
            ]
            data_sample_list = [SkeletonDataSample(gt=skel_frame_tensor) for skel_frame_tensor in skel_frame_tensors]
        except KeyError:
            data_sample_list = [SkeletonDataSample(gt=None) for _ in range(B)]
        
        batch_inputs = dict(
            points=points,
        )
        if 'previous_output' in data_batch_dict \
            and len(data_batch_dict['previous_output']) > 0 \
            and self.test_cfg.get('cache_feats', False) \
            and (not data_batch_dict.get('starting_flag', [False])[0]):
            prev_feat = data_batch_dict.pop('previous_output')["prev_feats"][:, 1:, ...]
            batch_inputs = dict(
                batch_inputs,
                cached_feats=prev_feat,
            )
        if not self.learnable_init_state:
            batch_inputs['h0'] = torch.zeros((self.lstm_cfg['num_layers'] * (2 if self.lstm_cfg.get('bidirectional', False) else 1), B, self.lstm_cfg['hidden_size']), dtype=torch.float32, device=get_device())
            batch_inputs['c0'] = torch.zeros((self.lstm_cfg['num_layers'] * (2 if self.lstm_cfg.get('bidirectional', False) else 1), B, self.lstm_cfg['hidden_size']), dtype=torch.float32, device=get_device())
        return batch_inputs, data_sample_list
