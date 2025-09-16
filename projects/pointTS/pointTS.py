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
                 test_cfg: dict = dict(serial_test=True)
                 ):
        super().__init__()
        self.backbone = MODELS.build(backbone_cfg)
        d_model = global_feat_dim
        encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True,
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

        self._reset_feat_buffer()

    def _reset_feat_buffer(self):
        self._feat_buf = None
        self._feat_len = 0
        self._feat_wptr = 0
        self._win_T = None

    def _update_feat_buffer(self, feats: torch.Tensor, starting: bool) -> torch.Tensor:
        """
        feats: [B, t, F]
        - on starting: optionally reset/choose window (kept simple here)
        - returns a chronological view [B, L_used, F]
        """
        if starting:
            # set a stable window size at sequence start if desired (optional)
            # self._win_T = int(self.test_cfg.get('window', max(2, feats.size(1))))
            self._feat_buf = None
            self._feat_len = 0
            self._feat_wptr = 0

        B, t, F = feats.shape
        device, dtype = feats.device, feats.dtype

        if self._win_T is None:
            # fallback: infer window if not set yet
            self._win_T = t if t > 1 else 1

        # allocate / reallocate when shape or window changes
        need_alloc = (
            self._feat_buf is None or
            self._feat_buf.size(0) != B or
            self._feat_buf.size(1) != self._win_T or
            self._feat_buf.size(2) != F
        )
        if need_alloc:
            self._feat_buf = torch.empty(B, self._win_T, F, device=device, dtype=dtype)
            self._feat_len = 0
            self._feat_wptr = 0

        # multi-frame warm-up / refresh path
        if t > 1:
            take = min(t, self._win_T)
            window = feats[:, -take:, :]
            if take == self._win_T:
                self._feat_buf.copy_(window)
                self._feat_len = self._win_T
                self._feat_wptr = 0
                return self._feat_buf
            else:
                self._feat_buf[:, :take, :].copy_(window)
                self._feat_len = take
                self._feat_wptr = 0
                return self._feat_buf[:, :self._feat_len, :]

        # ---------- streaming single-frame path (t == 1) ----------
        frame = feats[:, 0:1, :]

        # buffer not yet full: append at end
        if self._feat_len < self._win_T:
            idx = self._feat_len
            self._feat_buf[:, idx:idx+1, :].copy_(frame)
            self._feat_len += 1
            return self._feat_buf[:, :self._feat_len, :]

        # buffer full: write into the *oldest* slot then build view starting at (oldest+1)
        old_wptr = self._feat_wptr
        self._feat_buf[:, old_wptr:old_wptr+1, :].copy_(frame)

        # the chronological sequence should start at the element after the one we just overwrote
        start = (old_wptr + 1) % self._win_T

        if start == 0:
            view = self._feat_buf
        else:
            view = torch.cat([
                self._feat_buf[:, start:, :],
                self._feat_buf[:, :start, :]
            ], dim=1)

        # now advance the write pointer to the new oldest slot
        self._feat_wptr = start

        return view
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
        starting = bool(batch_inputs.get('starting_flag', True))
        B, T, N, C = final_pcd_tensor.size()
        feats_list = []
        for t in range(T):
            f_t = self.backbone(final_pcd_tensor[:, t, :, :])   # [B, d_model]
            feats_list.append(f_t)
        feats = torch.stack(feats_list, dim=1)                   # [B, T, d_model]
        serial = self.test_cfg.get('serial_test', False) and not self.training

        if serial:
            seq_for_tx = self._update_feat_buffer(feats, starting=starting)
        else:
            # Non-serial: use the fresh sequence directly (no buffer side-effects)
            seq_for_tx = feats
        y = self.transformer(seq_for_tx)                        # [B, T, d_model]
        if self.agg == 'mean':
            y_agged = y.mean(dim=1)
        elif self.agg == 'last':
            y_agged = y[:, -1, :]
        else:
            raise ValueError(f"Unsupported aggregation method: {self.agg}")
        out = self.fc_out(y_agged)
        # x_out = out.view(B, len(self.keypoints_involved), 3)
        return dict(tensor=out,)
    
    def pack_input(self, data_batch_dict: dict, training: bool = True):
        pcd_frame_list: List[Tuple[np.ndarray]] = data_batch_dict['pcd_frames'] # F x B x N x C
        batch_size = len(pcd_frame_list[0])
        frame_len = len(pcd_frame_list)

        
        starting = bool(data_batch_dict.get('starting_flag', [True])[0])
        serial = self.test_cfg.get('serial_test', False) and not training
        if serial and not starting:
            pcd_frame_list = pcd_frame_list[-1:]
            frame_len = 1
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
        batch_inputs = dict(
            data_batch_dict,
            final_pcd_tensor=final_pcd_tensor,
            starting_flag=starting,   # NEW: control buffer reset
        )
        return batch_inputs, data_sample_list