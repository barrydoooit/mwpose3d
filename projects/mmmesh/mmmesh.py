import sys
import warnings
from typing import List, Literal, Tuple
import numpy as np
import torch
import torch.nn as nn
from mmengine.device import get_device

from mwpose3d.datasets.skel_data_sample import SkeletonDataSample
from mwpose3d.models.base import BaseSkeletonEstimModel
from mwpose3d.registry import MODELS
try:
    from mwpose3d.models.utils.sdtw_cuda_loss import SoftDTW
except Exception as e:
    warnings.warn(f"SoftDTW is not available due to: {e}. Training with SoftDTW will trigger error")



@MODELS.register_module()
class MmMeshPredictor(BaseSkeletonEstimModel):
    def __init__(self,
                 base_pointnet_cfg: dict,
                 global_module_cfg: dict,
                 anchor_module_cfg: dict,
                 fusion_module_cfg: dict,
                 point_cloud_size: int = 32, 
                 frame_len: int = 1,
                 in_channels: int= 6,
                 criterion: str = "MSELoss",
                 train_cfg: dict = dict(warmup_frames=0),
                 test_cfg: dict = dict(serial=False)
                 ):
        super().__init__()
        self.point_cloud_size = point_cloud_size
        self.in_channels = in_channels
        self.frame_len = frame_len
        self.base_pointnet = MODELS.build(base_pointnet_cfg)
        self.global_module = MODELS.build(global_module_cfg)
        self.anchor_module = MODELS.build(anchor_module_cfg)
        self.fusion_module = MODELS.build(fusion_module_cfg)
        self.criterion = criterion

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _make_fusion_module(self):
        fusion_layers = []
        channels = self.fusion_module_cfg.get("channels")
        for i in range(len(channels) - 1):
            fusion_layers.append(nn.Linear(channels[i], channels[i+1]))
            if i != len(channels) - 2:
                fusion_layers.append(nn.ReLU())
        
    def loss(self, batch_inputs, data_samples):
        tensor, _, _, _, _ = tuple(self._forward(batch_inputs, data_samples).values())
        B, T, J = tensor.size()
        gt = torch.stack([data_sample.gt[-T:, :] for data_sample in data_samples], dim=0)
        if self.criterion == "MSELoss":
            criterion = nn.MSELoss()#nn.L1Loss()
            loss = criterion(tensor, gt)
        elif self.criterion == "sdtw":
            sdtw = SoftDTW(use_cuda=True, gamma=0.1)
            mse_loss = nn.MSELoss()(tensor, gt)
            tensor = tensor.reshape(B, T, -1)
            gt = gt.reshape(B, T, -1)
            loss = 0.01 * sdtw(tensor, gt).mean() + mse_loss
        return loss
    
    def predict(self, batch_inputs, data_samples):
        output = self._forward(batch_inputs, data_samples)
        tensor, hn_g, cn_g, hn_a, cn_a = tuple(output.values())
        for b, data_sample in enumerate(data_samples):
            data_sample.pred = tensor[b]
            data_sample.pred = data_sample.pred[-1, :]
            if data_sample.gt is not None:
                data_sample.gt = data_sample.gt[-1, :]
        return output

    def _forward(self, batch_inputs, data_samples):
        final_pcd_tensor = batch_inputs['final_pcd_tensor']
        only_need_recent_frame = batch_inputs.get('only_need_recent_frame', False)
        if only_need_recent_frame:
            final_pcd_tensor = final_pcd_tensor[:, -1:, ...]
        h0_g, c0_g = batch_inputs['h0_g'], batch_inputs['c0_g']
        h0_a, c0_a = batch_inputs['h0_a'], batch_inputs['c0_a']

        B, T, N, C = final_pcd_tensor.size()
        flat = final_pcd_tensor.view(B * T, N, C)
        feat_flat = self.base_pointnet(flat)
        feats = feat_flat.view(B, T, -1) # B x T x C
        warmup_frames = 0 if only_need_recent_frame else self.train_cfg.get('warmup_frames', 0)
        if warmup_frames > 0:
            warm_feats: torch.Tensor = feats[:, :warmup_frames, ...].detach()
            main_feats = feats[:, warmup_frames:, ...]
        else:
            warm_feats = None
            main_feats = feats
        
        warm_feats = warm_feats.reshape(B * warmup_frames, N, -1) if warm_feats is not None else None
        main_feats = main_feats.reshape(B * (T - warmup_frames), N, -1)

        if warmup_frames > 0:
            with torch.no_grad():
                _, g_loc, _, hn_w_g, cn_w_g = self.global_module(warm_feats, h0_g, c0_g, B, warmup_frames)
                _, _, hn_w_a, cn_w_a = self.anchor_module(warm_feats, g_loc, h0_a, c0_a, B, warmup_frames, 28)
        else:
            hn_w_g, cn_w_g = h0_g, c0_g
            hn_w_a, cn_w_a = h0_a, c0_a

        g_vec, g_loc, g_weights, hn_g, cn_g = self.global_module(main_feats, hn_w_g, cn_w_g, B, T - warmup_frames)
        a_vec, a_weights, hn_a, cn_a = self.anchor_module(main_feats, g_loc, hn_w_a, cn_w_a, B, T - warmup_frames, 28)

        x_out = self.fusion_module(g_vec, a_vec, B, T - warmup_frames)

        return dict(
            tensor=x_out,
            hn_g=hn_g,
            cn_g=cn_g,
            hn_a=hn_a,
            cn_a=cn_a
        )
    
    def pack_input(self, data_batch_dict: dict):
        pcd_frame_list: List[Tuple[np.ndarray]] = data_batch_dict['pcd_frames'] # F x B x N x C
        batch_size = len(pcd_frame_list[0])
        frame_len = len(pcd_frame_list)
        final_pcd_frame = np.zeros((frame_len, batch_size, self.point_cloud_size, self.in_channels
                                   ), dtype=np.float32)
        
        for frame_seq, batched_frames in enumerate(pcd_frame_list):
            for batch_idx, pcd_frame in enumerate(batched_frames):
                final_pcd_frame[frame_seq, batch_idx] = pcd_frame[:self.point_cloud_size]
                
        final_pcd_tensor = torch.from_numpy(final_pcd_frame).float().to(get_device())
        final_pcd_tensor = final_pcd_tensor.permute(1, 0, 2, 3).contiguous() # B x F x N x C
        
        try:
            skel_frame_list: List[Tuple[np.ndarray]] = data_batch_dict['skel_frames']
            skel_frame_tensors = [
            torch.tensor(np.stack(frame_batches, axis=0), dtype=torch.float32, device=get_device()) \
                for frame_batches in list(zip(*skel_frame_list))
            ]
            data_sample_list = [SkeletonDataSample(gt=skel_frame_tensor) for skel_frame_tensor in skel_frame_tensors]
        except KeyError:
            data_sample_list = [SkeletonDataSample(gt=None) for _ in range(batch_size)]

        if 'previous_output' in data_batch_dict and \
            (not data_batch_dict.get('starting_flag', [False])[0]) and \
            self.test_cfg.get('serial', False):
            prev = data_batch_dict.pop('previous_output')
            h0_g = prev.get('hn_g')
            c0_g = prev.get('cn_g')
            h0_a = prev.get('hn_a')
            c0_a = prev.get('cn_a')
            assert h0_g is not None and c0_g is not None and \
                h0_a is not None and c0_a is not None, "Previous output should contain all hidden states"
            batch_inputs = dict(
                final_pcd_tensor=final_pcd_tensor,
                h0_g=h0_g, c0_g=c0_g,
                h0_a=h0_a, c0_a=c0_a,
                only_need_recent_frame=True
            )
        else:
            if not self.global_module.grnn.learnable_init_state:
                h0_g = torch.zeros((self.global_module.grnn.num_layers, batch_size, self.global_module.grnn.hidden_size), dtype=torch.float32, device=get_device())
                c0_g = torch.zeros((self.global_module.grnn.num_layers, batch_size, self.global_module.grnn.hidden_size), dtype=torch.float32, device=get_device())
            else:
                h0_g, c0_g = None, None
            
            if not self.anchor_module.arnn.learnable_init_state:
                h0_a = torch.zeros((self.anchor_module.arnn.num_layers, batch_size, self.anchor_module.arnn.hidden_size), dtype=torch.float32, device=get_device())
                c0_a = torch.zeros((self.anchor_module.arnn.num_layers, batch_size, self.anchor_module.arnn.hidden_size), dtype=torch.float32, device=get_device())
            else:
                h0_a, c0_a = None, None
            
            batch_inputs = dict(
                final_pcd_tensor=final_pcd_tensor,
                h0_g=h0_g,
                c0_g=c0_g,
                h0_a=h0_a,
                c0_a=c0_a
            )
        return batch_inputs, data_sample_list
        
    def pack_input_online(self, data_batch_dict: dict):
        raise NotImplementedError("Online mode is out of maintenance")
        pcd_frame_list: List[Tuple[np.ndarray]] = data_batch_dict['pcd_frames']
        num_recent_frames = data_batch_dict.get('num_recent_frames', 1)
        pcd_frame_list = pcd_frame_list[-num_recent_frames:]
        batch_size = len(pcd_frame_list[0])
        assert batch_size == 1, "Online mode of model only supports batch size 1"
        final_pcd_frame = np.zeros((num_recent_frames, batch_size, self.point_cloud_size, self.in_channels), dtype=np.float32)
        
        for frame_seq, batched_frames in enumerate(pcd_frame_list):
            for batch_idx, pcd_frame in enumerate(batched_frames):
                point_count, _ = pcd_frame.shape
                pcd_frame[:, -1] = (pcd_frame[:, -1] - self.intensity_norm[0]) / self.intensity_norm[1]
                if point_count < self.point_cloud_size:
                    raise ValueError("Point cloud should be padded in advance")
                    padding = np.zeros((self.point_cloud_size - point_count, self.in_channels))
                    pcd_frame = np.concatenate([pcd_frame, padding], axis=0) 
                else:
                    pcd_frame = pcd_frame[:self.point_cloud_size]
                sorted_indices = np.argsort(pcd_frame[:, self.sort_dim])
                final_pcd_frame[frame_seq, batch_idx] = pcd_frame[sorted_indices]
        
        final_pcd_tensor = torch.from_numpy(final_pcd_frame).float().to(get_device())
        final_pcd_tensor = final_pcd_tensor.permute(1, 0, 2, 3).contiguous() # B x F x N x C
        
        if all(k in data_batch_dict for k in ['h0_g', 'c0_g', 'h0_a', 'c0_a']):
            h0_g = data_batch_dict['h0_g']
            c0_g = data_batch_dict['c0_g']
            h0_a = data_batch_dict['h0_a']
            c0_a = data_batch_dict['c0_a']
        else:
            h0_g = torch.zeros((3, batch_size, self.global_module.grnn.in_channel), dtype=torch.float32, device=get_device())
            c0_g = torch.zeros((3, batch_size, self.global_module.grnn.in_channel), dtype=torch.float32, device=get_device())
            h0_a = torch.zeros((3, batch_size, self.anchor_module.arnn.input_size), dtype=torch.float32, device=get_device())
            c0_a = torch.zeros((3, batch_size, self.anchor_module.arnn.input_size), dtype=torch.float32, device=get_device())
        
        batch_inputs = dict(
            final_pcd_tensor=final_pcd_tensor,
            h0_g=h0_g,
            c0_g=c0_g,
            h0_a=h0_a,
            c0_a=c0_a
        )
        data_batch_dict["final_pcd_tensor"] = final_pcd_tensor
        return batch_inputs, [SkeletonDataSample(gt=None) for _ in range(final_pcd_tensor.shape[0])]