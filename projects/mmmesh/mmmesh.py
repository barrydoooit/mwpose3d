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
except Exception:
    warnings.warn("SoftDTW is not available. Training with SoftDTW will trigger error")



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
                 sort_dim: int = -1,
                 sort_order: Literal["asc", "desc"] = "desc",
                 intensity_norm: tuple = (22.876, 5.058),
                 keypoints_involved: List[int] = [x for x in range(20) if x not in [7, 11, 15, 19]],
                 criterion: str = "MSELoss"
                 ):
        super().__init__()
        self.point_cloud_size = point_cloud_size
        self.in_channels = in_channels
        self.frame_len = frame_len
        self.intensity_norm = intensity_norm
        self.sort_dim = sort_dim
        self.sort_order = 1 if sort_order == "asc" else -1
        self.keypoints_involved = keypoints_involved
        self.base_pointnet = MODELS.build(base_pointnet_cfg)
        self.global_module = MODELS.build(global_module_cfg)
        self.anchor_module = MODELS.build(anchor_module_cfg)
        self.fusion_module = MODELS.build(fusion_module_cfg)
        self.criterion = criterion

    def _make_fusion_module(self):
        fusion_layers = []
        channels = self.fusion_module_cfg.get("channels")
        for i in range(len(channels) - 1):
            fusion_layers.append(nn.Linear(channels[i], channels[i+1]))
            if i != len(channels) - 2:
                fusion_layers.append(nn.ReLU())
        
    def loss(self, batch_inputs, data_samples):
        tensor, _, _, _, _ = tuple(self._forward(batch_inputs, data_samples).values())
        gt = torch.stack([data_sample.gt for data_sample in data_samples], dim=0)
        if self.criterion == "MSELoss":
            criterion = nn.MSELoss()#nn.L1Loss()
            loss = criterion(tensor, gt)
        elif self.criterion == "sdtw":
            sdtw = SoftDTW(use_cuda=True, gamma=0.1)
            batch_size, length_size, _, _ = batch_inputs['final_pcd_tensor'].size()
            mse_loss = nn.MSELoss()(tensor, gt)
            tensor = tensor.reshape(batch_size, length_size, -1)
            gt = gt.reshape(batch_size, length_size, -1)
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
        h0_g, c0_g = batch_inputs['h0_g'], batch_inputs['c0_g']
        h0_a, c0_a = batch_inputs['h0_a'], batch_inputs['c0_a']
        batch_size, length_size, point_count, in_channels = final_pcd_tensor.size()
        
        x = final_pcd_tensor.view(batch_size * length_size, point_count, in_channels)
        x = self.base_pointnet(x)
        g_vec, g_loc, global_weights, hn_g, cn_g = self.global_module(x, h0_g, c0_g, batch_size, length_size)
        a_vec, anchor_weights, hn_a, cn_a = self.anchor_module(x, g_loc, h0_a, c0_a, batch_size, length_size, 28)
        x = self.fusion_module(g_vec, a_vec, batch_size, length_size)
        return dict(
            tensor=x,
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
                point_count, _ = pcd_frame.shape    
                if point_count < self.point_cloud_size:
                    raise ValueError("Point cloud should be padded in advance")
                    padding = np.zeros((self.point_cloud_size - point_count, self.in_channels))
                    pcd_frame = np.concatenate([pcd_frame, padding], axis=0) 
                else:
                    sorted_indices = np.argsort(self.sort_order * pcd_frame[:, self.sort_dim])
                    pcd_frame = pcd_frame[sorted_indices]
                    pcd_frame[:, -1] = (pcd_frame[:, -1] - self.intensity_norm[0]) / self.intensity_norm[1]
                final_pcd_frame[frame_seq, batch_idx] = pcd_frame[:self.point_cloud_size]
                
        final_pcd_tensor = torch.from_numpy(final_pcd_frame).float().to(get_device())
        final_pcd_tensor = final_pcd_tensor.permute(1, 0, 2, 3).contiguous() # B x F x N x C
        
        
        skel_frame_list: List[Tuple[np.ndarray]] = data_batch_dict['skel_frames']
        skel_frame_tensors = [
           torch.tensor(np.stack(frame_batches, axis=0), dtype=torch.float32, device=get_device()) \
               for frame_batches in list(zip(*skel_frame_list))
        ]
        data_sample_list = [SkeletonDataSample(gt=skel_frame_tensor) for skel_frame_tensor in skel_frame_tensors]
        
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
        return batch_inputs, data_sample_list
        
    def pack_input_online(self, data_batch_dict: dict):
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