from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.device import get_device
from dl_engine.dataset.skel_data_sample import SkeletonDataSample
from dl_engine.model.base import MODELS, BaseSkeletonEstimModel

@MODELS.register_module()
class MarsPredictor(BaseSkeletonEstimModel):
    def __init__(self, 
                 point_cloud_size: int = 64, 
                 frame_len: int = 1,
                 in_channels: int= 5,
                 keypoints_involved: List[int] = [x for x in range(20) if x not in [7, 11, 15, 19]]
                 ):
        super().__init__()
        self.point_cloud_size = point_cloud_size
        self.in_channels = in_channels
        self.frame_len = frame_len
        self.keypoints_involved = keypoints_involved
        self.make_layers()

    def make_layers(self):
        self.conv1 = nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16, momentum=0.05)
        self.dropout_conv1 = nn.Dropout2d(0.3)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32, momentum=0.05)
        self.dropout_conv2 = nn.Dropout2d(0.3)

        F, N = self.frame_len, self.point_cloud_size
        self.flatten_dim = 32 * F * N

        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.bn_fc = nn.BatchNorm1d(512, momentum=0.05)
        self.dropout_fc = nn.Dropout(0.4)
        self.fc_out = nn.Linear(512, len(self.keypoints_involved) * 3)

        
    def loss(self, batch_inputs, data_samples):
        tensor = self._forward(batch_inputs, data_samples)
        criterion = nn.MSELoss()
        gt = torch.stack([data_sample.gt for data_sample in data_samples], dim=0)
        loss = criterion(tensor, gt)
        return loss
    
    def predict(self, batch_inputs, data_samples):
        tensor = self._forward(batch_inputs, data_samples)
        for b, data_sample in enumerate(data_samples):
            data_sample.pred = tensor[b]
        return tensor

    def _forward(self, batch_inputs, data_samples):
        x = batch_inputs.permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.dropout_conv1(x)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.dropout_conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc_out(x)
        return x
    
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
        
        
        skel_frame_list: List[Tuple[np.ndarray]] = data_batch_dict['skel_frames']
        skel_frame_tensors = [
           torch.tensor(np.stack(frame_batches, axis=0), dtype=torch.float32, device=get_device()) \
               for frame_batches in list(zip(*skel_frame_list))
        ]
        data_sample_list = [
            SkeletonDataSample(gt=skel_frame_tensor[:, :(skel_frame_tensor.shape[1] // 3) * 3])#.reshape(skel_frame_tensor.shape[0], -1, 3))
            for skel_frame_tensor in skel_frame_tensors
        ]
        data_batch_dict["final_pcd_tensor"] = final_pcd_tensor
        return final_pcd_tensor, data_sample_list
        
            