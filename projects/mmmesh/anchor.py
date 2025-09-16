import copy
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from mmengine.device import get_device
from . import utils as msh_utils
from mwpose3d.registry import MODELS



@MODELS.register_module()
class AnchorPointNet(nn.Module):
    def __init__(self,
                 channels: List[int] = [31, 32, 48, 64],
                 kernel_size: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self._make_layers()
    
    def _make_layers(self):
        self.conv1 = nn.Conv1d(self.channels[0], self.channels[1], self.kernel_size)
        self.cb1 = nn.BatchNorm1d(self.channels[1])
        self.caf1 = nn.ReLU()
        self.conv2 = nn.Conv1d(self.channels[1], self.channels[2], self.kernel_size)
        self.cb2 = nn.BatchNorm1d(self.channels[2])
        self.caf2 = nn.ReLU()
        self.conv3 = nn.Conv1d(self.channels[2], self.channels[3], self.kernel_size)
        self.cb3 = nn.BatchNorm1d(self.channels[3])
        self.caf3 = nn.ReLU()
        
        self.attn = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))
        x = x.transpose(1, 2)
        attn_weights = self.softmax(self.attn(x))
        attn_vec = torch.sum(attn_weights * x, dim=1)
        return attn_vec, attn_weights

@MODELS.register_module()
class AnchorVoxelNet(nn.Module):
    def __init__(self,
                 channels: List[int] = [64, 96, 128, 64],
                 kernel_size: Tuple[Tuple[int, int, int]] =(
                     (3,3,3), (5,1,1),(3,1,1), # (9 3 3) -> (7, 1, 1) -> (3, 1, 1) -> (1, 1, 1), when voxel size is (9, 3, 3)
                 ),
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self._make_layers()
    
    def _make_layers(self):
        self.conv1 = nn.Conv3d(self.channels[0], self.channels[1], self.kernel_size[0], padding=(0,0,0))
        self.cb1 = nn.BatchNorm3d(self.channels[1])
        self.caf1 = nn.ReLU()
        self.conv2 = nn.Conv3d(self.channels[1], self.channels[2], self.kernel_size[1])
        self.cb2 = nn.BatchNorm3d(self.channels[2])
        self.caf2 = nn.ReLU()
        self.conv3 = nn.Conv3d(self.channels[2], self.channels[3], self.kernel_size[2])
        self.cb3 = nn.BatchNorm3d(self.channels[3])
        self.caf3 = nn.ReLU()
    
    def forward(self, x):
        batch_size = x.size()[0]
        x = x.permute(0, 4, 1, 2, 3)
        
        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))
        
        x = x.view(batch_size, self.channels[-1])
        return x

class AnchorRNN(nn.Module):
    def __init__(self,
                 input_size: int = 64,
                 hidden_size: int = 64,
                 num_layers: int = 3,
                 batch_first: bool = True,
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 learnable_init_state: bool = False,
                 use_carry_init: bool = False,
                 store_x: bool = False
                 ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.learnable_init_state = learnable_init_state
        self.use_carry_init = use_carry_init
        self.store_x = store_x

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=batch_first,
                           dropout=dropout,
                           bidirectional=bidirectional)

        dir_mult = 2 if bidirectional else 1
        self.num_layers = num_layers * dir_mult

        if learnable_init_state:
            self.h0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))
            self.c0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))
        else:
            self.h0 = None
            self.c0 = None
        self._x_buf = None
        self._carry_h0 = None 
        self._carry_c0 = None
    
    def _init_states(self, x, h0, c0):
        if h0 is not None and c0 is not None:
            return h0, c0
        B = x.size(0)
        dir_mult = 2 if self.bidirectional else 1
        if self.learnable_init_state:
            h0 = self.h0.expand(self.num_layers, B, self.hidden_size).contiguous()
            c0 = self.c0.expand(self.num_layers, B, self.hidden_size).contiguous()
        else:
            h0 = x.new_zeros(self.rnn.num_layers * dir_mult, B, self.rnn.hidden_size)
            c0 = x.new_zeros_like(h0)
        return h0, c0

    def _maybe_concat_with_buffer(self, x):
        if self._x_buf is None:
            return x
        x_newest = x[:, -1:, :]
        return torch.cat([self._x_buf, x_newest], dim=1)

    def _compose_next_init(self, h1, c1, device, dtype):
        if not self.use_carry_init:
            return
        if not self.bidirectional:
            self._carry_h0, self._carry_c0 = h1.detach(), c1.detach()
            return
        L, B, H = self.rnn.num_layers, h1.size(1), h1.size(2)
        if self.learnable_init_state:
            h_next = self.h0.expand(self.num_layers, B, H).contiguous().clone().to(device=device, dtype=dtype)
            c_next = self.c0.expand(self.num_layers, B, H).contiguous().clone().to(device=device, dtype=dtype)
        else:
            h_next = torch.zeros(self.num_layers, B, H, device=device, dtype=dtype)
            c_next = torch.zeros_like(h_next)
        # forward halves (even indices) from h1/c1, backward halves reset
        h_next[0::2] = h1[0::2].detach()
        c_next[0::2] = c1[0::2].detach()
        self._carry_h0, self._carry_c0 = h_next, c_next
        
    def reset_buffer(self):
        self._x_buf = None
        self._carry_h0 = None
        self._carry_c0 = None
    
    def forward(self, x, h0=None, c0=None, new_sequence: bool = True):
        if new_sequence:
            self.reset_buffer()
        x = self._maybe_concat_with_buffer(x)
        if self.use_carry_init and self._carry_h0 is not None and self._carry_c0 is not None:
            h0, c0 = self._carry_h0, self._carry_c0
        else:
            h0, c0 = self._init_states(x, h0, c0)
        a_vec, (hn, cn) = self.rnn(x, (h0, c0))
        with torch.no_grad():
            _, (h1, c1) = self.rnn(x[:, :1, :], (h0, c0))
            self._compose_next_init(h1, c1, device=x.device, dtype=x.dtype)
        if self.store_x:
            self._x_buf = x.detach()
        return a_vec, hn, cn


@MODELS.register_module()
class AnchorModule(nn.Module):
    @staticmethod
    def _init_anchors(
        xyz_range: List[float] = [-0.3, -0.3, -0.3, 0.3, 0.3, 2.1],
        xyz_interval: List[float] = [0.3, 0.3, 0.3]
    ):  
        x_min, y_min, z_min, x_max, y_max, z_max = xyz_range
        x_voxel_size, y_voxel_size, z_voxel_size = round((x_max - x_min) / xyz_interval[0]) + 1,\
                                                    round((y_max - y_min) / xyz_interval[1]) + 1,\
                                                    round((z_max - z_min) / xyz_interval[2]) + 1
        centroids = np.zeros((z_voxel_size, y_voxel_size, x_voxel_size, 3), dtype=np.float32)
        for z_no in range(z_voxel_size):
            for y_no in range(y_voxel_size):
                for x_no in range(x_voxel_size):
                    lx = x_min + x_no * xyz_interval[0]
                    ly = y_min + y_no * xyz_interval[1]
                    lz = z_min + z_no * xyz_interval[2]
                    centroids[z_no, y_no, x_no, 0] = lx
                    centroids[z_no, y_no, x_no, 1] = ly
                    centroids[z_no, y_no, x_no, 2] = lz
        return centroids
    
    @staticmethod
    def _group_anchors(anchors, nsample, xyz, points):
        B, N, C = xyz.shape
        _, S, _ = anchors.shape
        idx = msh_utils.point_ball_set(nsample, xyz, anchors)
        grouped_xyz = msh_utils.index_points(xyz, idx) # [B, npoint, nsample, C]
        grouped_anchors = anchors.view(B, S, 1, C).repeat(1, 1, nsample, 1)
        grouped_xyz_norm = grouped_xyz - grouped_anchors 
        
        grouped_points = msh_utils.index_points(points, idx)
        new_points = torch.cat([grouped_anchors, grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+C+D]
        return new_points
        
        
    def __init__(self,
                 anchor_cfg: dict,
                 anchor_pointnet_cfg: dict,
                 anchor_voxelnet_cfg: dict,
                 anchor_rnn_cfg: dict,
                 ):
        super().__init__()
        anchor_cfg = copy.deepcopy(anchor_cfg)
        self.grouping_nsample = anchor_cfg.pop("grouping_nsample")
        self.register_buffer("template_points", torch.from_numpy(
            self._init_anchors(**anchor_cfg)).to(get_device()))
        self.voxel_size = self.template_points.shape[:3]
        self.spatial_volume = self.voxel_size[0] * self.voxel_size[1] * self.voxel_size[2]
        self.apointnet = AnchorPointNet(**anchor_pointnet_cfg)
        self.avoxel = AnchorVoxelNet(**anchor_voxelnet_cfg)
        self.arnn = AnchorRNN(**anchor_rnn_cfg)
    
    def forward(self, x, g_loc, h0, c0, batch_size, length_size, feature_size):
        if g_loc.size(1) != length_size:
            g_loc = g_loc[:, -length_size:, :].contiguous()
        g_loc = g_loc.view(batch_size * length_size, 1, 2).repeat(1, self.spatial_volume, 1)
        anchors = self.template_points.view(1, self.spatial_volume, 3).repeat(batch_size * length_size, 1, 1)
        anchors[:,:,:2] += g_loc
        grouped_points = self._group_anchors(anchors, self.grouping_nsample, xyz=x[..., :3], points=x[..., 3:])
        grouped_points = grouped_points.view(batch_size * length_size * self.spatial_volume, self.grouping_nsample, 3 + feature_size)
        voxel_points, attn_weights = self.apointnet(grouped_points)
        voxel_points = voxel_points.view(batch_size * length_size, self.voxel_size[0], self.voxel_size[1], self.voxel_size[2], self.apointnet.channels[-1])
        voxel_vec = self.avoxel(voxel_points)
        voxel_vec = voxel_vec.view(batch_size, length_size, self.avoxel.channels[-1])
        a_vec, hn, cn = self.arnn(voxel_vec, h0, c0, new_sequence=(length_size > 1))
        return a_vec, attn_weights, hn, cn