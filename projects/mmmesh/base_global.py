from typing import List
import torch
import torch.nn as nn
from mwpose3d.registry import MODELS



@MODELS.register_module()
class BasePointNet(nn.Module):
    def __init__(self,
                 channels: List[int] = [6, 8, 16, 24],
                 kernel_size: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self._make_layers()
    
    def _make_layers(self):
        self.conv1 = nn.Conv1d(self.channels[0], self.channels[1], self.kernel_size)
        self.cb1 = nn.BatchNorm1d(self.channels[1])
        self.calf1 = nn.ReLU()
        self.conv2 = nn.Conv1d(self.channels[1], self.channels[2], self.kernel_size)
        self.cb2 = nn.BatchNorm1d(self.channels[2])
        self.calf2 = nn.ReLU()
        self.conv3 = nn.Conv1d(self.channels[2], self.channels[3], self.kernel_size)
        self.cb3 = nn.BatchNorm1d(self.channels[3])
        self.calf3 = nn.ReLU()
    
    def forward(self, pcd): # pcd: (B, N, 6)
        x = pcd.permute(0, 2, 1)
        x = self.calf1(self.cb1(self.conv1(x)))
        x = self.calf2(self.cb2(self.conv2(x)))
        x = self.calf3(self.cb3(self.conv3(x)))
        
        x = x.permute(0, 2, 1)
        x = torch.cat((pcd[:,:,:4], x), dim=-1)
        
        return x

@MODELS.register_module()
class GlobalPointNet(nn.Module):
    def __init__(self,
                 channels: List[int] = [28, 32, 48, 64],
                 kernel_size: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self._make_layers()
    
    def _make_layers(self):
        self.conv1 = nn.Conv1d(self.channels[0], self.channels[1], self.kernel_size)
        self.cb1 = nn.BatchNorm1d(self.channels[1])
        self.calf1 = nn.ReLU()
        self.conv2 = nn.Conv1d(self.channels[1], self.channels[2], self.kernel_size)
        self.cb2 = nn.BatchNorm1d(self.channels[2])
        self.calf2 = nn.ReLU()
        self.conv3 = nn.Conv1d(self.channels[2], self.channels[3], self.kernel_size)
        self.cb3 = nn.BatchNorm1d(self.channels[3])
        self.calf3 = nn.ReLU()
        
        self.attn = nn.Linear(self.channels[3], 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.calf1(self.cb1(self.conv1(x)))
        x = self.calf2(self.cb2(self.conv2(x)))
        x = self.calf3(self.cb3(self.conv3(x)))
        x = x.permute(0, 2, 1)
        
        attn_weights = self.softmax(self.attn(x))
        attn_vec = torch.sum(attn_weights * x, dim=1)
        return attn_vec, attn_weights

@MODELS.register_module()
class GlobalRNN(nn.Module):
    def __init__(self,
                 in_channel: int = 64,
                 hidden_size: int = 64,
                 num_layers: int = 3,
                 batch_first: bool = True,
                 dropout: float = 0.1,
                 fc_channels: List[int] = [64, 16, 2],
                 learnable_init_state: bool = False,
                 bidirectional: bool = False,
                 store_x: bool = False,
                  ):
        super().__init__()
        self.num_layers = num_layers
        self.in_channel = in_channel
        self.learnable_init_state = learnable_init_state
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.store_x = store_x
        self.rnn = nn.LSTM(in_channel, hidden_size, num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        dir_mult = 2 if bidirectional else 1
        self.fc1 = nn.Linear(fc_channels[0] * dir_mult, fc_channels[1])
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(fc_channels[1], fc_channels[2])

        if learnable_init_state:
            self.h0 = nn.Parameter(torch.zeros(self.num_layers * dir_mult, 1, hidden_size))
            self.c0 = nn.Parameter(torch.zeros(self.num_layers * dir_mult, 1, hidden_size))
        else:
            self.h0 = None
            self.c0 = None
        
        self.reset_buffer()
    
    def reset_buffer(self):
        self._x_buf = None
        self._x_len = 0
        self._x_T = 0
        self._wptr = 0

    def _init_states(self, x: torch.Tensor, h0: torch.Tensor | None, c0: torch.Tensor | None):
        batch_size = x.size(0)
        dir_mult = 2 if self.bidirectional else 1
        if h0 is None or c0 is None:
            if self.learnable_init_state:
                h0 = self.h0.expand(self.num_layers * dir_mult, batch_size, self.hidden_size).contiguous()
                c0 = self.c0.expand(self.num_layers * dir_mult, batch_size, self.hidden_size).contiguous()
            else:
                h0 = x.new_zeros(self.rnn.num_layers * dir_mult, batch_size, self.rnn.hidden_size)
                c0 = x.new_zeros_like(h0)
        return h0, c0
    
    def _maybe_concat_with_buffer(self, x: torch.Tensor, concat: bool = True):
        if not concat or self._x_buf is None:
            return x
        B, T_in, C = x.shape
        device, dtype = x.device, x.dtype

        if T_in > 1:
            self._x_T = T_in
            self._x_buf = torch.empty(B, self._x_T, C, device=device, dtype=dtype)
            self._x_buf[:, :T_in, :].copy_(x.detach())
            self._x_len = T_in
            self._wptr = 0
            return self._x_buf[:, :self._x_len, :]

        # T_in == 1
        if self._x_buf is None:
            self._x_T = 1
            self._x_buf = torch.empty(B, 1, C, device=device, dtype=dtype)
            self._x_buf[:, 0:1, :].copy_(x.detach())
            self._x_len = 1
            self._wptr = 0
            return self._x_buf[:, :1, :]
        
        if self._x_len < self._x_T:
            idx = self._x_len
            self._x_buf[:, idx:idx+1, :].copy_(x.detach())
            self._x_len += 1
            return self._x_buf[:, :self._x_len, :]
        
        idx = self._wptr
        self._x_buf[:, idx:idx+1, :].copy_(x.detach())
        self._wptr = (self._wptr + 1) % self._x_T

        # Build chronological view: [oldest..newest] = buf[wptr:]+buf[:wptr]
        if self._wptr == 0:
            return self._x_buf[:, :self._x_T, :]
        else:
            return torch.cat([
                self._x_buf[:, self._wptr:, :],
                self._x_buf[:, :self._wptr, :]
            ], dim=1)

    def forward(self, x, h0=None, c0=None, new_sequence: bool = True):
        if self.training:
            h0, c0 = self._init_states(x, h0, c0)
            g_vec, (hn, cn) = self.rnn(x, (h0, c0))
            g_loc = self.fc2(self.faf1(self.fc1(g_vec)))
            return g_vec, g_loc, hn, cn
        
        if new_sequence:
            self.reset_buffer()
        x_full = self._maybe_concat_with_buffer(x)
        h0, c0 = self._init_states(x_full, h0, c0)
        g_vec, (hn, cn) = self.rnn(x_full, (h0, c0))
        g_loc = self.fc2(self.faf1(self.fc1(g_vec)))
        return g_vec, g_loc, hn, cn

@MODELS.register_module()
class GlobalModule(nn.Module):
    def __init__(self,
                 global_pointnet_cfg: dict,
                 global_rnn_cfg: dict,
                 ):
        super().__init__()
        self.gpointnet = GlobalPointNet(**global_pointnet_cfg)
        self.grnn = GlobalRNN(**global_rnn_cfg)
    
    def forward(self, x, h0, c0, batch_size, length_size):
        x, attn_weights = self.gpointnet(x)
        x = x.view(batch_size, length_size, self.gpointnet.channels[-1])
        new_sequence = length_size > 1
        g_vec, g_loc, hn, cn = self.grnn(x, h0, c0, new_sequence=new_sequence)
        return g_vec, g_loc, attn_weights, hn, cn
        