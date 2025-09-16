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
                 use_carry_init: bool = False,
                 store_x: bool = False
                  ):
        super().__init__()
        self.num_layers = num_layers
        self.in_channel = in_channel
        self.learnable_init_state = learnable_init_state
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.use_carry_init = use_carry_init
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
        
        self._x_buf = None
        self._carry_h0: torch.Tensor | None = None
        self._carry_c0: torch.Tensor | None = None
    
    def reset_buffer(self):
        self._x_buf = None
        self._carry_h0 = None
        self._carry_c0 = None
    
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
        x_newest = x[:, -1:, :]
        x_full = torch.cat([self._x_buf, x_newest], dim=1)
        return x_full

    def _compose_next_init(self, h1: torch.Tensor, c1: torch.Tensor, device, dtype):
        if not self.use_carry_init:
            return
        if not self.bidirectional:
            self._carry_h0 = h1
            self._carry_c0 = c1
            return
        L, B, H = self.num_layers, h1.size(1), h1.size(2)
        if self.learnable_init_state:
            h_next = self.h0.expand(L * 2, B, H).contiguous().clone().to(device=device, dtype=dtype)
            c_next = self.c0.expand(L * 2, B, H).contiguous().clone().to(device=device, dtype=dtype)
        else:
            h_next = torch.zeros(L * 2, B, H, device=device, dtype=dtype)
            c_next = torch.zeros_like(h_next)
        h_next[0::2] = h1[0::2]
        c_next[0::2] = c1[0::2]
        self._carry_h0 = h_next
        self._carry_c0 = c_next
    
    def forward(self, x, h0=None, c0=None, new_sequence: bool = True):
        if new_sequence:
            self.reset_buffer()
        x = self._maybe_concat_with_buffer(x)
        if self.use_carry_init and self._carry_h0 is not None and self._carry_c0 is not None:
            h0, c0 = self._carry_h0, self._carry_c0
        else:
            h0, c0 = self._init_states(x, h0, c0)

        g_vec, (hn, cn) = self.rnn(x, (h0, c0))
        g_loc = self.fc2(self.faf1(self.fc1(g_vec)))
        with torch.no_grad():
            _, (h1, c1) = self.rnn(x[:, :1, :], (h0, c0))
            self._compose_next_init(h1, c1, device=x.device, dtype=x.dtype)
        if self.store_x:
            self._x_buf = x.detach()
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
        