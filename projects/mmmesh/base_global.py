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
                 bidirectional: bool = False,):
        super().__init__()
        self.num_layers = num_layers * 2 if bidirectional else num_layers
        self.in_channel = in_channel
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(in_channel, hidden_size, num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.fc1 = nn.Linear(fc_channels[0], fc_channels[1])
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(fc_channels[1], fc_channels[2])

        self.learnable_init_state = learnable_init_state
        if learnable_init_state:
            self.h0 = nn.Parameter(torch.zeros(self.num_layers, 1, hidden_size))
            self.c0 = nn.Parameter(torch.zeros(self.num_layers, 1, hidden_size))
        
        
    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)
        if self.learnable_init_state:
            h0 = self.h0.expand(-1, batch_size, -1).contiguous()
            c0 = self.c0.expand(-1, batch_size, -1).contiguous()
        g_vec, (hn, cn) = self.rnn(x, (h0, c0))
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
        g_vec, g_loc, hn, cn = self.grnn(x, h0, c0)
        return g_vec, g_loc, attn_weights, hn, cn
        