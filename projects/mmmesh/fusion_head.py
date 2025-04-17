from typing import List
import torch
import torch.nn as nn

from mwpose3d.registry import MODELS



@MODELS.register_module()
class SimpleKpFusionHead(nn.Module):
    def __init__(self,
                 channels: List[int] = [128, 128, 48],
    ):
        super().__init__()
        self.channels = channels
        fusion_layers = []
        for i in range(len(channels) - 1):
            fusion_layers.append(nn.Linear(channels[i], channels[i+1]))
            if i != len(channels) - 2:
                fusion_layers.append(nn.ReLU())
        self.fusion_layers = nn.Sequential(*fusion_layers)
    
    def forward(self, g_vec, a_vec, batch_size, length_size):
        x = torch.cat((g_vec, a_vec), dim=-1)
        x = self.fusion_layers(x)
        return x
    