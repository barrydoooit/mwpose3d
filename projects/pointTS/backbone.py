import torch
import torch.nn as nn

from mwpose3d.registry import MODELS



@MODELS.register_module()
class PointNetBackbone(nn.Module):
    """
    PointNet-based spatial feature extractor for a fused radar point-cloud frame.

    Args:
        input_channels (int): Number of input channels per point (default: 3 for x,y,z).
        conv_channels (tuple of int): Output channels for successive Conv1d layers.
        global_feat_dim (int): Dimension of the global feature after MLP.
    """
    def __init__(self,
                 input_channels: int = 3,
                 conv_channels: tuple = (128, 256, 512, 1024),
                 global_feat_dim: int = 256):
        super().__init__()
        assert len(conv_channels) == 4, "Require 4 layers as per PoinTS architecture"

        # Four 1x1 conv layers with BatchNorm and ReLU
        layers = []
        in_c = input_channels
        for out_c in conv_channels:
            layers.append(nn.Conv1d(in_c, out_c, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        self.conv = nn.Sequential(*layers)

        # MLP to reduce 1024 -> global_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(conv_channels[-1], global_feat_dim, bias=False),
            nn.BatchNorm1d(global_feat_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N, C) where N is num_points, C=input_channels
        Returns:
            feat: Tensor of shape (B, global_feat_dim)
        """
        # Transpose to (B, C, N)
        x = x.transpose(1, 2)
        # Apply pointwise convs
        x = self.conv(x)
        # Global max pool over points -> (B, C_last, 1)
        x = torch.max(x, dim=2, keepdim=False)[0]
        # x is (B, 1024)
        # MLP -> (B, global_feat_dim)
        feat = self.mlp(x)
        return feat