from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mwpose3d.registry import MODELS
from .transformer import *


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k):
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz) # b x n x n (self correlation result)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k (k nearest neighbors)
        knn_xyz = index_points(xyz, knn_idx) # b x n x k x 3 (n points, each point with its k nearest neighbors)
        
        pre = features
        x = self.fc1(features) # b x n x f -> b x n x d_model (f = d_points)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        # q: b x n x d_model, k: b x n x k x d_model, v: b x n x k x d_model
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x 3 -> b x n x k x d_model
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc) # b x n x k x d_model
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x d_model
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre # b x n x d_model -> b x n x f
        return res, attn



class TransitionDown(nn.Module):
    def __init__(self, in_channel, internal_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, internal_channel, 1),
            nn.BatchNorm1d(internal_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(internal_channel, out_channel, 1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        # self.mlp_convs = nn.ModuleList()
        # self.mlp_bns = nn.ModuleList()
        # last_channel = in_channel
        # for out_channel in mlp: # mlp
        #     self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
        #     self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        #     last_channel = out_channel
        
    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_points_concat: sample points feature data, [B, N, 3+D]
        """
        new_features = torch.cat([xyz, features], dim=-1) # b x n x (3 + features)
        new_features = new_features.permute(0, 2, 1) # [B, 3+D, n]
        new_features = self.conv1(new_features) # [B, internal_channel, n]
        new_features = self.conv2(new_features) # [B, out_channel, n]
        new_features = new_features.permute(0, 2, 1) # [B, n, out_channel]
        return new_features
        

class Backbone(nn.Module):
    def __init__(self, nblocks, nneighbor, input_dim, transformer_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor) # input channels = 32
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        append_blocks = nblocks - 1
        for i in range(append_blocks):
            channel = 32 * 2 ** (i+1)
            self.transition_downs.append(TransitionDown(channel // 2 + 3, channel, channel))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.append_blocks = append_blocks
    
    def forward(self, x):
        # x: (B, N, 5). 5 = (x, y, z, d, l)
        xyz = x[..., :3]
        # print(f"xyz: {xyz.shape}")
        # print(f"x: {x.shape}")
        points = self.transformer1(xyz, self.fc1(x))[0]
        # print(f"points: {points.shape}")
        xyz_and_feats = [(xyz, points)]
        for i in range(self.append_blocks):
            points = self.transition_downs[i](xyz, points)
            # print(f"points: {points.shape}")
            points = self.transformers[i](xyz, points)[0]
            # print(f"points: {points.shape}")
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats

@MODELS.register_module()
class PointTransformerRegFeatureExtractor(nn.Module):
    def __init__(self, input_dim=5, nblocks=5, nneighbor=16, seq_tag=False,
                 transformer_dim=128, n_p=17, dropout=0., drop_key=0.,
                 dim=512, depth=5, heads=4, dim_head=128, mlp_dim=256, collapse_t_into: Literal["n", "b"] = 'n'):
        super().__init__()
        self.backbone = Backbone(
            nblocks = nblocks,
            nneighbor = nneighbor,
            input_dim = input_dim,
            transformer_dim = transformer_dim
        )
        
        self.nblocks = nblocks
        self.n_p = n_p
        self.seq_tag = seq_tag
        self.collapse_t_into = collapse_t_into

        depth = nblocks
        self.joint_posembeds_vector = nn.Parameter(torch.tensor(self.get_positional_embeddings1(self.n_p, dim)).float())
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout, dropkey=drop_key)

        mid_dim = 32
        self.fc2 = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(256, mid_dim),
        )

        self.fc3 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(mid_dim, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        if x.dim() == 4:
            b, t, n, c = x.shape
            if self.seq_tag:
                if t > 1:
                    seq = (torch.arange(t, device=x.device).float() - (t - 1) / 2) / ((t - 1) / 2)
                else:
                    seq = torch.zeros(t, device=x.device)
                seq = seq.view(1, t, 1, 1).expand(b, t, n, 1)
                x = torch.cat([x, seq], dim=-1)  # (B, T, N, C+1)
                c = c + 1

            if self.collapse_t_into == "n":
                # Original behavior: time -> points
                x_flat = x.view(b, t * n, c)
                points, _ = self.backbone(x_flat)                 # (B, T*N, dim)
                joint_emb = self.joint_posembeds_vector.expand(b, -1, -1)  # (B, J, dim)
                tokens = torch.cat([joint_emb, points], dim=1)    # (B, J + T*N, dim)
                out = self.transformer(tokens)[:, :self.n_p, :]   # (B, J, dim)
            else:
                # New behavior: time -> batch (per-frame backbone), then aggregate time in final transformer
                x_bt = x.view(b * t, n, c)                        # (B*T, N, C)
                points_bt, _ = self.backbone(x_bt)                # (B*T, N, dim)
                points = points_bt.view(b, t, n, -1)              # (B, T, N, dim)

                # Per-frame joint tokens + per-frame point tokens
                joint_emb = self.joint_posembeds_vector.view(1, 1, self.n_p, -1).expand(b, t, -1, -1)  # (B,T,J,dim)
                tokens = torch.cat([joint_emb, points], dim=2)    # (B, T, J+N, dim)

                # Aggregate across time in the final Transformer
                tokens_flat = tokens.view(b, t * (self.n_p + n), -1)  # (B, T*(J+N), dim)
                out_all = self.transformer(tokens_flat)               # (B, T*(J+N), dim)

                # Keep the joint tokens of the *last* frame (consistent with original extractor contract)
                frame_len = self.n_p + n
                start = (t - 1) * frame_len
                out = out_all[:, start:start + self.n_p, :]          # (B, J, dim)

        else:
            # x: (B, N, C)
            b, n, c = x.shape
            points, _ = self.backbone(x)                          # (B, N, dim)
            joint_emb = self.joint_posembeds_vector.expand(b, -1, -1)
            tokens = torch.cat([joint_emb, points], dim=1)        # (B, J+N, dim)
            out = self.transformer(tokens)[:, :self.n_p, :]

        feat = self.fc2(out)
        pts = self.fc3(feat)

        
        return pts, feat
    
    def get_positional_embeddings1(self, sequence_length, d):
        result = np.ones([1, sequence_length, d])
        for i in range(sequence_length):
            for j in range(d):
                result[0][i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result
