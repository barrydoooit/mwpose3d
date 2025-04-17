import math
from typing import Literal
from copy import deepcopy as dcy
import torch
import torch.nn as nn

from mmengine.device import get_device

from .GraFormer import GraAttenLayer, GraphNet, MultiHeadedAttention
from .ChebConv import _GraphConv, ChebConv



class GCNdiff(nn.Module):
    def __init__(self,
                 adj,
                 limb_size: int,
                 con_gcn_cfg: dict,
                 global_feat_size: int,
                 past_frames: int,
                 radar_input_c: int,
                 global_flag: Literal[0, 1],
                 local_flag: Literal[0, 1, 2],
                 temp_flag: Literal[0, 1],
                 limb_flag: Literal[0, 1, 2],
                 ):
        super().__init__()
        
        self.adj = adj
        self.limb_size = limb_size
        self.hid_dim, self.emd_dim, self.coords_dim, num_layers, n_head, dropout, self.n_joints = \
            con_gcn_cfg['hid_dim'], con_gcn_cfg['emd_dim'], con_gcn_cfg['coords_dim'], \
            con_gcn_cfg['num_layers'], con_gcn_cfg['n_head'], con_gcn_cfg['dropout'], con_gcn_cfg['n_joints']
        self.local_points = con_gcn_cfg.get('local_points', 50)
        
        self.emd_dim = self.hid_dim * 4
        self.src_mask = torch.tensor([[[True] * self.n_joints]]).to(get_device())
        self.src_mask = self.src_mask[:, :, :self.n_joints]
        
        self.n_layers = num_layers
        self.feature_size = global_feat_size
        self.past_frames = past_frames
        
        # _gconv_input = ChebConv(in_c=self.coords_dim[0], out_c=self.hid_dim, K=2)
        _gconv_input2 = ChebConv(in_c=3, out_c=self.hid_dim, K=2)
        _gconv_input3 = ChebConv(in_c=3, out_c=self.hid_dim, K=2)
        
        _gconv_layers = []
        _attention_layer = []
        
        dim_model = self.hid_dim
        attn = MultiHeadedAttention(n_head, dim_model, dropout)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=self.n_joints)
        
        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC_diff(adj=adj, input_dim=self.hid_dim, 
                                                 output_dim=self.hid_dim, emd_dim=self.emd_dim, hid_dim=self.hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(size=dim_model, self_attn=dcy(attn), feed_forward=dcy(gcn), dropout=dropout))
            
        # self.gconv_input = _gconv_input
        self.gconv_input2 = _gconv_input2
        self.gconv_input3 = _gconv_input3
        
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=3*(self.past_frames+1), K=2)
        
        ### Diffusion Configuration ###
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.hid_dim, self.emd_dim),
            torch.nn.Linear(self.emd_dim, self.emd_dim),
        ])
        
        self.global_flag, self.local_flag, self.temp_flag, self.limb_flag = global_flag, local_flag, temp_flag, limb_flag
        
        ### Global Embedding Start ###
        if self.global_flag == 1:
            ### global gcn projector ###
            _gconv_input1 = ChebConv(in_c=self.feature_size, out_c=self.hid_dim, K=2)
            self.gconv_input1 = _gconv_input1
        
        ### Local Embedding Start ###
        if self.local_flag == 1:
            self.input_dim = radar_input_c
            self.local_dim = self.hid_dim
            self.local_emb = nn.Module()   
            self.local_emb.dense = nn.ModuleList([
                torch.nn.Linear(self.input_dim,self.local_dim // 2),
                torch.nn.Linear(self.local_dim // 2,self.local_dim),
            ])
        
            ### Local Attn ###
            self.local_attn_iter = 5
            local_attn = MultiHeadedAttention(h=n_head, d_model=self.local_dim)
            _local_attention = []
            for i in range(self.local_attn_iter):
                _local_attention.append(dcy(local_attn))
            self.local_attention = nn.ModuleList(_local_attention)
        elif self.local_flag == 2:
            self.input_dim = 6
            self.local_dim = self.hid_dim
            self.local_emb = nn.Module()
            self.local_emb.dense = nn.ModuleList([
                torch.nn.Linear(self.input_dim, self.local_dim // 2),
                torch.nn.Linear(self.local_dim // 2, self.local_dim),
            ])
            
            ### Local Attn ###
            self.local_attn_iter = 5
            local_attn = MultiHeadedAttention(h=n_head, d_model=self.local_dim)
            _local_attention = []
            for i in range(self.local_attn_iter):
                _local_attention.append(dcy(local_attn))
            self.local_attention = nn.ModuleList(_local_attention)
        
        ### Temporal Embedding Start ###
        if self.temp_flag == 1:
            _gconv_input_t = ChebConv(in_c=3, out_c=self.hid_dim, K=2)
            self.gconv_input_t = _gconv_input_t
            
            ### Temporal Convolutional Encoder ###
            self.temporal_hid_dim = self.hid_dim * self.n_joints
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(self.temporal_hid_dim, self.temporal_hid_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(self.temporal_hid_dim, self.temporal_hid_dim, kernel_size=3, padding=1),
                nn.MaxPool1d(kernel_size=self.past_frames),
            )
            
        ### Limb Embedding Start ###
        if self.limb_flag == 1:
            # limb mlp encoder
            limb_feature_size = 1024
            self.limb_linear = nn.Sequential(
                nn.LayerNorm(self.n_joints * self.feature_size),
                nn.Linear(self.n_joints * self.feature_size, limb_feature_size),
                nn.ReLU(),
                nn.Linear(limb_feature_size, self.limb_size),
            )
            
            # limb token conditional projector
            self.limb_emb_layer = nn.Module()
            self.limb_emb_layer.dense = nn.ModuleList([
                torch.nn.Linear(self.limb_size, self.emd_dim),
                torch.nn.Linear(self.emd_dim, self.emd_dim),
            ])
        elif self.limb_flag == 2:
            self.limb_linear = nn.Sequential(
                nn.LayerNorm((self.past_frames + 1) * self.n_joints * 3),
                torch.nn.Linear((self.past_frames + 1) * self.n_joints * 3, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, self.limb_size)
            )
            
            # limb token conditional projector
            self.limb_emb_layer = nn.Module()
            self.limb_emb_layer.dense = nn.ModuleList([
                torch.nn.Linear(self.limb_size, self.emd_dim),
                torch.nn.Linear(self.emd_dim, self.emd_dim),
            ])
        
    def forward(self, x, t, cemd, radar, limb_len):
        temb = get_timestep_embedding(t, self.hid_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        
        ### Global Embedding ###
        if self.global_flag == 1:
            global_emb = self.gconv_input1(cemd, self.adj)
        
        ### Local Embedding ###
        if self.local_flag == 1:
            thre = 0.04
            local_emb_list = []
            
            b, t, n, c = radar.shape
            point_cloud = radar.view(b, t*n, c)
            
            for joint_id in range(self.n_joints):
                # select local 50 points, with 0 paddings
                anchor = x[:, [joint_id], -3:]
                dist = torch.sum(torch.pow(point_cloud[:,:,:3] - anchor, 2), dim=-1)
                index_sort = torch.argsort(dist, dim=-1).unsqueeze(-1).expand(-1, -1, c)
                point_cloud_select = torch.gather(point_cloud, 1, index_sort)[:, :self.local_points, :]
                
                # scaling by the number of points
                dist_sort = torch.sum(torch.pow(point_cloud_select[:,:,:3] - anchor, 2), dim=-1)
                dist_scale = torch.sum(dist_sort.lt(thre), dim=-1, keepdim=True) / self.local_points
                
                # shared embedding layers
                local_emb = self.local_emb.dense[0](point_cloud_select)
                local_emb = nonlinearity(local_emb)
                local_emb = self.local_emb.dense[1](local_emb)
                
                # local self attention
                for i in range (self.local_attn_iter):
                    local_emb = local_emb + self.local_attention[i](local_emb, local_emb, local_emb)
                
                # max pooling
                local_emb = torch.mean(local_emb, dim=1)
                local_score = dist_scale
                local_emb = local_emb * local_score
                local_emb_list.append(local_emb)

            # concatenration
            local_emb = torch.stack(local_emb_list, dim=1)
        
        ### Temporal Embedding ###
        if self.temp_flag == 1:
            x_past_list =[]
            for i in range(self.past_frames):
                x_curr = x[:, :, i*3:(i+1)*3]
                x_curr = self.gconv_input_t(x_curr, self.adj)
                x_past_list.append(x_curr)
            
            x_past = torch.stack(x_past_list, dim=1)
            past_emb = x_past.view(-1, x_past.size(1), x_past.size(2) * x_past.size(3)).permute(0, 2, 1)
            past_emb = self.temporal_conv(past_emb).squeeze().view(-1, x_past.size(2), x_past.size(3))
        
        ### Limb Embedding ###
        if self.limb_flag == 1:
            limb_len_pred = self.limb_linear(cemd.view(-1, self.n_joints * self.feature_size))
            
            if limb_len == None:
                limb_len = limb_len_pred
            
            limb_emb = self.limb_emb_layer.dense[0](limb_len)
            limb_emb = nonlinearity(limb_emb)
            limb_emb = self.limb_emb_layer.dense[1](limb_emb)
        elif self.limb_flag == 2:
            limb_len_pred = self.limb_linear(x.reshape(-1, (self.past_frames + 1) * self.n_joints * 3))
            if limb_len == None:
                limb_len = limb_len_pred
            limb_emb = self.limb_emb_layer.dense[0](limb_len)
            limb_emb = nonlinearity(limb_emb)
            limb_emb = self.limb_emb_layer.dense[1](limb_emb)
        
        ### Graph Convolutional Network ###
        condition_embedding = global_emb
        if self.local_flag == 1:
            condition_embedding += local_emb
        if self.temp_flag == 1:
            condition_embedding += past_emb
        if self.limb_flag == 1 or self.limb_flag == 2:
            temb += limb_emb
        
        out = self.gconv_input2(x[:,:,-3:], self.adj) + condition_embedding
        for i in range(self.n_layers):
            out = self.atten_layers[i](out, self.src_mask)
            out = self.gconv_layers[i](out, temb, condition= condition_embedding)
        
        out = self.gconv_output(out, self.adj)
        
        return out, limb_len_pred 
        
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

### the embedding of diffusion timestep ###
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class _ResChebGC_diff(nn.Module):
    def __init__(self, adj, input_dim, output_dim, emd_dim, hid_dim, p_dropout):
        super(_ResChebGC_diff, self).__init__()
        self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)
        ### time embedding ###
        self.temb_proj = torch.nn.Linear(emd_dim,hid_dim)

    def forward(self, x, temb, condition = None):
        residual = x
        out = self.gconv1(x, self.adj)
        out = out + self.temb_proj(nonlinearity(temb))[:, None, :]
        if condition != None:
            out = out + condition
        out = self.gconv2(out, self.adj)
        return residual + out