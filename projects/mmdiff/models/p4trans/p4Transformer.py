import numpy as np
import torch
import torch.nn as nn

from mwpose3d.registry import MODELS

from ..pointTrans.transformer import Transformer
from .point_4d_convolution import P4DConv



@MODELS.register_module()
class P4TransformerFeatureExtractor(nn.Module):
    def __init__(self,
                 radius,
                 nsamples,
                 spatial_stride,
                 temporal_kernel_size,
                 temporal_stride,
                 emb_relu,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 num_classes,
                 joint_num=17,
                 dropout1=0.0,
                 dropout2=0.0):
        super().__init__()
        
        self.tube_embedding = P4DConv(in_planes=3,
                                      mlp_planes=[dim],
                                      mlp_batch_norm=[False],
                                      mlp_activation=[False],
                                      spatial_kernel_size=[radius, nsamples], 
                                      spatial_stride=spatial_stride,
                                      temporal_kernel_size=temporal_kernel_size, 
                                      temporal_stride=temporal_stride, 
                                      temporal_padding=[1, 0],
                                      operator='+', 
                                      spatial_pooling='max', 
                                      temporal_pooling='max')
        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1,
                                       stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else nn.Identity()
        
        self.transformer = Transformer(dim, depth, heads,
                                       dim_head, mlp_dim, dropout=dropout1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )
        self.joint_num = joint_num
        
        self.joint_posembeds_vector = nn.Parameter(torch.tensor(
            self.get_positional_embeddings1(self.joint_num, 1024)).float()
        )
        
        # point Prediction Head
        input_dim = dim
        mid_dim = 64
        
        self.dim_reduce_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, mid_dim)
        )
        
        input_dim = mid_dim
        self.point_prediction_heads = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, 3)
        )

        self.var_prediction_heads = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, 3),
        )
    
    def forward(self, radar):
        point_cloud = radar[:, :, :, :3]
        point_fea = radar[:, :, :, 3:].permute(0, 1, 3, 2)  # [B, L, N, 3]
        device = radar.get_device()
        
        xyzs, features = self.tube_embedding(point_cloud, point_fea) # [B, L, n 3], [B, L, C, n]
        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3])) # [B, L*n, 4]
        
        features = features.permute(0, 1, 3, 2) # [B, L, n, C]
        xyzts_embd = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)
        embedding = xyzts_embd + features
        # open for template embedding
        joint_embedding =self.joint_posembeds_vector.expand(radar.size()[0], -1, -1)
        embedding = torch.cat([joint_embedding, embedding], dim=1)
        embedding = self.emb_relu(embedding)
        
        # open for template embedding
        output = self.transformer(embedding)
        joint_embedding = output[:, :self.joint_num, :]
        joint_embedding = self.dim_reduce_head(joint_embedding)
        joint_embed_out = joint_embedding
        output = self.point_prediction_heads(joint_embedding)

        return output, joint_embed_out
    
    def get_positional_embeddings1(self, sequence_length, d):
        result = np.ones([1, sequence_length, d])
        for i in range(sequence_length):
            for j in range(d):
                result[0, i, j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result
        