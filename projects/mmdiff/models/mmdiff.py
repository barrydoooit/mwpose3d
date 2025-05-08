from copy import deepcopy
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmengine.device import get_device

from mwpose3d.datasets.skel_data_sample import SkeletonDataSample
from mwpose3d.models.base import BaseSkeletonEstimModel
from mwpose3d.registry import MODELS
from .utils import compress_joints_and_edges, compute_alpha, get_beta_schedule
from .gcn_diff import GCNdiff
from .ChebConv import adj_mx_from_edges


@MODELS.register_module()
class mmDiffPredictor(BaseSkeletonEstimModel):
    def _check_model_validty(self):
        if self.model_feat != None:
            input = torch.rand(self.input_shape, dtype=torch.float32, device=get_device())
            self.model_feat.to(get_device())
            output = self.model_feat(input)
            for i in range(2):
                if tuple(output[i].size()) != tuple(self.output_shape[i]):
                    raise RuntimeError(f"Desired [{i}] output shape of self.model_feat is {self.output_shape[i]}, receive output shape of {list(output[i].size())}")
            print("self.model_feat vadility passes.")
        
    def __init__(self,
                 point_cloud_size: int, 
                 keypoints_involved: List[int],
                 feature_extractor: dict,
                 diff_config,
                 beta_cfg: dict,
                 test_cfg: dict,
                 global_feat_size: int,
                 past_frames: int, # NOTE: Num of extra previous predictions for phase 2
                 seq_frames: int, # NOTE: Num of frames used for phase 1 feat extract
                 radar_input_c: int,
                 radar_input_c_mid: int,
                 input_shape: Tuple[int, int, int, int],
                 output_shape: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
                 global_flag: Literal[0, 1],
                 local_flag: Literal[0, 1, 2],
                 temp_flag: Literal[0, 1],
                 limb_flag: Literal[0, 1, 2],
                 gt_norm_joint: Optional[int] = None,
                ):
        super().__init__()
        self.point_cloud_size = point_cloud_size
        self.keypoints_involved = keypoints_involved
        self.num_joints = len(keypoints_involved)
        
        self.model_feat = MODELS.build(feature_extractor)
        self.input_shape = deepcopy(input_shape)
        self.output_shape = deepcopy(output_shape)
        self.radar_input_c = radar_input_c
        self.radar_input_c_mid = radar_input_c_mid
        assert self.input_shape[3] == radar_input_c
        assert self.output_shape[1][2] == global_feat_size
        
        self._check_model_validty()
        
        self.betas = torch.from_numpy(get_beta_schedule(
            beta_schedule=beta_cfg["beta_schedule"],
            beta_start=beta_cfg["beta_start"],
            beta_end=beta_cfg["beta_end"],
            num_diffusion_timesteps=beta_cfg["num_diffusion_timesteps"],
        )).float().to(get_device())
        self.num_timesteps = self.betas.shape[0]
        self.test_cfg = test_cfg

        ### Generate Diff Model ###
        self.diff_config = diff_config
        self.joint2idx, self.idx2joint, self.edges_compressed = compress_joints_and_edges(keypoints_involved)
        self.gt_norm_joint = self.joint2idx[gt_norm_joint] if gt_norm_joint is not None else None
        self.adj = adj_mx_from_edges(num_pts=self.num_joints, edges=self.edges_compressed, sparse=False).to(get_device())
        self.model_diff = GCNdiff(self.adj, 
                                  len(self.edges_compressed),
                                  self.diff_config,
                                  global_feat_size=global_feat_size,
                                  past_frames=past_frames,
                                  radar_input_c=radar_input_c_mid,
                                  global_flag=global_flag,
                                  local_flag=local_flag,
                                  temp_flag=temp_flag,
                                  limb_flag=limb_flag)
        
        ### History Coarse Pose Band ###
        self.past_frames = past_frames
        self.seq_frames = seq_frames
        # self.pose_coarse_curr = None # shape: (b, num_joints, 3)
        # self.cemd_curr = None # shape: (b, num_joints, global_feat_size)
        # self.x_history_list = None # shape: (b, num_joints, 3*(self.past_frames+1)), [[past_coarse_poses], curr_coarse_pose]
        
        self.criterion = nn.MSELoss()
        self._mode: Literal['train', 'pred_coarse', 'pred_fine'] = 'train' # TODO: replace loss-pretrain and loss-train by configuring _mode in train loop
    
    @property
    def mode(self) -> Literal['train', 'pred_coarse', 'pred_fine']:
        return self._mode
    
    @mode.setter
    def mode(self, value: Literal['train', 'pred_coarse', 'pred_fine']):
        assert value in ['train', 'pred_coarse', 'pred_fine']
        self._mode = value
    
    def forward(self, 
                inputs: torch.Tensor,
                data_samples: Optional[List[SkeletonDataSample]] = None,
                mode: str = 'tensor'):
        if mode == 'loss-pretrain':
            return self.loss(inputs, data_samples, pretrain=True)
        elif mode == 'loss-train':
            return self.loss(inputs, data_samples, pretrain=False)
        elif mode == 'predict':
            res = self.predict(inputs, data_samples)
            # for data_sample in data_samples:
            #     if data_sample.pred is not None:
            #         data_sample.pred = self.denorm_joints(data_sample.pred)
            #     if data_sample.gt is not None:
            #         data_sample.gt = self.denorm_joints(data_sample.gt)
            return res
        else:
            return self._forward(inputs, data_samples)
    

    """
    NOTE: 
    1. Only 5 frames are used for phase 1 feat extract
    2. past frames are used for phase 2 (i.e., past_frames predictions before current are used)
    3. Phase 2 need joints_predict, joint_emb, radar_mm (zeros)
    4. for predictions list, training use ground truth as the current prediction
    5. Phase 1 need 5 frames, phase 2 need (5-1) + (6+1) frames
    6. Phase 2 need (6+1) skeleton gt frame
    """
    def _forward(self, *args, **kwargs):
        raise NotImplementedError
        
    def extract_feat(self, pcd: torch.Tensor, only_current_frame: bool, return_list=False) -> torch.Tensor:
        frames_to_extract = self.past_frames + 1 if not only_current_frame else 1
        first_frame = pcd.size(1) -  frames_to_extract
        joint_predicts = []
        joint_embs = []
        for f_idx in range(frames_to_extract):
            _f_idx = f_idx + first_frame
            pcd_seq = pcd[:, _f_idx  -self.seq_frames + 1:_f_idx + 1, :, :]
            joints_predict, joint_emb = self.model_feat(pcd_seq)
            joint_predicts.append(joints_predict)
            joint_embs.append(joint_emb)
        if return_list:
            out_pose_pr = joint_predicts
        else:
            out_pose_pr = torch.concat(joint_predicts, dim=2)
        out_pose_feat_curr = joint_embs[-1]
        return out_pose_pr, out_pose_feat_curr
        
    def loss(self, batch_inputs, data_samples, pretrain=False):
        if pretrain:
            outputs, _ = self.extract_feat(batch_inputs['final_pcd_tensor'], only_current_frame=True)
            gt = torch.stack([data_sample.gt[-1] for data_sample in data_samples], dim=0)
            # gt = gt[:,:,:] - gt[:, [self.gt_norm_joint], :] # NOTE: gt normal now disabled
            outputs = outputs.type(torch.FloatTensor).to(get_device())
            loss = self.criterion(outputs, gt)
            return loss
        
        with torch.no_grad():
            out_pose_pr_list, out_pose_feat = self.extract_feat(batch_inputs['final_pcd_tensor'], only_current_frame=False, return_list=True)
        gt = torch.stack([data_sample.gt[-1] for data_sample in data_samples], dim=0)
        out_pose_pr_list.pop(-1)
        out_pose_pr_list.append(gt)
        out_pose_pr = torch.concat(out_pose_pr_list, dim=2)

        out_pose_noise_scale = torch.ones_like(gt, dtype=torch.float32, device=get_device())

        out_pose_3d = gt
        # out_pose_3d = out_pose_3d[:,:,:] - out_pose_3d[:, [self.gt_norm_joint], :] # NOTE: gt normal now disabled
        limb_list = []
        for i in range(len(self.edges_compressed)):
            limb_src = self.edges_compressed[i][0]
            limb_dst = self.edges_compressed[i][1]
            limb_vec = out_pose_3d[:, limb_dst, :] - out_pose_3d[:, limb_src, :]
            limb_list.append(limb_vec)

        out_limb = torch.stack(limb_list, dim=1)
        out_limb = torch.sqrt(torch.sum(torch.pow(out_limb, 2), dim=-1))

        out_radar = torch.ones(size=(out_pose_feat.size(0), 50, self.radar_input_c_mid), dtype=torch.float32, device=get_device()).unsqueeze(1)
        out_radar = out_radar.repeat(1, 4, 1, 1) # NOTE: Only implemented the MetaFi version of radar_mm feature
        
        x_history, targets_noise_scale, cemd, targets_3d, radar, limb_len_gt = \
        tuple(map(torch.detach, [out_pose_pr, out_pose_noise_scale, out_pose_feat, out_pose_3d, out_radar, out_limb]))

        n = targets_3d.size(0)
        x = targets_3d
        e = torch.randn_like(x)
        b = self.betas
        t = torch.randint(low=0, high=self.num_timesteps, size=(n //2 + 1,)).to(get_device())
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        e = e * (targets_noise_scale)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)

        # generate x_t (refer to DDIM equation)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()

        # predict noise
        x_curr = x
        x_all_list = torch.cat([x_history[...,:-3], x_curr], dim=-1)
        output, limb_len_pred = self.model_diff(x_all_list, t.float(), cemd, radar, limb_len=None)

        output_noise = output[:, :, -3:]

        loss_diff = (e - output_noise).square().sum(dim=(1, 2)).mean(dim=0)
        limb_loss = (limb_len_pred - limb_len_gt).abs().sum(dim=-1).mean(dim=0)
        loss_diff = loss_diff + limb_loss * 10
        return loss_diff
    
    def denorm_joints(self, skel_frame: torch.Tensor) -> torch.Tensor:
        if self.gt_norm_joint is None:
            return skel_frame
        # assert skel_frame.dim() == 1 and skel_frame.shape[0] == self.num_joints * 3
        skel_frame = skel_frame.reshape(-1, 3)
        reference = skel_frame[self.gt_norm_joint].clone()
        skel_frame = skel_frame + reference
        skel_frame[self.gt_norm_joint] = reference
        return skel_frame.flatten()
    
    @torch.no_grad()
    def predict(self, batch_inputs, data_samples):
        if self.mode == 'pred_coarse':
            out_pose_pr_curr, out_pose_feat = self.extract_feat(batch_inputs['final_pcd_tensor'], only_current_frame=True)
            for b, data_sample in enumerate(data_samples):
                data_sample.pred = out_pose_pr_curr[b]
                data_sample.pred = data_sample.pred.flatten()
                if data_sample.gt is not None:
                    data_sample.gt = data_sample.gt[-1, ...].flatten()
            return dict(
                coarse_pr_list=[out_pose_pr_curr]
            )
        assert self.mode == 'pred_fine', "Mode should either be 'pred_coarse' or 'pred_fine', but get {}".format(self.mode)
        if "out_pose_pr_past_frames" in batch_inputs:
            out_pose_pr_curr, out_pose_feat = self.extract_feat(batch_inputs['final_pcd_tensor'], only_current_frame=True)
            out_pose_pr_past_frames: List[torch.Tensor] = batch_inputs["out_pose_pr_past_frames"]
            out_pose_pr_past_frames = out_pose_pr_past_frames[-self.past_frames:]
            out_pose_pr_list = out_pose_pr_past_frames + [out_pose_pr_curr]
        else:
            out_pose_pr_list, out_pose_feat = self.extract_feat(batch_inputs['final_pcd_tensor'], only_current_frame=False, return_list=True)
            assert len(out_pose_pr_list) == self.past_frames + 1
        
                    
        out_pose_pr = torch.concat(out_pose_pr_list, dim=2)
        # x_history = out_pose_pr[:,:,:] - out_pose_pr[:, [self.gt_norm_joint], :] # NOTE: gt normal now disabled
        x_history = out_pose_pr.repeat(self.test_cfg["test_times"], 1, 1)
        cemd = out_pose_feat.repeat(self.test_cfg["test_times"], 1, 1)
        radar = torch.ones(size=(out_pose_feat.size(0), 50, self.radar_input_c_mid), dtype=torch.float32, device=get_device()).unsqueeze(1).repeat(1, 4, 1, 1).repeat(self.test_cfg["test_times"], 1, 1, 1) # NOTE: Only implemented the MetaFi version of radar_mm feature
        
        output_pose = self._predict_diffusion_steps(
            seq=self.test_cfg["seq"],
            betas=self.betas,
            eta=self.test_cfg["eta"],
            radar=radar,
            x_coarse=x_history[...,-3:],
            x_history=x_history,
            cemd=cemd
        )
        output_pose = torch.mean(output_pose[-1].reshape(self.test_cfg["test_times"], -1, self.num_joints, output_pose[-1].shape[-1]), dim=0)
        
        for b, data_sample in enumerate(data_samples):
            data_sample.pred = output_pose[b]
            data_sample.pred = data_sample.pred.flatten()
            if data_sample.gt is not None:
                data_sample.gt = data_sample.gt[-1, ...].flatten()
        
        return dict(
            coarse_pr_list=out_pose_pr_list
        )
    
    @torch.no_grad()
    def _predict_diffusion_steps(self, seq: List[int], betas, eta, radar, x_coarse, x_history, cemd):
        n = radar.size(0)
        x = x_coarse
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(get_device())
            next_t = (torch.ones(n) * j).to(get_device())
            at = compute_alpha(betas, t.long())
            at_next = compute_alpha(betas, next_t.long())
            xt = xs[-1]
            x_all_list = torch.cat([x_history[...,:-3], xt], dim=-1)
            et, limb_len_pred = self.model_diff(x_all_list, t.float(), cemd, radar, limb_len=None)
            et = et[:, :, -3:]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t)
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 * c1).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(xt) + c2 * et
            xs.append(xt_next)
        return xs
            
    def pack_input(self, data_batch_dict: dict):
        pcd_frame_list: List[Tuple[np.ndarray]] = data_batch_dict['pcd_frames'] # F x B x N x C
        batch_size = len(pcd_frame_list[0])
        frame_len = len(pcd_frame_list)
        final_pcd_frame = np.zeros((frame_len, batch_size, self.point_cloud_size, self.radar_input_c), dtype=np.float32)
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
            SkeletonDataSample(gt=skel_frame_tensor[:, :(skel_frame_tensor.shape[1] // 3) * 3].reshape(skel_frame_tensor.shape[0], -1, 3))
            for skel_frame_tensor in skel_frame_tensors
        ]
        
        batch_inputs = dict(
            final_pcd_tensor=final_pcd_tensor,
        )
        data_batch_dict.update(batch_inputs)
        
        return batch_inputs, data_sample_list