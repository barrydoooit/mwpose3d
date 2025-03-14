import torch
import json
import numpy as np
from typing import Callable, Literal, Optional, Tuple, List, Dict

from ..base import METRICS, BaseMetric



@METRICS.register_module()
class PivotRotationAnalyzer(BaseMetric):
    def __init__(self,
                 bones: Tuple[Tuple[int, int], ...],
                 keypoint_involved: Tuple[int, ...],
                 pos_pivot: Literal['first', 'mid', 'last'] = 'mid',
                 window_size_frames: int = 1,
                 log_name: str = 'pos_angle_norm.json',
                 output_dir: str = 'exp_data/test_logs'):
        self.keypoint_involved = sorted(keypoint_involved)
        self.bones = bones
        self.pos_pivot = pos_pivot
        self.window_size_frames = window_size_frames
        self.log_name = log_name
        self.output_dir = output_dir
        self.get_pos_pivot: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None
        if self.pos_pivot == 'mid':
            self.get_pos_pivot = lambda joint1, joint2: (joint1 + joint2) / 2.0
        elif self.pos_pivot == 'first':
            self.get_pos_pivot = lambda joint1, joint2: joint1
        elif self.pos_pivot == 'last':
            self.get_pos_pivot = lambda joint1, joint2: joint2
        
        self.pos_errors: Dict[Tuple[int, int], List[np.ndarray]] = {
            bone: [] for bone in bones
        }
        self.angle_errors: Dict[Tuple[int, int], List[np.ndarray]] = {
            bone: [] for bone in bones
        }
        
        self.gt_data: List[torch.Tensor] = []
        self.pred_data: List[torch.Tensor] = []
        self.report = []
        
    def process_sample(self, data_sample):
        gt = data_sample.gt
        pred = data_sample.pred
        assert isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor), "Currently only support torch.Tensor as gt type, make conversion in the data_sample first"
        assert gt.shape == pred.shape, "gt and pred should have the same shape"
        
        frame_report = {}
        for bone in self.bones:
            joint1, joint2 = bone
            joint1_idx, joint2_idx = self.keypoint_involved.index(joint1), self.keypoint_involved.index(joint2)
            gt_joint1 = gt[joint1_idx * 3: (joint1_idx + 1) * 3]
            gt_joint2 = gt[joint2_idx * 3: (joint2_idx + 1) * 3]
            pred_joint1 = pred[joint1_idx * 3: (joint1_idx + 1) * 3]
            pred_joint2 = pred[joint2_idx * 3: (joint2_idx + 1) * 3]
            
            pivot_gt = self.get_pos_pivot(gt_joint1, gt_joint2)
            pivot_pred = self.get_pos_pivot(pred_joint1, pred_joint2)
            pos_error = (pivot_pred - pivot_gt).detach().cpu().numpy()
            self.pos_errors[bone].append(pos_error)
            
            v_gt = (gt_joint2 - gt_joint1).detach().cpu().numpy()
            v_pred = (pred_joint2 - pred_joint1).detach().cpu().numpy()
            norm_gt = np.linalg.norm(v_gt)
            norm_pred = np.linalg.norm(v_pred)
            if norm_gt == 0 or norm_pred == 0:
                angle_error_vector = np.zeros(3)
            else:
                v_gt_unit = v_gt / norm_gt
                v_pred_unit = v_pred / norm_pred
                dot_val = np.dot(v_pred_unit, v_gt_unit)
                dot_val = np.clip(dot_val, -1.0, 1.0)
                theta = np.arccos(dot_val)
                cross = np.cross(v_pred_unit, v_gt_unit)
                norm_cross = np.linalg.norm(cross)
                if norm_cross == 0:
                    axis = np.zeros(3)
                else:
                    axis = cross / norm_cross
                angle_error_vector = axis * theta
            self.angle_errors[bone].append(angle_error_vector)
            
            frame_report[f'seg_{bone}'] = {
                "pos_error": pos_error.tolist(),
                "angle_error": angle_error_vector.tolist()
            }
            
        self.report.append(frame_report)
        self.gt_data.append(gt)
        self.pred_data.append(pred)
    
    def _get_pos_error_scalars(self, pos_error_list: List[np.ndarray], mapping=np.linalg.norm):
        pos_error_scalars = []
        for i in range(len(pos_error_list)):
            pos_error_scalars.append(mapping(pos_error_list[i]))
        if self.window_size_frames < len(pos_error_scalars):
            windowed_avgs = []
            for i in range(len(pos_error_scalars) - self.window_size_frames + 1):
                windowed_avgs.append(np.mean(pos_error_scalars[i: i + self.window_size_frames]))
            return np.asarray(windowed_avgs)
        else:
            return float(np.mean(pos_error_scalars))
    
    def _get_angle_error_scalars(self, angle_error_list: List[np.ndarray], mapping=np.linalg.norm):
        angle_error_scalars = []
        for i in range(len(angle_error_list)):
            angle_error_scalars.append(mapping(angle_error_list[i]))
        if self.window_size_frames < len(angle_error_scalars):
            windowed_avgs = []
            for i in range(len(angle_error_scalars) - self.window_size_frames + 1):
                windowed_avgs.append(np.mean(angle_error_scalars[i: i + self.window_size_frames]))
            return np.asarray(windowed_avgs)
        else:
            return float(np.mean(angle_error_scalars))
        
    def evaluate(self, show=True):
        summary = {}
        for bone in self.bones:
            pos_error_list = self.pos_errors[bone]
            angle_error_list = self.angle_errors[bone]
            pos_error_scalars = self._get_pos_error_scalars(pos_error_list)
            angle_error_scalars = self._get_angle_error_scalars(angle_error_list)
            summary[f'seg_{bone}'] = {
                "pos_error_scalar": pos_error_scalars.tolist(),
                "angle_error_scalar": angle_error_scalars.tolist()
            }
        
        # TODO: Implement hooking management in runner to handle all data dumping
        output_dir = self.output_dir
        file_name = self.log_name

        with open(f'{output_dir}/{file_name}', 'w') as f:
            json.dump(summary, f)
        
        return summary

            
        