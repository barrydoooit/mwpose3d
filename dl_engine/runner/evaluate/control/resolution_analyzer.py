import math
from typing import List, Optional
import numpy as np
import torch
import kinect_toolkits as kntk
from ..base import METRICS, BaseMetric



@METRICS.register_module()
class ControlResolutionAnalyzer(BaseMetric):
    def __init__(self,
                 keypoint_involved: list,
                 controlled_keypoints: list,
                 seg_length_n: int,
                 seg_correct_threshold: float,
                 acc_guarantee_k: List[float],
                 motion_range_clip_ratio_xyz: List[float] = [0.9, 0.9, 0.9],
                 voxel_unit: float = 0.01,
                 ):
        self.keypoint_involved = keypoint_involved
        self.controlled_keypoints = controlled_keypoints
        self.seg_length_n = seg_length_n
        self.seg_correct_threshold = seg_correct_threshold
        self.acc_guarantee_k = acc_guarantee_k
        self.motion_range_clip_ratio_xyz = motion_range_clip_ratio_xyz
        self.voxel_unit = voxel_unit
        self.report = []
        
    def process_sample(self, data_sample):
        gt = data_sample.gt
        pred = data_sample.pred
        assert isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor), "Currently only support torch.Tensor as gt type, make conversion in the data_sample first"
        assert gt.shape == pred.shape, "gt and pred should have the same shape"
        frame_report = {}
        for idx, joint in enumerate(self.keypoint_involved):
            gt_joint = gt[idx * 3: (idx + 1) * 3]
            pred_joint = pred[idx * 3: (idx + 1) * 3]
            frame_report[joint] = {
                "gt_joint": gt_joint.tolist(),
                "pred_joint": pred_joint.tolist(),
            }
        self.report.append(frame_report)

        ...
    
    def evaluate(self, show=True):
        print(f"Evaluating Using {self.__class__.__name__}...")
        joint_data = {
            joint: {"gt": [], "pred": []} for joint in self.controlled_keypoints
        }
        for frame in self.report:
            for joint in self.controlled_keypoints:
                joint_data[joint]["gt"].append(frame[joint]["gt_joint"])
                joint_data[joint]["pred"].append(frame[joint]["pred_joint"])
         
        for joint in self.controlled_keypoints:       
            for axis_idx in range(3):
                axis_max = np.max(np.array(joint_data[joint]["gt"])[:, axis_idx])
                axis_min = np.min(np.array(joint_data[joint]["gt"])[:, axis_idx])
                axis_range = axis_max - axis_min
                axis_max_clipped = axis_max - axis_range * (1 - self.motion_range_clip_ratio_xyz[axis_idx]) / 2
                axis_min_clipped = axis_min + axis_range * (1 - self.motion_range_clip_ratio_xyz[axis_idx]) / 2
                for frame in joint_data[joint]["gt"]:
                    frame[axis_idx] = min(max(frame[axis_idx], axis_min_clipped), axis_max_clipped)
                for frame in joint_data[joint]["pred"]:
                    frame[axis_idx] = min(max(frame[axis_idx], axis_min_clipped), axis_max_clipped)
        
        results = {}
        def analyze_single(joint):
            gt_points = np.array(joint_data[joint]["gt"])
            pred_points = np.array(joint_data[joint]["pred"])
            axis_results = {}
            for axis, axis_name in zip(range(3), ["azimuth (x)", "distance (y)", "elevation (z)"]):
                gt_vals = gt_points[:, axis]
                pred_vals = pred_points[:, axis]
                axis_results[axis_name] = self.find_min_voxel_reolutionss_for_guarantees_1d(
                    gt_vals, 
                    pred_vals, 
                    self.acc_guarantee_k if not isinstance(self.acc_guarantee_k, float) else [self.acc_guarantee_k]
                )
            results[joint] = axis_results
        for joint in self.controlled_keypoints:
            analyze_single(joint)
        if show:
            print(self.format_results(results))
        return results
                
    def format_results(self, results):
        lines = []
        header = (
            f"{'Keypoint': <15} "
            f"{'Acc Guarantee': <15} "
            f"{'Azimuth Res (x)': <15} "
            f"{'Distance Res (y)': <15} "
            f"{'Elevation Res (z)': <15}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for keypoint, axis_data in results.items():
            guarantee_keys = list(axis_data["azimuth (x)"].keys())
            for acc in guarantee_keys:
                azimuth_res = axis_data["azimuth (x)"][acc][0]
                distance_res = axis_data["distance (y)"][acc][0]
                elevation_res = axis_data["elevation (z)"][acc][0]
                
                azimuth_res_str = f"{azimuth_res:.2f}" if azimuth_res is not None else "N/A"
                distance_res_str = f"{distance_res:.2f}" if distance_res is not None else "N/A"
                elevation_res_str = f"{elevation_res:.2f}" if elevation_res is not None else "N/A"
                
                line = (
                    f"{kntk.KeypointType(int(keypoint)).name.lower().capitalize(): <15} "
                    f"{(str(round(acc * 100, 0)) + '%'): <15} "
                    f"{azimuth_res_str: <15} "
                    f"{distance_res_str: <15} "
                    f"{elevation_res_str: <15}"
                )
                lines.append(line)
        return "\n".join(lines)
    
    def reset(self):
        self.report = []
    
    
    
    
    def _compute_windowed_accuracy_1d(self, gt_vals, pred_vals, voxel_length):
        optimal_offset, correct_preds = self._find_optimal_offset(gt_vals, pred_vals, voxel_length)
        discretized_gt = list(map(lambda x: math.floor((x - optimal_offset) / voxel_length), gt_vals))
        discretized_pred = list(map(lambda x: math.floor((x - optimal_offset) / voxel_length), pred_vals))
        n = self.seg_length_n
        total_segs = len(discretized_gt) - n + 1
        count_full_correct = 0
        for i in range(total_segs):
            window_gt = discretized_gt[i: i + n]
            window_pred = discretized_pred[i: i + n]
            correct_count = sum(x == y for x, y in zip(window_gt, window_pred))
            if correct_count >= math.ceil(n * self.seg_correct_threshold):
                count_full_correct += 1
        accuracy = count_full_correct / total_segs if total_segs > 0 else 0.0
        return accuracy, optimal_offset
        
    def _find_optimal_offset(self, gt_vals, pred_vals, voxel_length, eps=1e-9):
        pairs = list(zip(gt_vals, pred_vals))
        events = []
        eligible_pairs = 0
        
        for g, p in pairs:
            a, b = (g, p) if g < p else (p, g)
            d = b - a
            if d > voxel_length:
                continue
            eligible_pairs += 1
            
            r_a = a - math.floor(a / voxel_length) * voxel_length
            r_b = b - math.floor(b / voxel_length) * voxel_length
            
            if r_a <= r_b:
                events.append((r_a, 1))
                events.append((r_b, -1))
            else:
                events.append((r_a, 1))
                # events.append((voxel_length, -1))
                events.append((0, 1))
                events.append((r_b, -1))
        
        events.sort(key=lambda e: (e[0], -e[1]))
        
        current_forbidden = 0
        min_forbidden = int(1e9)
        optimal_offset = 0
        
        i = 0
        n = len(events)
        while i < n:
            offset = events[i][0]
            group_delta = 0
            while i < n and abs(events[i][0] - offset) < eps:
                group_delta += events[i][1]
                i += 1
            current_forbidden += group_delta
            if current_forbidden < min_forbidden:
                min_forbidden = current_forbidden
                optimal_offset = offset + eps
                
        
        max_intact_pairs = eligible_pairs - min_forbidden
        # print(f"max_intact_pairs: {max_intact_pairs}", f"eligible_pairs: {eligible_pairs}", f"min_forbidden: {min_forbidden}")
        return optimal_offset, max_intact_pairs
    
    def find_min_voxel_reolutionss_for_guarantees_1d(self, gt_vals, pred_vals, acc_grt_list):
        results = {acc_grt: None for acc_grt in acc_grt_list}
        for acc_grt in acc_grt_list:
            res, offset = self._find_min_voxel_resolutions_for_single_guarantee_1d(gt_vals, pred_vals, acc_grt)
            results[acc_grt] = (res, offset)
        return results
    
    def _find_min_voxel_resolutions_for_single_guarantee_1d(self, gt_vals, pred_vals, acc_grt):
        left = self.voxel_unit
        min_val = min(gt_vals.min(), pred_vals.min())
        max_val = max(gt_vals.max(), pred_vals.max())
        range_val = max_val - min_val
        right = max(left * 2, 3.0)
        
        min_voxel_res = None
        best_offset = None
        _tol = self.voxel_unit / 10.0
        
        while right - left > _tol:
            mid = (left + right) / 2.0
            accuracy, offset = self._compute_windowed_accuracy_1d(gt_vals, pred_vals, mid)
            accuracy_satisfied = accuracy >= acc_grt
            if accuracy_satisfied:
                min_voxel_res = mid
                best_offset = offset
                right = mid
            else:
                left = mid
        return min_voxel_res, best_offset