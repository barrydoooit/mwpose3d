from typing import Optional
import torch

from .simple_gtpred_visualizer import SimpleGTPredVisualizer
from ..base import BaseMetric
from mwpose3d.registry import METRICS


@METRICS.register_module()
class SimpleGTPredAnalyzer(BaseMetric):
    def __init__(self,
                 keypoint_involved: list,
                 visualizer_cfg: Optional[dict] = None):
        self.keypoint_involved = keypoint_involved
        self.visuzalize = visualizer_cfg is not None
        self.report = []
        self.gt_data = []
        self.pred_data = []
        self.pcd_data = []
        if self.visuzalize:
            self._make_visualizer(visualizer_cfg)
        else:
            self.visualizer = None
            
    def _make_visualizer(self, visualizer_cfg):
        self.visualizer = SimpleGTPredVisualizer(
            keypoint_involved=visualizer_cfg.get("keypoint_involved", self.keypoint_involved),
            keypoint_for_stats=visualizer_cfg.get("keypoint_for_stats", self.keypoint_involved),
            error_type=visualizer_cfg.get("error_type", "abs_error")
        )
        
    def process_sample(self, data_sample, data_batch):
        gt = data_sample.gt
        pred = data_sample.pred
        pcd = data_batch.get("final_pcd_tensor")[:, -2:, ...]
        assert isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor), "Currently only support torch.Tensor as gt type, make conversion in the data_sample first"
        assert gt.shape == pred.shape, "gt and pred should have the same shape"
        frame_report = {}
        for idx, joint in enumerate(self.keypoint_involved):
            gt_joint = gt[idx * 3: (idx + 1) * 3]
            pred_joint = pred[idx * 3: (idx + 1) * 3]
            frame_report[joint] = {
                "gt_joint": gt_joint.tolist(),
                "pred_joint": pred_joint.tolist(),
                "abs_error": torch.norm(gt_joint - pred_joint).item(),
                "square_error": torch.norm(torch.pow(gt_joint - pred_joint, 2)).item()
            }
        self.report.append(frame_report)
        self.gt_data.append(gt)
        self.pred_data.append(pred)
        self.pcd_data.append(pcd)
        
        if self.visuzalize:
            self.visualizer.update(gt, pred, pcd, frame_report)
    
    def evaluate(self, show=True):
        joint_errors = {}
        for iteration_report in self.report:
            for joint, values in iteration_report.items():
                if joint not in joint_errors:
                    joint_errors[joint] = {"abs": [], "square": []}
                joint_errors[joint]["abs"].append(values["abs_error"])
                joint_errors[joint]["square"].append(values["square_error"])
        summary = {}
        for joint, errors in joint_errors.items():
            mae = sum(errors["abs"]) / len(errors["abs"])
            mse = sum(errors["square"]) / len(errors["square"])
            rmse = mse ** 0.5
            summary[joint] = {"mae": mae, "rmse": rmse, "mse": mse}
        
        if not show:
            if self.visualize:
                self.visualizer.finalize(self.gt_data, self.pred_data, self.report, pc_data=self.pcd_data)
            return summary
        
        print("Summary Report:")
        for joint, metrics in summary.items():
            print(f"{joint}: MAE = {metrics['mae']:.4f}, RMSE = {metrics['rmse']:.4f}, MSE = {metrics['mse']:.4f}")   
        if self.visuzalize:
            self.visualizer.finalize(self.gt_data, self.pred_data, self.report, pc_data=self.pcd_data)
        return summary
    
    def reset(self):
        self.report = []
        self.gt_data = []
        self.pred_data = []