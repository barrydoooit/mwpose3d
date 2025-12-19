import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch

from mwpose3d.evaluation.metrics.simple_gtpred.simple_gtpred_visualizer import SimpleGTPredVisualizerQT
try:
    from .simple_gtpred_visualizer import SimpleGTPredVisualizer
except Exception as e:
    print("Visualizer not available.")
from ..base import BaseMetric
from mwpose3d.registry import METRICS



def round_floats(obj, decimals=2):
    """Recursively round float values in a complex data structure."""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, np.float32):
        return round_floats(float(obj))  # Convert np.float32 to float, to make JSON happy
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(elem, decimals) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(round_floats(elem, decimals) for elem in obj)
    elif isinstance(obj, np.ndarray):   # Convert ndarray to list, to make JSON happy
        return round_floats(obj.tolist(), decimals)
    else:
        return obj
    
@METRICS.register_module()
class SimpleGTPredAnalyzer(BaseMetric):
    def __init__(self,
                 keypoints_involved: list,
                 clip_keys: Optional[list] = None,
                 out_file: Optional[str] = None,
                 visualizer_cfg: Optional[dict] = None):
        self.keypoints_involved = keypoints_involved
        self.visualize = visualizer_cfg is not None
        self.report = []
        self.gt_data = []
        self.pred_data = []
        self.pcd_data = []
        self.clip_keys = clip_keys
        self.last_key = None
        self.out_file = out_file  # Prevent attribute error in evaluate
        self.out_file = out_file
        self.make_out_file(self.out_file)

        if self.visualize:
            self._make_visualizer(visualizer_cfg)
        else:
            self.visualizer = None

    @staticmethod
    def make_out_file(file: str):
        if file is None:
            return

        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)


    def _make_visualizer(self, visualizer_cfg):
        # Sensible defaults
        cfg = {
            "keypoints_involved": visualizer_cfg.get("keypoints_involved", self.keypoints_involved),
            "keypoint_for_stats": visualizer_cfg.get("keypoint_for_stats", self.keypoints_involved),
            "error_type": visualizer_cfg.get("error_type", "abs_error"),
            "window_size": visualizer_cfg.get("window_size", 100),
            "max_points_per_frame": visualizer_cfg.get("max_points_per_frame", 50000),
            "follow": visualizer_cfg.get("follow", True),
        }
        self.visualizer = SimpleGTPredVisualizerQT(**cfg)
        
    def process_sample(self, data_sample, data_batch):
        if self.clip_keys is not None:
            clip_label = "_".join([str(data_batch[k][0]) for k in self.clip_keys if k in data_batch])
            if self.last_key is None:
                self.last_key = clip_label
                print(f"Processing clip: {self.last_key}")
            elif self.last_key != clip_label:
                print(f"Finished clip: {self.last_key}")
                self.out_file = self.out_file.parent / f"{self.last_key}_gtpred.json" if self.out_file is not None else None
                print(f"Saving to: {self.out_file}")
                self.evaluate()
                self.reset()
                self.last_key = clip_label
                print(f"Processing clip: {clip_label}")
        pcd_ts = data_batch.get("pcd_ts", None)
        gt = data_sample.gt
        pred = data_sample.pred
        assert isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor), "Currently only support torch.Tensor as gt type, make conversion in the data_sample first"
        assert gt.shape == pred.shape, "gt and pred should have the same shape"
        assert pred.dim() == 1, "Currently only support single frame/batch prediction, make conversion in the data_sample first"
        frame_report = {}
        for idx, joint in enumerate(self.keypoints_involved):
            gt_joint = gt[idx * 3: (idx + 1) * 3]
            pred_joint = pred[idx * 3: (idx + 1) * 3]
            frame_report[joint] = {
                "gt_joint": gt_joint.tolist(),
                "pred_joint": pred_joint.tolist(),
                "abs_error": torch.norm(gt_joint - pred_joint).item(),
                "square_error": torch.norm(torch.pow(gt_joint - pred_joint, 2)).item(),
            }
            if pcd_ts is not None:
                frame_report[joint]["pcd_ts"] = pcd_ts[-1][-1]
        self.report.append(frame_report)
        self.gt_data.append(gt)
        self.pred_data.append(pred)
        
        if self.visualize:
            pcd_vis = data_batch["pcd_frames"][-1][0]
            self.pcd_data.append(pcd_vis)

            centroid_xyz = None
            if "track_centroid" in data_batch:
                try:
                    tc = data_batch["track_centroid"][-1][0]  # assume B=1
                    tc_np = tc.detach().cpu().numpy() if isinstance(tc, torch.Tensor) else np.asarray(tc)
                    frame_idx = len(self.gt_data) - 1
                    if tc_np.ndim == 2 and tc_np.shape[1] >= 2:
                        i = max(0, min(frame_idx, tc_np.shape[0] - 1))
                        x, y = float(tc_np[i, 0]), float(tc_np[i, 1])
                    elif tc_np.ndim == 1 and tc_np.shape[0] >= 2:
                        x, y = float(tc_np[0]), float(tc_np[1])
                    else:
                        x = y = None
                    if x is not None:
                        centroid_xyz = np.array([x, y, 1.0], dtype=np.float32)
                except Exception:
                    centroid_xyz = None

            # Live, per-frame update & event pump (no blocking mainloop)
            self.visualizer.update(
                gt_tensor=self.gt_data[-1],
                pred_tensor=self.pred_data[-1],
                pc_tensor=self.pcd_data[-1],
                track_centroid=centroid_xyz,
                frame_report=frame_report
            )
            self.visualizer.idle() 
    
    def evaluate(self, show=True):
        joint_errors = {}
        for iteration_report in self.report:
            for joint, values in iteration_report.items():
                if joint not in joint_errors:
                    joint_errors[joint] = {"abs": [], "square": []}
                joint_errors[joint]["abs"].append(values["abs_error"])
                joint_errors[joint]["square"].append(values["square_error"])
        summary = {}
        average_mae = 0.0
        average_mse = 0.0
        average_rmse = 0.0
        for joint, errors in joint_errors.items():
            mae = sum(errors["abs"]) / len(errors["abs"])
            mse = sum(errors["square"]) / len(errors["square"])
            rmse = mse ** 0.5
            summary[joint] = {"mae": mae, "rmse": rmse, "mse": mse}
            average_mae += mae
            average_mse += mse
            average_rmse += rmse
        average_mae /= len(joint_errors.items())
        average_mse /= len(joint_errors.items())
        average_rmse /= len(joint_errors.items())
        
        if not show:
            return summary
        
        print("Summary Report:")
        for joint, metrics in summary.items():
            print(f"{joint:03}: MAE = {metrics['mae']:.4f}, RMSE = {metrics['rmse']:.4f}, MSE = {metrics['mse']:.4f}")
        print(f"avg: MAE = {average_mae:.4f}, RMSE = {average_rmse:.4f}, MSE = {average_mse:.4f}")
        if self.visualize:
            self.visualizer.finalize()
        
        if self.out_file is not None:
            with open(self.out_file, 'w') as f:
                json.dump(round_floats(self.report, 3), f)
        return summary
    
    def reset(self):
        self.report = []
        self.gt_data = []
        self.pred_data = []
        self.pcd_data = []
        if self.visualizer is not None:
            self.visualizer.reset()