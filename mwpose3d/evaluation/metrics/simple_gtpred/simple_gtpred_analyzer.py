import json
from pathlib import Path
from typing import Optional
import torch

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
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(elem, decimals) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(round_floats(elem, decimals) for elem in obj)
    else:
        return obj


@METRICS.register_module()
class SimpleGTPredAnalyzer(BaseMetric):
    def __init__(
        self,
        keypoints_involved: list,
        out_file: Optional[str] = None,
        visualizer_cfg: Optional[dict] = None,
    ):
        self.keypoints_involved = keypoints_involved
        self.visuzalize = visualizer_cfg is not None
        self.report = []
        self.gt_data = []
        self.pred_data = []
        self.pcd_data = []

        self.out_file = out_file
        self.make_out_file(self.out_file)

        if self.visuzalize:
            raise NotImplementedError(
                "Visualizer is not correctly maintained, as the pcd data structure is "
                "heterogeneous from different models."
            )
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
        self.visualizer = SimpleGTPredVisualizer(
            keypoints_involved=visualizer_cfg.get(
                "keypoints_involved", self.keypoints_involved
            ),
            keypoint_for_stats=visualizer_cfg.get(
                "keypoint_for_stats", self.keypoints_involved
            ),
            error_type=visualizer_cfg.get("error_type", "abs_error"),
        )

    def process_sample(self, data_sample, data_batch):
        gt = data_sample.gt
        pred = data_sample.pred
        assert isinstance(gt, torch.Tensor) and isinstance(
            pred, torch.Tensor
        ), "Currently only support torch.Tensor as gt type, make conversion in the data_sample first"
        assert gt.shape == pred.shape, "gt and pred should have the same shape"
        frame_report = {}
        for idx, joint in enumerate(self.keypoints_involved):
            gt_joint = gt[idx * 3 : (idx + 1) * 3]
            pred_joint = pred[idx * 3 : (idx + 1) * 3]
            frame_report[joint] = {
                "gt_joint": gt_joint.tolist(),
                "pred_joint": pred_joint.tolist(),
                "abs_error": torch.norm(gt_joint - pred_joint).item(),
                "square_error": torch.norm(torch.pow(gt_joint - pred_joint, 2)).item(),
            }
        self.report.append(frame_report)
        self.gt_data.append(gt)
        self.pred_data.append(pred)

        if self.visuzalize:
            pcd = data_batch.get("final_pcd_tensor")[:, -2:, ...]
            self.pcd_data.append(pcd)
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
        average_mae = 0.0
        average_mse = 0.0
        average_rmse = 0.0
        for joint, errors in joint_errors.items():
            mae = sum(errors["abs"]) / len(errors["abs"])
            mse = sum(errors["square"]) / len(errors["square"])
            rmse = mse**0.5
            summary[joint] = {"mae": mae, "rmse": rmse, "mse": mse}
            average_mae += mae
            average_mse += mse
            average_rmse += rmse
        average_mae /= len(joint_errors.items())
        average_mse /= len(joint_errors.items())
        average_rmse /= len(joint_errors.items())

        if not show:
            if self.visualize:
                self.visualizer.finalize(
                    self.gt_data, self.pred_data, self.report, pc_data=self.pcd_data
                )
            return summary

        print("Summary Report:")
        for joint, metrics in summary.items():
            print(
                f"{joint:03}: MAE = {metrics['mae']:.4f}, RMSE = {metrics['rmse']:.4f}, MSE = {metrics['mse']:.4f}"
            )
        print(
            f"avg: MAE = {average_mae:.4f}, RMSE = {average_rmse:.4f}, MSE = {average_mse:.4f}"
        )
        if self.visuzalize:
            self.visualizer.finalize(
                self.gt_data, self.pred_data, self.report, pc_data=self.pcd_data
            )

        if self.out_file is not None:
            with open(self.out_file, "w") as f:
                json.dump(round_floats(self.report, 3), f)
        return summary

    def reset(self):
        self.report = []
        self.gt_data = []
        self.pred_data = []
