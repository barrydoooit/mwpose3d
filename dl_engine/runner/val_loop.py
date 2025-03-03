from typing import TYPE_CHECKING, Dict, List, Sequence, Union
import torch
from torch.utils.data import DataLoader

from dl_engine.dataset.skel_data_sample import SkeletonDataSample
from kinect_toolkits.kinectData import KeypointType, Connectivity
from .base_loop import BaseLoop, LOOPS

import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

if TYPE_CHECKING:
    from dl_engine.runner.runner import Runner

@LOOPS.register_module()
class ValLoop(BaseLoop):
    def __init__(
        self,
        runner: 'Runner',
        dataloader: Union[DataLoader, Dict],
    ):
        super().__init__(runner, dataloader)
        self._iter = 0
        self.report = []
        self.gt_data = []  # Collect all ground truth data
        self.pred_data = []  # Collect all prediction data

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        self._run_epoch()
        summary = self._summarize_report()
        print("Summary Report:")
        for joint, metrics in summary.items():
            print(f"{joint}: MAE = {metrics['mae']:.4f}, RMSE = {metrics['rmse']:.4f}, MSE = {metrics['mse']:.4f}")
        
        
        return self.runner.model
    
    def _run_epoch(self) -> None:
        self.runner.model.eval()
        with torch.no_grad():
            for idx, data_batch in enumerate(self.dataloader):
                self._run_iter(idx, data_batch)

    def _run_iter(self, idx: int, data_batch: dict) -> None:
        assert hasattr(self.runner.model, 'pack_input')
        batch_inputs, data_samples = self.runner.model.pack_input(data_batch)
        assert len(data_samples) == 1, 'TestLoop only supports batch_size=1'
        tensor = self.runner.model(batch_inputs, data_samples, mode='predict')
        gt, pred = self.evaluate(data_samples[0])
        self.gt_data.append(gt)  # Store gt
        self.pred_data.append(pred)  # Store pred
        self._iter += 1
    
    def evaluate(self, data_sample: SkeletonDataSample):
        keypoint_involved = self.runner.model.keypoints_involved
        gt = data_sample.gt
        pred = data_sample.pred
        assert gt.shape == pred.shape
        
        frame_report = dict()

        for idx, joint in enumerate(keypoint_involved):
            gt_joint: torch.Tensor = gt[idx*3:idx*3+3]
            pred_joint: torch.Tensor = pred[idx*3:idx*3+3]
            frame_report[joint] = dict(
                gt_joint=gt_joint.tolist(),
                pred_joint=pred_joint.tolist(),
                abs_error=torch.norm(gt_joint - pred_joint).item(),
                square_error=torch.norm(torch.pow(gt_joint - pred_joint, 2)).item()
            )
        self.report.append(frame_report)
        return gt, pred

    def _summarize_report(self) -> dict:
        joint_errors = {}
        for iteration_report in self.report:
            for joint, values in iteration_report.items():
                if joint not in joint_errors:
                    joint_errors[joint] = {'abs': [], 'square': []}
                joint_errors[joint]['abs'].append(values['abs_error'])
                joint_errors[joint]['square'].append(values['square_error'])

        summary = {}
        for joint, errors in joint_errors.items():
            mae = sum(errors['abs']) / len(errors['abs'])
            mse = sum(errors['square']) / len(errors['square'])
            rmse = mse ** 0.5
            summary[joint] = {"mae": mae, "rmse": rmse, "mse": mse}

        return summary