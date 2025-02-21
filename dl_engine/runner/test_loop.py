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
class TestLoop(BaseLoop):
    def __init__(
        self,
        runner: 'Runner',
        dataloader: Union[DataLoader, Dict],
    ):
        super().__init__(runner, dataloader)
        self._iter = 0
        self.report = []
        self.visualizer = SkeletonVisualizer(
            keypoint_involved=self.runner.model.keypoints_involved,
            connectivity=Connectivity
        )

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
        self.visualizer.update(gt, pred)
        self._iter += 1
    
    def evaluate(self, data_sample: SkeletonDataSample) -> None:
        keypoint_involved = self.runner.model.keypoints_involved
        gt = data_sample.gt
        pred = data_sample.pred
        assert gt.shape == pred.shape
        
        report = dict()

        for idx, joint in enumerate(keypoint_involved):
            gt_joint: torch.Tensor = gt[idx*3:idx*3+3]
            pred_joint: torch.Tensor = pred[idx*3:idx*3+3]
            report[joint] = dict(
                gt_joint=gt_joint.tolist(),
                pred_joint=pred_joint.tolist(),
                abs_error=torch.norm(gt_joint - pred_joint).item(),
                square_error=torch.norm(torch.pow(gt_joint - pred_joint, 2)).item()
            )
        self.report.append(report)
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
    

class SkeletonVisualizer:
    """
    Creates a tkinter window with two dynamic matplotlib figures:
    one for displaying the ground truth (GT) and one for the prediction.
    """
    def __init__(self, keypoint_involved: List[KeypointType], connectivity: Dict[KeypointType, List[KeypointType]]):
        self.keypoint_involved = keypoint_involved
        self.connectivity = connectivity
        
        self.root = tk.Tk()
        self.root.title("Skeleton Visualization")

        # Create two frames side by side for GT and Prediction.
        frame_gt = tk.Frame(self.root)
        frame_pred = tk.Frame(self.root)
        frame_gt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frame_pred.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Configure Ground Truth figure.
        self.fig_gt = Figure(figsize=(5, 5))
        self.ax_gt = self.fig_gt.add_subplot(111)
        self.ax_gt.set_title("Ground Truth")
        self.ax_gt.set_xlim(-1, 1)
        self.ax_gt.set_ylim(-1, 1)
        self.ax_gt.invert_yaxis()

        self.canvas_gt = FigureCanvasTkAgg(self.fig_gt, master=frame_gt)
        self.canvas_gt.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Configure Prediction figure.
        self.fig_pred = Figure(figsize=(5, 5))
        self.ax_pred = self.fig_pred.add_subplot(111)
        self.ax_pred.set_title("Prediction")
        self.ax_pred.set_xlim(-1, 1)
        self.ax_pred.set_ylim(-1, 1)
        self.ax_pred.invert_yaxis()

        self.canvas_pred = FigureCanvasTkAgg(self.fig_pred, master=frame_pred)
        self.canvas_pred.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update(self, gt_tensor, pred_tensor):
        """
        Update the two figures with new keypoint data.
        Assumes that gt_tensor and pred_tensor are (3*n)-length arrays (or tensors)
        where every triplet represents (x, y, z) for a keypoint.
        """
        # Clear previous drawings.
        self.ax_gt.cla()
        self.ax_pred.cla()

        self.ax_gt.set_title("Ground Truth")
        self.ax_gt.set_xlim(-1, 1)
        self.ax_gt.set_ylim(-1, 1)
        self.ax_gt.invert_yaxis()

        self.ax_pred.set_title("Prediction")
        self.ax_pred.set_xlim(-1, 1)
        self.ax_pred.set_ylim(-1, 1)
        self.ax_pred.invert_yaxis()

        # Extract (x, y) coordinates for each keypoint (ignoring z).
        gt_coords = {}
        pred_coords = {}
        n = len(self.keypoint_involved)
        for idx, kp in enumerate(self.keypoint_involved):
            # Convert tensor elements to Python scalars if needed.
            x_gt = gt_tensor[idx*3].item() if isinstance(gt_tensor, torch.Tensor) else gt_tensor[idx*3]
            y_gt = gt_tensor[idx*3+1].item() if isinstance(gt_tensor, torch.Tensor) else gt_tensor[idx*3+1]
            gt_coords[kp] = (x_gt, y_gt)
            x_pred = pred_tensor[idx*3].item() if isinstance(pred_tensor, torch.Tensor) else pred_tensor[idx*3]
            y_pred = pred_tensor[idx*3+1].item() if isinstance(pred_tensor, torch.Tensor) else pred_tensor[idx*3+1]
            pred_coords[kp] = (x_pred, y_pred)
            
            # Plot keypoints.
            self.ax_gt.scatter(x_gt, y_gt, color='blue')
            self.ax_pred.scatter(x_pred, y_pred, color='red')
            
            # Optionally, label the keypoints.
            self.ax_gt.text(x_gt, y_gt, kp.name, fontsize=8)
            self.ax_pred.text(x_pred, y_pred, kp.name, fontsize=8)

        # Draw lines for connectivity.
        for kp in self.keypoint_involved:
            if kp in self.connectivity:
                for connected_kp in self.connectivity[kp]:
                    if kp in gt_coords and connected_kp in gt_coords:
                        x_vals = [gt_coords[kp][0], gt_coords[connected_kp][0]]
                        y_vals = [gt_coords[kp][1], gt_coords[connected_kp][1]]
                        self.ax_gt.plot(x_vals, y_vals, color='blue')
                    if kp in pred_coords and connected_kp in pred_coords:
                        x_vals = [pred_coords[kp][0], pred_coords[connected_kp][0]]
                        y_vals = [pred_coords[kp][1], pred_coords[connected_kp][1]]
                        self.ax_pred.plot(x_vals, y_vals, color='red')

        self.canvas_gt.draw()
        self.canvas_pred.draw()
        self.root.update_idletasks()
        self.root.update()