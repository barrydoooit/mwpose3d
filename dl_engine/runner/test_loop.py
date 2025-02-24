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
        self.gt_data = []  # Collect all ground truth data
        self.pred_data = []  # Collect all prediction data
        # Pass additional arguments: keypoint_for_stats (can be a subset) and error_type
        self.visualizer = SkeletonVisualizer(
            keypoint_involved=self.runner.model.keypoints_involved,
            connectivity=Connectivity,
            keypoint_for_stats=[0, 6, 10],
            error_type="abs_error"  # or "square_error"
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
        
        # Setup replay after collecting all data (if desired)
        self.visualizer.setup_replay(self.gt_data, self.pred_data, self.report)
        self.visualizer.root.mainloop()
        
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
        # Pass the most recent frame report so the stats plot can update individually
        self.visualizer.update(gt, pred, self.report[-1])
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

class SkeletonVisualizer:
    def __init__(self, 
                 keypoint_involved: List[int], 
                 connectivity: Dict[KeypointType, List[KeypointType]],
                 keypoint_for_stats: List[int],
                 error_type: str):
        # Convert full keypoints and stats keypoints to enum types
        self.keypoint_involved = [KeypointType(kp) for kp in keypoint_involved]
        self.connectivity = connectivity
        self.keypoint_for_stats = [KeypointType(kp) for kp in keypoint_for_stats]
        self.error_type = error_type

        self.root = tk.Tk()
        self.root.title("3D Skeleton Visualization")

        # Top frame for skeletons
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Ground Truth and Prediction frames
        frame_gt = tk.Frame(top_frame)
        frame_pred = tk.Frame(top_frame)
        frame_gt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frame_pred.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Initialize GT plot
        self.fig_gt = Figure(figsize=(5, 5))
        self.ax_gt = self.fig_gt.add_subplot(111, projection='3d')
        self._setup_axes(self.ax_gt, "Ground Truth")
        self.canvas_gt = FigureCanvasTkAgg(self.fig_gt, master=frame_gt)
        self.canvas_gt.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize Pred plot
        self.fig_pred = Figure(figsize=(5, 5))
        self.ax_pred = self.fig_pred.add_subplot(111, projection='3d')
        self._setup_axes(self.ax_pred, "Prediction")
        self.canvas_pred = FigureCanvasTkAgg(self.fig_pred, master=frame_pred)
        self.canvas_pred.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create stats frame (visible from the start)
        self.frame_stats = tk.Frame(self.root)
        self.frame_stats.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Stats figure and axis
        self.fig_stats = Figure(figsize=(10, 3))
        self.ax_stats = self.fig_stats.add_subplot(111)
        self.ax_stats.set_title(f"Error ({self.error_type}) per Frame")
        self.ax_stats.set_xlabel("Frame")
        self.ax_stats.set_ylabel("Error (m)")

        # Adjust the subplot layout to leave room for the legend on the right
        self.fig_stats.subplots_adjust(right=0.75)
        self.current_frame_line = self.ax_stats.axvline(x=0, color='red', lw=2, linestyle='--')
        # Create a separate line (and error list) for each keypoint in keypoint_for_stats
        self.stats_lines = {}
        self.stats_errors = {}  # dict mapping keypoint -> list of errors
        for stat_kp in self.keypoint_for_stats:
            line, = self.ax_stats.plot([], [], label=stat_kp.name, lw=1)
            self.stats_lines[stat_kp] = line
            self.stats_errors[stat_kp] = []

        # Place legend on the outer right side with thicker legend icons
        leg = self.ax_stats.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        for legline in leg.get_lines():
            legline.set_linewidth(4)

        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, self.frame_stats)
        self.canvas_stats.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Progress bar for replay/inspection
        self.progress_bar = tk.Scale(
            self.frame_stats,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self._on_progress_change,
            label="Frame"
        )
        self.progress_bar.pack(fill=tk.X)

        # Data storage for replay
        self.gt_data = []
        self.pred_data = []
        self.report = []
        self.num_frames = 0

    def _setup_axes(self, ax, title):
        ax.set_title(title)
        ax.set_xlim(-1, 1)
        ax.set_ylim(3, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    def _plot_skeleton(self, tensor, ax, title):
        ax.cla()
        self._setup_axes(ax, title)
        coords = {}
        color = 'blue' if title == "Ground Truth" else 'red'

        for idx, kp_enum in enumerate(self.keypoint_involved):
            x = tensor[idx*3].item() if isinstance(tensor, torch.Tensor) else tensor[idx*3]
            z = tensor[idx*3+1].item() if isinstance(tensor, torch.Tensor) else tensor[idx*3+1]
            y = tensor[idx*3+2].item() if isinstance(tensor, torch.Tensor) else tensor[idx*3+2]
            coords[kp_enum] = (x, y, z)
            ax.scatter(x, y, z, color=color)

        for kp_enum in self.keypoint_involved:
            if kp_enum in self.connectivity:
                for connected_kp in self.connectivity[kp_enum]:
                    if kp_enum in coords and connected_kp in coords:
                        xs, ys, zs = zip(coords[kp_enum], coords[connected_kp])
                        ax.plot(xs, ys, zs, color=color)

    def update(self, gt_tensor, pred_tensor, frame_report=None):
        # Update the skeleton plots
        self._plot_skeleton(gt_tensor, self.ax_gt, "Ground Truth")
        self._plot_skeleton(pred_tensor, self.ax_pred, "Prediction")
        self.canvas_gt.draw()
        self.canvas_pred.draw()
        self.root.update_idletasks()

        # Store data for replay if needed
        if frame_report is not None:
            self.report.append(frame_report)
        self.gt_data.append(gt_tensor)
        self.pred_data.append(pred_tensor)

        # For each keypoint in the stats selection, update its error for this frame.
        if frame_report is not None:
            for stat_kp in self.keypoint_for_stats:
                # Look up the error from the report if available
                error = frame_report.get(stat_kp, {}).get(self.error_type, None)
                if error is None:
                    # Fallback: compute error manually if missing.
                    if stat_kp in self.keypoint_involved:
                        idx = self.keypoint_involved.index(stat_kp)
                        gt_joint = gt_tensor[idx*3: idx*3+3]
                        pred_joint = pred_tensor[idx*3: idx*3+3]
                        if self.error_type == "abs_error":
                            error = torch.norm(gt_joint - pred_joint).item()
                        elif self.error_type == "square_error":
                            error = torch.norm((gt_joint - pred_joint)**2).item()
                        else:
                            error = torch.norm(gt_joint - pred_joint).item()
                    else:
                        error = 0.0
                self.stats_errors[stat_kp].append(error)
        else:
            # If no report provided, compute errors manually for each keypoint.
            for stat_kp in self.keypoint_for_stats:
                if stat_kp in self.keypoint_involved:
                    idx = self.keypoint_involved.index(stat_kp)
                    gt_joint = gt_tensor[idx*3: idx*3+3]
                    pred_joint = pred_tensor[idx*3: idx*3+3]
                    if self.error_type == "abs_error":
                        error = torch.norm(gt_joint - pred_joint).item()
                    elif self.error_type == "square_error":
                        error = torch.norm((gt_joint - pred_joint)**2).item()
                    else:
                        error = torch.norm(gt_joint - pred_joint).item()
                    self.stats_errors[stat_kp].append(error)

        # All keypoint error lists should have the same length now
        self.num_frames = len(next(iter(self.stats_errors.values()))) if self.stats_errors else 0

        # Update each keypoint's line in the stats plot
        for stat_kp in self.keypoint_for_stats:
            xdata = list(range(len(self.stats_errors[stat_kp])))
            self.stats_lines[stat_kp].set_data(xdata, self.stats_errors[stat_kp])

        self.ax_stats.relim()
        self.ax_stats.autoscale_view()

        # Update the progress bar
        self.progress_bar.config(to=self.num_frames - 1)
        self.progress_bar.set(self.num_frames - 1)
        self.canvas_stats.draw()

    def _on_progress_change(self, value):
        frame_idx = int(float(value))
        self._update_replay(frame_idx)
        self.current_frame_line.set_xdata([frame_idx, frame_idx])
        self.canvas_stats.draw()

    def _update_replay(self, frame_idx):
        if frame_idx < len(self.gt_data) and frame_idx < len(self.pred_data):
            self._plot_skeleton(self.gt_data[frame_idx], self.ax_gt, "Ground Truth")
            self._plot_skeleton(self.pred_data[frame_idx], self.ax_pred, "Prediction")
            self.canvas_gt.draw()
            self.canvas_pred.draw()

    def setup_replay(self, gt_data, pred_data, report):
        self.gt_data = gt_data
        self.pred_data = pred_data
        self.report = report
