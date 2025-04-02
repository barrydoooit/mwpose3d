import torch
from kinect_toolkits.kinectData import KeypointType, Connectivity

import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt  # Needed for colormap

class SimpleGTPredVisualizer:
    def __init__(self, 
                 keypoint_involved: list, 
                 keypoint_for_stats: list,
                 error_type: str):
        # Convert keypoints to enum types (assuming KeypointType is callable)
        self.keypoint_involved = [KeypointType(kp) for kp in keypoint_involved]
        self.keypoint_for_stats = [KeypointType(kp) for kp in keypoint_for_stats]
        self.error_type = error_type

        self.root = tk.Tk()
        self.root.title("3D Skeleton Visualization")

        # Top frame for skeletons and point cloud
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create three frames: Ground Truth, Point Cloud, and Prediction
        frame_gt = tk.Frame(top_frame)
        frame_pc = tk.Frame(top_frame)
        frame_pred = tk.Frame(top_frame)
        frame_gt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frame_pc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frame_pred.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Initialize Ground Truth plot
        self.fig_gt = Figure(figsize=(5, 5))
        self.ax_gt = self.fig_gt.add_subplot(111, projection='3d')
        self._setup_axes(self.ax_gt, "Ground Truth")
        self.canvas_gt = FigureCanvasTkAgg(self.fig_gt, master=frame_gt)
        self.canvas_gt.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize Point Cloud plot
        self.fig_pc = Figure(figsize=(5, 5))
        self.ax_pc = self.fig_pc.add_subplot(111, projection='3d')
        self._setup_axes(self.ax_pc, "Point Cloud")
        self.canvas_pc = FigureCanvasTkAgg(self.fig_pc, master=frame_pc)
        self.canvas_pc.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize Prediction plot
        self.fig_pred = Figure(figsize=(5, 5))
        self.ax_pred = self.fig_pred.add_subplot(111, projection='3d')
        self._setup_axes(self.ax_pred, "Prediction")
        self.canvas_pred = FigureCanvasTkAgg(self.fig_pred, master=frame_pred)
        self.canvas_pred.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize persistent plot objects for skeletons
        self.connectivity_pairs = []
        for kp in self.keypoint_involved:
            if kp in Connectivity:
                for connected in Connectivity[kp]:
                    if connected in self.keypoint_involved:
                        self.connectivity_pairs.append((kp, connected))
                        
        # For Ground Truth axis
        self.scatter_gt = self.ax_gt.scatter([], [], [], color='blue')
        self.lines_gt = []
        for pair in self.connectivity_pairs:
            line, = self.ax_gt.plot([], [], [], color='blue', lw=1)
            self.lines_gt.append((pair, line))
                        
        # For Prediction axis
        self.scatter_pred = self.ax_pred.scatter([], [], [], color='red')
        self.lines_pred = []
        for pair in self.connectivity_pairs:
            line, = self.ax_pred.plot([], [], [], color='red', lw=1)
            self.lines_pred.append((pair, line))
            
        # Create stats frame (visible from the start)
        self.frame_stats = tk.Frame(self.root)
        self.frame_stats.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Stats figure and axis
        self.fig_stats = Figure(figsize=(10, 3))
        self.ax_stats = self.fig_stats.add_subplot(111)
        self.ax_stats.set_title(f"Error ({self.error_type}) per Frame")
        self.ax_stats.set_xlabel("Frame")
        self.ax_stats.set_ylabel("Error (m)")
        self.fig_stats.subplots_adjust(right=0.75)
        self.current_frame_line = self.ax_stats.axvline(x=0, color='red', lw=2, linestyle='--')

        # Create a separate line (and error list) for each keypoint in keypoint_for_stats
        self.stats_lines = {}
        self.stats_errors = {}  # dict mapping keypoint -> list of errors
        for stat_kp in self.keypoint_for_stats:
            line, = self.ax_stats.plot([], [], label=stat_kp.name, lw=1)
            self.stats_lines[stat_kp] = line
            self.stats_errors[stat_kp] = []

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
        self.pc_data = []  # New storage for point cloud data
        self.report = []
        self.num_frames = 0

    def _setup_axes(self, ax, title):
        ax.set_title(title)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    def _update_skeleton(self, tensor, scatter, lines):
        """
        Update the persistent scatter and line objects using the new tensor.
        The tensor is assumed to be a flat list or tensor of coordinates:
        [x0, z0, y0, x1, z1, y1, ...] and we map it to (x, y, z) with y coming from index+2.
        """
        coords = []
        for idx, kp_enum in enumerate(self.keypoint_involved):
            x = tensor[idx*3].item() if isinstance(tensor, torch.Tensor) else tensor[idx*3]
            y = tensor[idx*3+1].item() if isinstance(tensor, torch.Tensor) else tensor[idx*3+1]
            z = tensor[idx*3+2].item() if isinstance(tensor, torch.Tensor) else tensor[idx*3+2]
            coords.append((x, y, z))
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        zs = [pt[2] for pt in coords]
        scatter._offsets3d = (xs, ys, zs)
        # Update connectivity lines
        for (pair, line) in lines:
            idx1 = self.keypoint_involved.index(pair[0])
            idx2 = self.keypoint_involved.index(pair[1])
            source = coords[idx1]
            target = coords[idx2]
            line.set_data([source[0], target[0]], [source[1], target[1]])
            line.set_3d_properties([source[2], target[2]])

    def update_point_cloud(self, pc_tensor):
        """
        Update the point cloud plot with the provided tensor.
        pc_tensor: torch tensor of shape [1, F, N, C]
                   where F is the number of frames, N is the number of points,
                   and C is the point attributes (first three are X, Y, Z).
        If F > 1, a gradient color from blue to green is used to represent different frames.
        """
        # Clear the axis and reset settings
        self.ax_pc.cla()
        self._setup_axes(self.ax_pc, "Point Cloud")
        
        # Determine the number of frames and points
        _, F, N, C = pc_tensor.shape
        
        if F > 1:
            cmap = plt.get_cmap("winter_r")  # reversed colormap: transitions from green to blue
            for f in range(F):
                frame_points = pc_tensor[0, f]  # shape [N, C]
                # Extract X, Y, Z coordinates
                x = frame_points[:, 0].cpu().numpy() if isinstance(frame_points, torch.Tensor) else frame_points[:, 0]
                y = frame_points[:, 1].cpu().numpy() if isinstance(frame_points, torch.Tensor) else frame_points[:, 1]
                z = frame_points[:, 2].cpu().numpy() if isinstance(frame_points, torch.Tensor) else frame_points[:, 2]
                # Calculate a gradient color based on the frame index
                fraction = f / (F - 1)
                color = cmap(fraction)
                self.ax_pc.scatter(x, y, z, color=color)
        else:
            # Single frame: use a default color
            frame_points = pc_tensor[0, 0]  # shape [N, C]
            x = frame_points[:, 0].cpu().numpy() if isinstance(frame_points, torch.Tensor) else frame_points[:, 0]
            y = frame_points[:, 1].cpu().numpy() if isinstance(frame_points, torch.Tensor) else frame_points[:, 1]
            z = frame_points[:, 2].cpu().numpy() if isinstance(frame_points, torch.Tensor) else frame_points[:, 2]
            self.ax_pc.scatter(x, y, z, color="blue")
            
        self.canvas_pc.draw_idle()

    def update(self, gt_tensor, pred_tensor, pc_tensor, frame_report=None):
        """
        Update the skeleton and point cloud plots.
        
        Parameters:
            gt_tensor: Ground truth skeleton tensor.
            pred_tensor: Prediction skeleton tensor.
            pc_tensor: Point cloud tensor of shape [1, F, N, C].
            frame_report: (Optional) Dictionary with frame error reports.
        """
        # Update the skeleton plots
        self._update_skeleton(gt_tensor, self.scatter_gt, self.lines_gt)
        self._update_skeleton(pred_tensor, self.scatter_pred, self.lines_pred)
        # Update the point cloud plot
        self.update_point_cloud(pc_tensor)
        self.canvas_gt.draw_idle()
        self.canvas_pred.draw_idle()

        # Store data for replay if needed
        if frame_report is not None:
            self.report.append(frame_report)
        self.gt_data.append(gt_tensor)
        self.pred_data.append(pred_tensor)
        self.pc_data.append(pc_tensor)

        # Update stats errors for each keypoint in keypoint_for_stats
        if frame_report is not None:
            for stat_kp in self.keypoint_for_stats:
                error = frame_report.get(stat_kp, {}).get(self.error_type, None)
                if error is None:
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

        self.num_frames = len(next(iter(self.stats_errors.values()))) if self.stats_errors else 0

        for stat_kp in self.keypoint_for_stats:
            xdata = list(range(len(self.stats_errors[stat_kp])))
            self.stats_lines[stat_kp].set_data(xdata, self.stats_errors[stat_kp])

        self.ax_stats.relim()
        self.ax_stats.autoscale_view()
        self.progress_bar.config(to=self.num_frames - 1)
        self.progress_bar.set(self.num_frames - 1)
        self.canvas_stats.draw_idle()

    def _on_progress_change(self, value):
        frame_idx = int(float(value))
        self._update_replay(frame_idx)
        self.current_frame_line.set_xdata([frame_idx, frame_idx])
        self.canvas_stats.draw_idle()

    def _update_replay(self, frame_idx):
        # Update skeleton and point cloud for the selected replay frame
        if (frame_idx < len(self.gt_data) and 
            frame_idx < len(self.pred_data) and 
            frame_idx < len(self.pc_data)):
            self._update_skeleton(self.gt_data[frame_idx], self.scatter_gt, self.lines_gt)
            self._update_skeleton(self.pred_data[frame_idx], self.scatter_pred, self.lines_pred)
            self.update_point_cloud(self.pc_data[frame_idx])
            self.canvas_gt.draw_idle()
            self.canvas_pred.draw_idle()

    def setup_replay(self, gt_data, pred_data, report, pc_data):
        self.gt_data = gt_data
        self.pred_data = pred_data
        self.pc_data = pc_data
        self.report = report
    
    def finalize(self, gt_data, pred_data, report, pc_data):
        self.setup_replay(gt_data, pred_data, report, pc_data)
        self.root.mainloop()
