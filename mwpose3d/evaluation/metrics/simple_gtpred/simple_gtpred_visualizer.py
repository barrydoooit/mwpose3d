from __future__ import annotations
import math
import torch
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt  # for colormap
import random

from mwpose3d.utils.kinect_toolkits.kinectData import KeypointType, Connectivity


class SimpleGTPredVisualizer:
    """
    Live Tk/Matplotlib visualizer for GT vs Pred and a sliding-window error plot.

    - Call update(...) every frame.
    - The window stays responsive because we call `idle()` (root.update_*) each frame.
    - Error plot shows a sliding window (last `window_size` frames). Set follow=False
      if you want to keep window static and only pan the progress bar to move focus.

    Args:
        keypoints_involved: list of keypoints to draw the skeleton.
        keypoint_for_stats: list of keypoints to compute/display the error traces.
        error_type: "abs_error" or "square_error".
        window_size: number of recent frames to show in the error plot.
        follow: if True, the xlim follows the latest frame (sliding window).
        max_points_per_frame: limit point count for speed (random decimation).
    """
    def __init__(
        self,
        keypoints_involved,
        keypoint_for_stats,
        error_type="abs_error",
        window_size=100,
        follow=True,
        max_points_per_frame=50000,
    ):
        # Normalize to KeypointType
        self.keypoints_involved = [KeypointType(kp) for kp in keypoints_involved]
        self.keypoint_for_stats = [KeypointType(kp) for kp in keypoint_for_stats]
        self.error_type = error_type
        self.window_size = max(10, int(window_size))
        self.follow = bool(follow)
        self.max_points_per_frame = max_points_per_frame

        # --- Tk layout ---
        self.root = tk.Tk()
        self.root.title("3D Skeleton Visualization")

        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        frame_gt = tk.Frame(top_frame)
        frame_pc = tk.Frame(top_frame)
        frame_pred = tk.Frame(top_frame)
        frame_gt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frame_pc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frame_pred.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- GT axis ---
        self.fig_gt = Figure(figsize=(5, 5))
        self.ax_gt = self.fig_gt.add_subplot(111, projection='3d')
        self._setup_axes(self.ax_gt, "Ground Truth")
        self.canvas_gt = FigureCanvasTkAgg(self.fig_gt, master=frame_gt)
        self.canvas_gt.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- PC axis ---
        self.fig_pc = Figure(figsize=(5, 5))
        self.ax_pc = self.fig_pc.add_subplot(111, projection='3d')
        self._setup_axes(self.ax_pc, "Point Cloud")
        self.canvas_pc = FigureCanvasTkAgg(self.fig_pc, master=frame_pc)
        self.canvas_pc.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Pred axis ---
        self.fig_pred = Figure(figsize=(5, 5))
        self.ax_pred = self.fig_pred.add_subplot(111, projection='3d')
        self._setup_axes(self.ax_pred, "Prediction")
        self.canvas_pred = FigureCanvasTkAgg(self.fig_pred, master=frame_pred)
        self.canvas_pred.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Build connectivity
        self.connectivity_pairs = []
        for kp in self.keypoints_involved:
            if kp in Connectivity:
                for connected in Connectivity[kp]:
                    if connected in self.keypoints_involved:
                        self.connectivity_pairs.append((kp, connected))

        # Persistent artists
        self.scatter_gt = self.ax_gt.scatter([], [], [], color='blue')
        self.lines_gt = []
        for pair in self.connectivity_pairs:
            line, = self.ax_gt.plot([], [], [], color='blue', lw=1)
            self.lines_gt.append((pair, line))

        self.scatter_pred = self.ax_pred.scatter([], [], [], color='red')
        self.lines_pred = []
        for pair in self.connectivity_pairs:
            line, = self.ax_pred.plot([], [], [], color='red', lw=1)
            self.lines_pred.append((pair, line))

        # --- Stats area ---
        self.frame_stats = tk.Frame(self.root)
        self.frame_stats.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.fig_stats = Figure(figsize=(10, 3))
        self.ax_stats = self.fig_stats.add_subplot(111)
        self.ax_stats.set_title(f"Error ({self.error_type})")
        self.ax_stats.set_xlabel("Frame")
        self.ax_stats.set_ylabel("Error")
        self.fig_stats.subplots_adjust(right=0.75)
        self.current_frame_line = self.ax_stats.axvline(x=0, color='red', lw=2, linestyle='--')

        # Lines per KP
        self.stats_lines = {}
        self.stats_errors = {kp: [] for kp in self.keypoint_for_stats}
        for stat_kp in self.keypoint_for_stats:
            line, = self.ax_stats.plot([], [], label=stat_kp.name, lw=1)
            self.stats_lines[stat_kp] = line

        leg = self.ax_stats.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        for legline in leg.get_lines():
            legline.set_linewidth(4)

        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, self.frame_stats)
        self.canvas_stats.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Progress slider
        self.progress_bar = tk.Scale(
            self.frame_stats,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self._on_progress_change,
            label="Frame"
        )
        self.progress_bar.pack(fill=tk.X)

        # Streaming buffers
        self.gt_data = []
        self.pred_data = []
        self.pc_data = []
        self.report = []
        self.num_frames = 0

        # State
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------ Public control ------------

    def idle(self):
        """Pump the Tk event loop without blocking the training loop."""
        if self._closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._closed = True

    def reset(self):
        self.gt_data.clear()
        self.pred_data.clear()
        self.pc_data.clear()
        self.report.clear()
        for kp in self.stats_errors:
            self.stats_errors[kp].clear()
        self.num_frames = 0
        if not self._closed:
            self._clear_axes()
            self.idle()

    # ------------ Drawing helpers ------------

    def _on_close(self):
        self._closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def _clear_axes(self):
        # GT
        self._setup_axes(self.ax_gt, "Ground Truth")
        for _, ln in self.lines_gt:
            ln.set_data([], [])
            ln.set_3d_properties([])
        self.scatter_gt._offsets3d = ([], [], [])
        self.canvas_gt.draw_idle()
        # Pred
        self._setup_axes(self.ax_pred, "Prediction")
        for _, ln in self.lines_pred:
            ln.set_data([], [])
            ln.set_3d_properties([])
        self.scatter_pred._offsets3d = ([], [], [])
        self.canvas_pred.draw_idle()
        # PC
        self.ax_pc.cla()
        self._setup_axes(self.ax_pc, "Point Cloud")
        self.canvas_pc.draw_idle()
        # Stats
        self.ax_stats.cla()
        self.ax_stats.set_title(f"Error ({self.error_type})")
        self.ax_stats.set_xlabel("Frame")
        self.ax_stats.set_ylabel("Error")
        self.current_frame_line = self.ax_stats.axvline(x=0, color='red', lw=2, linestyle='--')
        self.stats_lines.clear()
        for kp in self.keypoint_for_stats:
            line, = self.ax_stats.plot([], [], label=kp.name, lw=1)
            self.stats_lines[kp] = line
        leg = self.ax_stats.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        for legline in leg.get_lines():
            legline.set_linewidth(4)
        self.canvas_stats.draw_idle()

    def _setup_axes(self, ax, title):
        ax.set_title(title)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    def _update_skeleton(self, tensor, scatter, lines):
        coords = []
        is_t = isinstance(tensor, torch.Tensor)
        for idx, _ in enumerate(self.keypoints_involved):
            x = tensor[idx*3].item() if is_t else tensor[idx*3]
            y = tensor[idx*3+1].item() if is_t else tensor[idx*3+1]
            z = tensor[idx*3+2].item() if is_t else tensor[idx*3+2]
            coords.append((x, y, z))
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        zs = [p[2] for p in coords]
        scatter._offsets3d = (xs, ys, zs)
        # bones
        for (pair, line) in lines:
            idx1 = self.keypoints_involved.index(pair[0])
            idx2 = self.keypoints_involved.index(pair[1])
            s = coords[idx1]
            t = coords[idx2]
            line.set_data([s[0], t[0]], [s[1], t[1]])
            line.set_3d_properties([s[2], t[2]])

    def _decimate_points(self, xyz):
        """xyz: Tensor [N,3] on CPU; randomly downsample to max_points_per_frame."""
        if xyz is None:
            return None
        N = xyz.shape[0]
        if self.max_points_per_frame is None or N <= self.max_points_per_frame:
            return xyz
        idx = torch.randperm(N)[: self.max_points_per_frame]
        return xyz[idx]

    def _draw_point_cloud(self, pc_tensor):
        # Reset axis each frame to avoid overplot
        self.ax_pc.cla()
        self._setup_axes(self.ax_pc, "Point Cloud")
        if pc_tensor is None:
            self.canvas_pc.draw_idle()
            return

        # Accept [1,F,N,C], [F,N,C], [N,C]
        is_t = isinstance(pc_tensor, torch.Tensor)
        if not is_t:
            self.canvas_pc.draw_idle()
            return

        pts = []
        if pc_tensor.dim() == 4:          # [B,F,N,C]
            B, F, N, C = pc_tensor.shape
            F = min(F, 8)  # soft limit to keep drawing fast
            # draw all frames with gradient from green->blue
            cmap = plt.get_cmap("winter_r")
            for f in range(F):
                frame = pc_tensor[0, f]
                xyz = frame[:, :3].to("cpu")
                xyz = self._decimate_points(xyz)
                color = cmap(f / max(1, F - 1))
                self.ax_pc.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, s=1)
        elif pc_tensor.dim() == 3:        # [F,N,C]
            F, N, C = pc_tensor.shape
            F = min(F, 8)
            cmap = plt.get_cmap("winter_r")
            for f in range(F):
                frame = pc_tensor[f]
                xyz = frame[:, :3].to("cpu")
                xyz = self._decimate_points(xyz)
                color = cmap(f / max(1, F - 1))
                self.ax_pc.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, s=1)
        elif pc_tensor.dim() == 2:        # [N,C]
            xyz = pc_tensor[:, :3].to("cpu")
            xyz = self._decimate_points(xyz)
            self.ax_pc.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="blue", s=1)

        self.canvas_pc.draw_idle()

    # ------------ Streaming update ------------

    def update(self, gt_tensor, pred_tensor, pc_tensor=None, frame_report=None):
        if self._closed:
            return

        # Update skeletons
        self._update_skeleton(gt_tensor, self.scatter_gt, self.lines_gt)
        self._update_skeleton(pred_tensor, self.scatter_pred, self.lines_pred)
        self.canvas_gt.draw_idle()
        self.canvas_pred.draw_idle()

        # Update point cloud
        self._draw_point_cloud(pc_tensor)

        # Store (only small refs to keep memory under control)
        self.gt_data.append(gt_tensor)
        self.pred_data.append(pred_tensor)
        self.pc_data.append(pc_tensor)
        if frame_report is not None:
            self.report.append(frame_report)

        # Update stats buffers
        if frame_report is not None:
            for stat_kp in self.keypoint_for_stats:
                # Prefer precomputed error if present
                v = frame_report.get(stat_kp, None)
                if v is not None and self.error_type in v:
                    self.stats_errors[stat_kp].append(v[self.error_type])
                    continue
                # Otherwise compute
                if stat_kp in self.keypoints_involved:
                    idx = self.keypoints_involved.index(stat_kp)
                    gt_joint = gt_tensor[idx*3: idx*3+3]
                    pred_joint = pred_tensor[idx*3: idx*3+3]
                    diff = gt_joint - pred_joint
                    if self.error_type == "abs_error":
                        err = torch.norm(diff, p=2).item()
                    elif self.error_type == "square_error":
                        err = torch.sum(diff * diff).item()
                    else:
                        err = torch.norm(diff, p=2).item()
                else:
                    err = 0.0
                self.stats_errors[stat_kp].append(err)
        else:
            # Compute on the fly
            for stat_kp in self.keypoint_for_stats:
                if stat_kp in self.keypoints_involved:
                    idx = self.keypoints_involved.index(stat_kp)
                    gt_joint = gt_tensor[idx*3: idx*3+3]
                    pred_joint = pred_tensor[idx*3: idx*3+3]
                    diff = gt_joint - pred_joint
                    if self.error_type == "abs_error":
                        err = torch.norm(diff, p=2).item()
                    elif self.error_type == "square_error":
                        err = torch.sum(diff * diff).item()
                    else:
                        err = torch.norm(diff, p=2).item()
                else:
                    err = 0.0
                self.stats_errors[stat_kp].append(err)

        # Frame index & progress bar
        self.num_frames = len(next(iter(self.stats_errors.values()))) if self.stats_errors else 0
        self.progress_bar.config(to=max(0, self.num_frames - 1))
        self.progress_bar.set(max(0, self.num_frames - 1))

        # Redraw stats (sliding window)
        right = max(0, self.num_frames - 1)
        left = max(0, right - self.window_size + 1)
        for kp in self.keypoint_for_stats:
            y = self.stats_errors[kp]
            x = list(range(len(y)))
            # Subslice to window to reduce draw cost
            xs = x[left:right+1]
            ys = y[left:right+1]
            self.stats_lines[kp].set_data(xs, ys)

        # Autoscale to data in-window
        self.ax_stats.relim()
        self.ax_stats.autoscale_view()
        if self.follow:
            self.ax_stats.set_xlim(left, max(left + self.window_size - 1, right))
        self.current_frame_line.set_xdata([right, right])
        self.canvas_stats.draw_idle()

    # ------------ Replay controls ------------

    def _on_progress_change(self, value):
        if self._closed:
            return
        frame_idx = int(float(value))
        self._update_replay(frame_idx)
        self.current_frame_line.set_xdata([frame_idx, frame_idx])
        # Show a window around the chosen frame
        half = self.window_size // 2
        self.ax_stats.set_xlim(max(0, frame_idx - half), frame_idx + half)
        self.canvas_stats.draw_idle()

    def _update_replay(self, frame_idx):
        if (0 <= frame_idx < len(self.gt_data) and 
            frame_idx < len(self.pred_data) and 
            frame_idx < len(self.pc_data)):
            self._update_skeleton(self.gt_data[frame_idx], self.scatter_gt, self.lines_gt)
            self._update_skeleton(self.pred_data[frame_idx], self.scatter_pred, self.lines_pred)
            self._draw_point_cloud(self.pc_data[frame_idx])
            self.canvas_gt.draw_idle()
            self.canvas_pred.draw_idle()

    # (keep API parity with previous)
    def setup_replay(self, gt_data, pred_data, report, pc_data):
        self.gt_data = gt_data
        self.pred_data = pred_data
        self.pc_data = pc_data
        self.report = report

    def finalize(self, *args, **kwargs):
        """
        Kept for backward-compatibility. No blocking call here anymore.
        If you really want to block, you can call `while not vis._closed: vis.idle()`.
        """
        pass

# simple_gtpred_visualizer.py

import math
from typing import Dict, List, Optional

import numpy as np
import torch

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from mwpose3d.utils.kinect_toolkits.kinectData import KeypointType, Connectivity


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _to_numpy_1d(t: torch.Tensor) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float()
        return t.numpy()
    return np.asarray(t, dtype=np.float32)


class _SkeletonView:
    """A small wrapper around a GLViewWidget holding scatter + multiple line segments."""
    def __init__(self, title: str, parent=None):
        self.view = gl.GLViewWidget(parent=parent)
        self.view.opts['distance'] = 4.0
        self.view.setWindowTitle(title)

        # optional grid for orientation
        g = gl.GLGridItem()
        g.scale(0.5, 0.5, 0.5)
        g.setDepthValue(10)
        self.view.addItem(g)

        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3), dtype=np.float32), size=5.0)
        self.view.addItem(self.scatter)

        self.lines: List[gl.GLLinePlotItem] = []

    def clear_lines(self):
        for ln in self.lines:
            self.view.removeItem(ln)
        self.lines.clear()

    def set_scatter(self, pos: np.ndarray, color=(0.2, 0.2, 1.0, 1.0), size: float = 5.0):
        if pos is None or pos.size == 0:
            self.scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return
        self.scatter.setData(pos=pos.astype(np.float32), color=color, size=size)

    def set_bones(self, segments: List[np.ndarray], color=(0.2, 0.2, 1.0, 1.0), width: float = 2.0):
        # Recreate line items only if count differs (simpler and still fast)
        if len(self.lines) != len(segments):
            self.clear_lines()
            for _ in segments:
                ln = gl.GLLinePlotItem(pos=np.zeros((2, 3), dtype=np.float32), color=color, width=width, antialias=True, mode='lines')
                self.view.addItem(ln)
                self.lines.append(ln)
        for ln, seg in zip(self.lines, segments):
            ln.setData(pos=seg.astype(np.float32), color=color, width=width, mode='lines')


class SimpleGTPredVisualizerQT:
    """
    High-performance live visualizer using PySide6 + pyqtgraph.

    - Call update(gt, pred, pc, frame_report) every frame.
    - Call idle() once per processed sample (non-blocking UI pump).
    - Window is NOT closed on evaluate(); you can scrub history anytime.
    - Error plot shows a sliding window of `window_size` frames.

    Args (visualizer_cfg):
        keypoints_involved: list of KP identifiers (ints/enums/names)
        keypoint_for_stats: list of KPs to plot errors for
        error_type: "abs_error" or "square_error"
        window_size: int, width of sliding window in error plot
        follow: bool, keep viewport following newest frame
        max_points_per_frame: int, decimate point cloud for speed
        show_point_cloud: bool, initial visibility for point cloud panel
    """
    def __init__(
        self,
        keypoints_involved: List,
        keypoint_for_stats: List,
        error_type: str = "abs_error",
        window_size: int = 100,
        follow: bool = True,
        max_points_per_frame: int = 50_000,
        show_point_cloud: bool = False,
    ):
        # Normalize to KeypointType
        self.keypoints_involved = [KeypointType(kp) for kp in keypoints_involved]
        self.keypoint_for_stats = [KeypointType(kp) for kp in keypoint_for_stats]
        self.error_type = error_type
        self.window_size = max(10, int(window_size))
        self.follow = bool(follow)
        self.max_points_per_frame = max_points_per_frame

        self.app = _ensure_qapp()

        # Smoother lines in pyqtgraph
        try:
            pg.setConfigOptions(antialias=True)
        except Exception:
            pass

        # ---------- Main window & layout ----------
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("3D Skeleton Visualizer (Qt)")
        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        # Top row: 3D views in a splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root_layout.addWidget(splitter, 4)

        # GT
        self.view_gt = _SkeletonView("Ground Truth")
        splitter.addWidget(self.view_gt.view)

        # PCD
        self.view_pc = gl.GLViewWidget()
        self.view_pc.opts['distance'] = 4.0
        grid_pc = gl.GLGridItem()
        grid_pc.scale(0.5, 0.5, 0.5)
        grid_pc.setDepthValue(10)
        self.view_pc.addItem(grid_pc)
        self.pc_item = gl.GLScatterPlotItem(pos=np.zeros((0, 3), dtype=np.float32), size=1.5, color=(0.1, 0.6, 0.9, 0.9))
        self.view_pc.addItem(self.pc_item)
        splitter.addWidget(self.view_pc)

        # Pred
        self.view_pred = _SkeletonView("Prediction")
        splitter.addWidget(self.view_pred.view)

        # Bottom area: controls + error plot + full-width slider
        controls = QtWidgets.QHBoxLayout()
        root_layout.addLayout(controls)

        self.chk_follow = QtWidgets.QCheckBox("Follow latest")
        self.chk_follow.setChecked(self.follow)
        self.chk_follow.toggled.connect(self._on_follow_toggled)
        controls.addWidget(self.chk_follow)

        self.chk_show_pc = QtWidgets.QCheckBox("Show Point Cloud")
        self.chk_show_pc.setChecked(show_point_cloud)
        self.chk_show_pc.toggled.connect(self._on_toggle_pointcloud)
        controls.addWidget(self.chk_show_pc)

        controls.addStretch(1)
        # Keep a small label in controls (slider itself is now at the bottom, full width)
        lbl = QtWidgets.QLabel("Frame:")
        controls.addWidget(lbl)

        # --- Error plot with custom drag-to-scrub viewbox ---
        class _DragViewBox(pg.ViewBox):
            def __init__(vbself, on_drag_cb):
                super().__init__()
                vbself._on_drag_cb = on_drag_cb

            def mouseDragEvent(vbself, ev, axis=None):
                # Left-button drag drives scrubbing; others fall back to default behavior
                if ev.button() == QtCore.Qt.LeftButton:
                    ev.accept()
                    if ev.isStart():
                        vbself._on_drag_cb(phase="start", dx=0.0)
                    elif ev.isFinish():
                        vbself._on_drag_cb(phase="finish", dx=0.0)
                    else:
                        dx = ev.pos().x() - ev.lastPos().x()
                        vbself._on_drag_cb(phase="move", dx=float(dx))
                else:
                    super().mouseDragEvent(ev, axis=axis)

        self._drag_multiplier = 2.0  # tuneable factor for scrub speed
        self._drag_origin_value: Optional[int] = None
        self._drag_dx_accum: float = 0.0

        self._err_vb = _DragViewBox(self._on_errplot_drag)
        self.err_plot = pg.PlotWidget(viewBox=self._err_vb, enableMenu=False)
        self.err_plot.setBackground('w')
        self.err_plot.showGrid(x=True, y=True, alpha=0.25)
        self.err_plot.setLabel('left', 'Error')
        self.err_plot.setLabel('bottom', 'Frame')
        root_layout.addWidget(self.err_plot, 3)

        # Error lines (one per KP) — thicker pens and distinct colors
        self.err_curves: Dict[KeypointType, pg.PlotDataItem] = {}
        n_kps = max(1, len(self.keypoint_for_stats))
        for i, kp in enumerate(self.keypoint_for_stats):
            color = pg.intColor(i, hues=max(8, n_kps))  # distinct hues
            pen = pg.mkPen(color=color, width=3)        # thicker lines
            curve = self.err_plot.plot(pen=pen, name=str(kp.name))
            self.err_curves[kp] = curve

        self.err_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=2))
        self.err_plot.addItem(self.err_vline)
        # Add legend
        self.err_plot.addLegend(offset=(10, 10))

        # --- Full-width bottom slider (progress bar) ---
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.slider.setMinimumHeight(26)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        root_layout.addWidget(self.slider)

        # Initial PC visibility
        self.view_pc.setVisible(show_point_cloud)

        # ---------- Data buffers ----------
        self.gt_data: List[torch.Tensor] = []
        self.pred_data: List[torch.Tensor] = []
        self.pc_data: List[Optional[torch.Tensor]] = []
        self.report: List[Dict] = []
        self.stats_errors: Dict[KeypointType, List[float]] = {kp: [] for kp in self.keypoint_for_stats}
        self.num_frames: int = 0

        # connectivity segments index mapping (pairs of indices)
        self._bone_pairs: List[tuple[int, int]] = []
        for kp in self.keypoints_involved:
            if kp in Connectivity:
                for connected in Connectivity[kp]:
                    if connected in self.keypoints_involved:
                        i1 = self.keypoints_involved.index(kp)
                        i2 = self.keypoints_involved.index(connected)
                        self._bone_pairs.append((i1, i2))

        # Colors
        self._color_gt = (0.2, 0.2, 1.0, 1.0)
        self._color_pred = (1.0, 0.2, 0.2, 1.0)

        # Slider state
        self._user_dragging = False

        # Show window immediately
        self.win.resize(1400, 900)
        self.win.show()

    # --------- Public API (called from analyzer) ---------
    def idle(self):
        """Process Qt events without blocking the caller (keep UI responsive)."""
        self.app.processEvents()

    def reset(self):
        for v in self.stats_errors.values():
            v.clear()
        self.gt_data.clear()
        self.pred_data.clear()
        self.pc_data.clear()
        self.report.clear()
        self.num_frames = 0
        self.slider.setRange(0, 0)
        self._draw_frame(None)  # clears
        self._update_error_plot()

    def update(self, gt_tensor, pred_tensor, pc_tensor=None, frame_report=None):
        # Store references (CPU tensors or numpy-friendly)
        self.gt_data.append(gt_tensor)
        self.pred_data.append(pred_tensor)
        self.pc_data.append(pc_tensor)
        if frame_report is not None:
            self.report.append(frame_report)

        # Update stats
        self._append_errors(gt_tensor, pred_tensor, frame_report)
        self.num_frames = len(next(iter(self.stats_errors.values()))) if self.stats_errors else len(self.gt_data)

        # Update slider maximum; if following and not dragging, keep it at newest
        self.slider.setMaximum(max(0, self.num_frames - 1))
        if self.chk_follow.isChecked() and not self._user_dragging:
            self.slider.setValue(self.num_frames - 1)

        # Draw newest (or slider-selected if user is dragging)
        idx = self.slider.value()
        self._draw_frame(idx)
        self._update_error_plot()

    def finalize(self, *args, **kwargs):
        """
        Keep the visualizer responsive and open until the user closes the window.

        This spins a nested Qt event loop tied to the window's lifetime, so callers
        (e.g., evaluate()) return only after the user closes the visualizer.
        """
        try:
            from PySide6 import QtCore

            # If the window is already gone, nothing to do.
            if not hasattr(self, "win") or self.win is None or not callable(getattr(self.win, "isVisible", None)):
                return
            if not self.win.isVisible():
                return

            # Ensure the window emits `destroyed` when closed by the user
            self.win.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

            # Nested event loop that exits when the window is destroyed
            loop = QtCore.QEventLoop()
            self.win.destroyed.connect(loop.quit)

            # Run until user closes the window
            loop.exec()

        except Exception:
            # Fallback: lightweight polling loop that keeps the UI alive
            import time
            while (
                hasattr(self, "win")
                and self.win is not None
                and callable(getattr(self.win, "isVisible", None))
                and self.win.isVisible()
            ):
                # Non-blocking UI pump
                try:
                    self.idle()
                except Exception:
                    pass
                time.sleep(0.016)  # ~60 FPS

    # --------- Internals ---------
    def _on_follow_toggled(self, checked: bool):
        # If turning follow on, jump to most recent
        if checked:
            self.slider.setValue(max(0, self.num_frames - 1))

    def _on_toggle_pointcloud(self, checked: bool):
        self.view_pc.setVisible(checked)
        # No need to redraw here; next update or slider move will take care

    def _on_slider_pressed(self):
        self._user_dragging = True

    def _on_slider_released(self):
        self._user_dragging = False
        # When released, force a redraw at the selected frame
        idx = self.slider.value()
        self._draw_frame(idx)
        self._update_error_plot()

    def _on_slider_changed(self, value: int):
        # Continuous feedback while dragging
        if self._user_dragging:
            self._draw_frame(value)
            self._update_error_plot()

    # --- Error-plot drag-to-scrub handling ---
    def _on_errplot_drag(self, phase: str, dx: float):
        """
        Map horizontal drag on the error plot to smooth scrubbing of frames.
        Dragging to the right -> earlier frames (decrease slider).
        Dragging to the left  -> later frames  (increase slider).
        The speed scales with the number of frames currently visible.
        """
        if self.num_frames <= 1:
            return

        if phase == "start":
            self._user_dragging = True
            self._drag_origin_value = int(self.slider.value())
            self._drag_dx_accum = 0.0
            return

        if phase == "finish":
            self._user_dragging = False
            self._drag_origin_value = None
            self._drag_dx_accum = 0.0
            return

        # phase == "move"
        self._drag_dx_accum += dx

        # Visible frame count = width of current x-range in frames
        xmin, xmax = self.err_plot.viewRange()[0]
        frames_visible = max(1, int(round(xmax - xmin))) + 1

        # Widget width in pixels (avoid division by zero)
        w = max(1, int(self.err_plot.size().width()))

        # How many frames to move based on drag distance
        delta_frames_float = (self._drag_dx_accum / float(w)) * frames_visible * self._drag_multiplier
        delta_frames = int(round(delta_frames_float))

        origin = self._drag_origin_value if self._drag_origin_value is not None else int(self.slider.value())
        new_val = int(origin - delta_frames)  # right drag (dx>0) -> earlier frames
        new_val = max(0, min(max(0, self.num_frames - 1), new_val))

        # Update slider + visuals
        self.slider.setValue(new_val)
        self._draw_frame(new_val)
        self._update_error_plot()

    def _append_errors(self, gt: torch.Tensor, pred: torch.Tensor, frame_report: Optional[Dict]):
        # Prefer precomputed errors per KP from the frame report if provided
        for kp in self.keypoint_for_stats:
            val: Optional[float] = None
            if frame_report is not None:
                r = frame_report.get(kp, None)
                if r is not None and self.error_type in r:
                    val = float(r[self.error_type])
            if val is None:
                # Compute on the fly
                if kp in self.keypoints_involved:
                    idx = self.keypoints_involved.index(kp)
                    g = gt[idx * 3: idx * 3 + 3]
                    p = pred[idx * 3: idx * 3 + 3]
                    diff = g - p
                    if self.error_type == "square_error":
                        val = float(torch.sum(diff * diff).item())
                    else:
                        val = float(torch.norm(diff, p=2).item())
                else:
                    val = 0.0
            self.stats_errors[kp].append(val)

    # ---- Drawing ----
    def _draw_frame(self, idx: Optional[int]):
        if idx is None or self.num_frames == 0 or idx >= self.num_frames:
            # Clear everything
            self.view_gt.set_scatter(np.zeros((0, 3), dtype=np.float32))
            self.view_gt.set_bones([])
            self.view_pred.set_scatter(np.zeros((0, 3), dtype=np.float32))
            self.view_pred.set_bones([])
            self.pc_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        # --- Skeleton GT/Pred ---
        gt = _to_numpy_1d(self.gt_data[idx])
        pred = _to_numpy_1d(self.pred_data[idx])

        # reshape to [K,3]
        K = len(self.keypoints_involved)
        gt_xyz = gt.reshape(K, 3)
        pred_xyz = pred.reshape(K, 3)

        self.view_gt.set_scatter(gt_xyz, color=self._color_gt, size=6.0)
        self.view_pred.set_scatter(pred_xyz, color=self._color_pred, size=6.0)

        # Build bone segments as [2,3] arrays
        gt_segments = [np.vstack([gt_xyz[i1], gt_xyz[i2]]) for (i1, i2) in self._bone_pairs]
        pr_segments = [np.vstack([pred_xyz[i1], pred_xyz[i2]]) for (i1, i2) in self._bone_pairs]

        self.view_gt.set_bones(gt_segments, color=self._color_gt, width=2.0)
        self.view_pred.set_bones(pr_segments, color=self._color_pred, width=2.0)

        # --- Point cloud (optional) ---
        if self.chk_show_pc.isChecked():
            pc = self.pc_data[idx]
            pos = self._extract_pc_positions(pc)  # np.ndarray [N,3] or None
            if pos is None or pos.size == 0:
                self.pc_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            else:
                self.pc_item.setData(pos=pos.astype(np.float32), size=1.5)
        else:
            self.pc_item.setData(pos=np.zeros((0, 3), dtype=np.float32))

    def _extract_pc_positions(self, pc_tensor) -> Optional[np.ndarray]:
        """Accept [1,F,N,C], [F,N,C], [N,C]; show up to 2 temporal slices with a gradient."""
        if pc_tensor is None:
            return None
        if not isinstance(pc_tensor, torch.Tensor):
            return None
        t = pc_tensor.detach().cpu().float()

        def decimate(xyz: torch.Tensor) -> torch.Tensor:
            N = xyz.shape[0]
            if self.max_points_per_frame and N > self.max_points_per_frame:
                idx = torch.randperm(N)[: self.max_points_per_frame]
                return xyz[idx]
            return xyz

        if t.dim() == 4:
            # [B,F,N,C] -> take first B, last up to 2 frames and stack
            B, F, N, C = t.shape
            f_take = min(2, F)
            frames = [t[0, F - i - 1, :, :3] for i in range(f_take)]
            xyz = torch.cat([decimate(fr) for fr in frames], dim=0)
            return xyz.numpy()
        elif t.dim() == 3:
            # [F,N,C] -> last up to 2 frames
            F, N, C = t.shape
            f_take = min(2, F)
            frames = [t[F - i - 1, :, :3] for i in range(f_take)]
            xyz = torch.cat([decimate(fr) for fr in frames], dim=0)
            return xyz.numpy()
        elif t.dim() == 2:
            # [N,C]
            xyz = decimate(t[:, :3])
            return xyz.numpy()
        return None

    def _update_error_plot(self):
        if self.num_frames == 0:
            # clear
            for curve in self.err_curves.values():
                curve.setData([], [])
            self.err_vline.setPos(0)
            return

        right = max(0, self.slider.value())  # current focus
        if self.chk_follow.isChecked() and not self._user_dragging:
            right = self.num_frames - 1
        left = max(0, right - self.window_size + 1)

        for kp, curve in self.err_curves.items():
            ys = self.stats_errors[kp]
            xs = np.arange(len(ys), dtype=np.int32)
            xs_w = xs[left:right + 1]
            ys_w = np.asarray(ys[left:right + 1], dtype=np.float32)
            curve.setData(xs_w, ys_w)

        # set ranges
        self.err_plot.setXRange(left, max(left + self.window_size - 1, right), padding=0.02)
        # Y autoscale to window
        ymins, ymaxs = [], []
        for kp in self.keypoint_for_stats:
            ys = self.stats_errors[kp][left:right + 1]
            if ys:
                ymins.append(min(ys))
                ymaxs.append(max(ys))
        if ymins and ymaxs:
            ymin, ymax = float(min(ymins)), float(max(ymaxs))
            if math.isfinite(ymin) and math.isfinite(ymax):
                if ymax <= ymin:
                    ymax = ymin + 1e-6
                self.err_plot.setYRange(ymin, ymax, padding=0.1)

        # vertical line at focus
        self.err_vline.setPos(right)
