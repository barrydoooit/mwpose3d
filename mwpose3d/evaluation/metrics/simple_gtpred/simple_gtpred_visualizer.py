from __future__ import annotations
import math
import torch
from mwpose3d.utils.kinect_toolkits.kinectData import KeypointType, Connectivity

import math
from typing import Dict, List, Optional

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from mwpose3d.utils.kinect_toolkits.kinectData import KeypointType, Connectivity
import numpy as np
import torch


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
        dataset_connectivity: dict mapping parent joint idx to list of child joint idxs
    """
    def __init__(
        self,
        keypoints_involved: List,
        keypoint_for_stats: List,
        error_type: str = "abs_error",
        window_size: int = 100,
        follow: bool = True,
        max_points_per_frame: int = 50_000,
        show_point_cloud: bool = True,
        dataset_connectivity: Optional[Dict[int, List[int]]] = None,
    ):
        # Normalize to KeypointType
        self.keypoints_involved = [KeypointType(kp) for kp in keypoints_involved]
        self.keypoint_for_stats = [KeypointType(kp) for kp in keypoint_for_stats]
        self.error_type = error_type
        self.window_size = max(10, int(window_size))
        self.follow = bool(follow)
        self.max_points_per_frame = max_points_per_frame
        self.show_point_cloud = show_point_cloud

        self.app = _ensure_qapp()

        # Smoother lines in pyqtgraph
        try:
            pg.setConfigOptions(antialias=True)
        except Exception:
            pass

        # ---------- Data buffers (persist across window rebuilds) ----------
        self.gt_data: List = []
        self.pred_data: List = []
        self.pc_data: List = []
        self.report: List[Dict] = []
        self.centroids: List[Optional[np.ndarray]] = []
        self._bbox_half = 0.05
        self._bbox_zmin = -1.0
        self._bbox_zmax = 1.0
        self.stats_errors: Dict[KeypointType, List[float]] = {kp: [] for kp in self.keypoint_for_stats}
        self.num_frames: int = 0

        # bone connectivity pairs
        self._bone_pairs: List[tuple[int, int]] = []
        
        if dataset_connectivity is not None:
            # Use the provided connectivity dict (keys and values are raw integers)
            # We assume the user passed indices that match the order of `keypoints_involved`
            # or joint IDs directly. We map them internally.
            for parent_idx, children in dataset_connectivity.items():
                if parent_idx in keypoints_involved:
                    i1 = keypoints_involved.index(parent_idx)
                    for child_idx in children:
                        if child_idx in keypoints_involved:
                            i2 = keypoints_involved.index(child_idx)
                            self._bone_pairs.append((i1, i2))
        else:
            # Fallback to the default Kinect connectivity logic using Enums
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
        self._drag_multiplier = 2.0
        self._drag_origin_value: Optional[int] = None
        self._drag_dx_accum: float = 0.0

        self._build_window()

    def _build_window(self):
        """Create the Qt window once. Uses hide-on-close so GL contexts are never destroyed."""
        outer = self

        class _HideOnClose(QtWidgets.QMainWindow):
            """Close button hides the window rather than destroying it."""
            closed = QtCore.Signal()

            def closeEvent(self, event):
                event.ignore()
                self.hide()
                self.closed.emit()

        # ---------- Main window & layout ----------
        self.win = _HideOnClose()
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
        self.gt_bbox_item = gl.GLLinePlotItem(pos=np.zeros((0, 3), dtype=np.float32), color=(0.0, 1.0, 0.0, 1.0), width=2.5, mode='lines')
        self.view_gt.view.addItem(self.gt_bbox_item)
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
        self.pc_bbox_item = gl.GLLinePlotItem(pos=np.zeros((0, 3), dtype=np.float32), color=(0.0, 1.0, 0.0, 1.0), width=2.5, mode='lines')
        self.view_pc.addItem(self.pc_bbox_item)
        splitter.addWidget(self.view_pc)

        # Pred
        self.view_pred = _SkeletonView("Prediction")
        self.pred_bbox_item = gl.GLLinePlotItem(pos=np.zeros((0, 3), dtype=np.float32), color=(0.0, 1.0, 0.0, 1.0), width=2.5, mode='lines')
        self.view_pred.view.addItem(self.pred_bbox_item)
        splitter.addWidget(self.view_pred.view)

        # Bottom area: controls + error plot + full-width slider
        controls = QtWidgets.QHBoxLayout()
        root_layout.addLayout(controls)

        self.chk_follow = QtWidgets.QCheckBox("Follow latest")
        self.chk_follow.setChecked(self.follow)
        self.chk_follow.toggled.connect(self._on_follow_toggled)
        controls.addWidget(self.chk_follow)

        self.chk_show_pc = QtWidgets.QCheckBox("Show Point Cloud")
        self.chk_show_pc.setChecked(self.show_point_cloud)
        self.chk_show_pc.toggled.connect(self._on_toggle_pointcloud)
        controls.addWidget(self.chk_show_pc)

        controls.addStretch(1)
        lbl = QtWidgets.QLabel("Frame:")
        controls.addWidget(lbl)

        # --- Error plot with custom drag-to-scrub viewbox ---
        class _DragViewBox(pg.ViewBox):
            def __init__(vbself, on_drag_cb):
                super().__init__()
                vbself._on_drag_cb = on_drag_cb

            def mouseDragEvent(vbself, ev, axis=None):
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

        self._err_vb = _DragViewBox(self._on_errplot_drag)
        self.err_plot = pg.PlotWidget(viewBox=self._err_vb, enableMenu=False)
        self.err_plot.setBackground('w')
        self.err_plot.showGrid(x=True, y=True, alpha=0.25)
        self.err_plot.setLabel('left', 'Error')
        self.err_plot.setLabel('bottom', 'Frame')
        root_layout.addWidget(self.err_plot, 3)

        # Error lines (one per KP)
        self.err_curves: Dict[KeypointType, pg.PlotDataItem] = {}
        n_kps = max(1, len(self.keypoint_for_stats))
        for i, kp in enumerate(self.keypoint_for_stats):
            color = pg.intColor(i, hues=max(8, n_kps))
            pen = pg.mkPen(color=color, width=3)
            curve = self.err_plot.plot(pen=pen, name=str(kp.name))
            self.err_curves[kp] = curve

        self.err_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=2))
        self.err_plot.addItem(self.err_vline)
        self.err_plot.addLegend(offset=(10, 10))

        # --- Full-width bottom slider ---
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.slider.setMinimumHeight(26)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        root_layout.addWidget(self.slider)

        # Initial PC visibility
        self.view_pc.setVisible(self.show_point_cloud)

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
        self.centroids.clear()
        self.num_frames = 0
        # Re-show the window for the next validation epoch (window is hidden, not destroyed)
        self.win.setWindowTitle("3D Skeleton Visualizer (Qt)")
        self.slider.setRange(0, 0)
        self._draw_frame(None)  # clears display
        self._update_error_plot()
        self.win.show()
        self.win.raise_()

    def update(self, gt_tensor, pred_tensor, pc_tensor=None, frame_report=None, track_centroid=None):
        # Always accumulate data (needed for the metrics report even if GUI is gone)
        self.gt_data.append(gt_tensor)
        self.pred_data.append(pred_tensor)
        self.pc_data.append(pc_tensor)
        if frame_report is not None:
            self.report.append(frame_report)
        self.centroids.append(None if track_centroid is None else np.asarray(track_centroid, dtype=np.float32))
        
        self._append_errors(gt_tensor, pred_tensor, frame_report)
        self.num_frames = len(next(iter(self.stats_errors.values()))) if self.stats_errors else len(self.gt_data)

        # Guard: skip all Qt widget calls if the window was closed
        if self.win is None:
            return

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
        Keep the visualizer responsive until the user closes (hides) the window.

        The window uses hide-on-close, so clicking X just hides it and emits
        `closed`. This spins a nested Qt event loop that exits on that signal.
        """
        try:
            # Ensure the window is visible
            if not self.win.isVisible():
                self.win.show()
                self.win.raise_()

            # Nested event loop that exits when the user clicks X
            loop = QtCore.QEventLoop()
            self.win.closed.connect(loop.quit)
            loop.exec()
            self.win.closed.disconnect(loop.quit)

        except Exception:
            # Fallback: lightweight polling loop that keeps the UI alive
            import time
            while self.win.isVisible():
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
            self.gt_bbox_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self.pred_bbox_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self.pc_bbox_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
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

        # Build bone segments as [2,3] arrays only if we have bone pairs
        if self._bone_pairs:
            gt_segments = [np.vstack([gt_xyz[i1], gt_xyz[i2]]) for (i1, i2) in self._bone_pairs]
            pr_segments = [np.vstack([pred_xyz[i1], pred_xyz[i2]]) for (i1, i2) in self._bone_pairs]
            self.view_gt.set_bones(gt_segments, color=self._color_gt, width=2.0)
            self.view_pred.set_bones(pr_segments, color=self._color_pred, width=2.0)
        else:
            self.view_gt.set_bones([])
            self.view_pred.set_bones([])

        self.view_gt.set_bones(gt_segments, color=self._color_gt, width=2.0)
        self.view_pred.set_bones(pr_segments, color=self._color_pred, width=2.0)

        # draw green tracking bbox (same in GT, Pred, and PC views)
        center = self.centroids[idx] if idx < len(self.centroids) else None
        if center is not None:
            cx, cy = float(center[0]), float(center[1])
            hs, zmin, zmax = self._bbox_half, self._bbox_zmin, self._bbox_zmax
            b = np.array([[cx-hs, cy-hs, zmin],
                          [cx+hs, cy-hs, zmin],
                          [cx+hs, cy+hs, zmin],
                          [cx-hs, cy+hs, zmin]], dtype=np.float32)
            t = b.copy(); t[:, 2] = zmax
            segs = []
            # 12 edges as pairs for GL_LINES (2*12 = 24 points)
            for i in range(4): segs += [b[i], t[i]]                 # verticals
            for i in range(4): segs += [b[i], b[(i+1) % 4]]         # bottom loop
            for i in range(4): segs += [t[i], t[(i+1) % 4]]         # top loop
            pts = np.vstack(segs).astype(np.float32)
            self.gt_bbox_item.setData(pos=pts, color=(0.0, 1.0, 0.0, 1.0), width=2.5, mode='lines')
            self.pred_bbox_item.setData(pos=pts, color=(0.0, 1.0, 0.0, 1.0), width=2.5, mode='lines')
        else:
            self.gt_bbox_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self.pred_bbox_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
        # --- Point cloud (optional) ---
        if self.chk_show_pc.isChecked():
            pc = self.pc_data[idx]
            pos = self._extract_pc_positions(pc)  # np.ndarray [N,3] or None
            if pos is None or pos.size == 0:
                self.pc_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            else:
                self.pc_item.setData(pos=pos.astype(np.float32), size=1.5)
            if center is not None:
                self.pc_bbox_item.setData(pos=pts, color=(0.0, 1.0, 0.0, 1.0), width=2.5, mode='lines')
            else:
                self.pc_bbox_item.setData(pos=np.zeros((0, 3), dtype=np.float32))

        else:
            self.pc_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self.pc_bbox_item.setData(pos=np.zeros((0, 3), dtype=np.float32)) 

    def _extract_pc_positions(self, pc_tensor) -> Optional[np.ndarray]:
        """Accept numpy array or torch.Tensor of shapes [1,F,N,C], [F,N,C], or [N,C];
        shows the most recent frame's XYZ positions."""
        if pc_tensor is None:
            return None

        # Normalize to numpy
        if isinstance(pc_tensor, torch.Tensor):
            arr = pc_tensor.detach().cpu().float().numpy()
        elif isinstance(pc_tensor, np.ndarray):
            arr = pc_tensor.astype(np.float32)
        else:
            return None

        def decimate(xyz: np.ndarray) -> np.ndarray:
            N = xyz.shape[0]
            if self.max_points_per_frame and N > self.max_points_per_frame:
                idx = np.random.choice(N, self.max_points_per_frame, replace=False)
                return xyz[idx]
            return xyz

        if arr.ndim == 4:
            # [B,F,N,C] -> take first B, last up to 2 frames and stack
            B, F, N, C = arr.shape
            f_take = min(2, F)
            frames = [arr[0, F - i - 1, :, :3] for i in range(f_take)]
            return np.concatenate([decimate(fr) for fr in frames], axis=0)
        elif arr.ndim == 3:
            # [F,N,C] -> last up to 2 frames
            F, N, C = arr.shape
            f_take = min(2, F)
            frames = [arr[F - i - 1, :, :3] for i in range(f_take)]
            return np.concatenate([decimate(fr) for fr in frames], axis=0)
        elif arr.ndim == 2:
            # [N,C]
            return decimate(arr[:, :3])
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
