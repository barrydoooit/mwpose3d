import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, List
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mwpose3d.utils.kinect_toolkits.kinectVis import SkeletonFigure

from mwpose3d.utils.pointcloud_toolkits.pcdVis import PointCloudFigureFrame
from mwpose3d.utils.kinect_toolkits.kinectVis import SkeletonFigureFrame

class CombinedFigureFrame(tk.Frame):
    def __init__(self, master=None, figure_cfg: dict={}, **kwargs):
        super().__init__(master, **kwargs)
        figure_cfg.setdefault('figsize', (5, 5))
        figure_cfg.setdefault('dpi', 100)
        # We reuse SkeletonFigure because it handles the complicated skeleton connections 
        # and we can just add a separate scatter plot for point clouds.
        self.figure = SkeletonFigure(**figure_cfg)
        self.pcd_scatter = self.figure.ax.scatter([], [], [], c='g', marker='x', alpha=0.5)
        
        self.canvas = FigureCanvasTkAgg(self.figure.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update(self, skeleton, points):
        self.figure.update_skeleton(skeleton)
        
        # Update point cloud scatter
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        zs = [point.z for point in points]
        self.pcd_scatter._offsets3d = (xs, ys, zs)
        
        self.canvas.draw()
    
    def reset(self):
        self.figure.ax.clear()
        self.canvas.draw()

if TYPE_CHECKING:
    from mwpose3d.utils.pointcloud_toolkits.structures import SimplePoint5D
    from tools.rawproc.episode import Episode


class CalibrateTimeWindow(tk.Toplevel):
    def __init__(self, episode: "Episode", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode = episode
        self.geometry(kwargs.get('geometry', '1500x700'))
        self.title(f"Calibrate Time Manually - {self.episode.episode_name}")
        self.init_ui()
        self.result_offset_ms = None
    
    def init_ui(self):
        # Create a main frame to hold everything
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a frame for the plots
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 1. Radar Point Cloud (Left)
        radar_frame = ttk.LabelFrame(plot_frame, text="Radar Point Cloud")
        radar_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.radar_plot = PointCloudFigureFrame(radar_frame,figure_cfg={
            'figsize': (5, 5), 'dpi': 100
        })
        self.radar_plot.pack(fill=tk.BOTH, expand=True)

        # 2. Combined Visualization (Center)
        combined_frame = ttk.LabelFrame(plot_frame, text="Combined Visualizer")
        combined_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.combined_plot = CombinedFigureFrame(combined_frame, figure_cfg={'figsize': (5, 5), 'dpi': 100})
        self.combined_plot.pack(fill=tk.BOTH, expand=True)
        
        # 3. Skeleton Visualization (Right)
        skel_frame = ttk.LabelFrame(plot_frame, text="Skeleton Visualization")
        skel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.skel_plot = SkeletonFigureFrame(skel_frame, figure_cfg={'figsize': (5, 5), 'dpi': 100})
        self.skel_plot.pack(fill=tk.BOTH, expand=True)
        
        # Frame for progress bars and button
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        pcd_ts_values = self.episode.pcd_df['ts'].unique().tolist()
        skel_unix_values = self.episode.skel_df['unix_ms'].tolist()
        min_ts = min(min(pcd_ts_values), min(skel_unix_values))
        max_ts = max(max(pcd_ts_values), max(skel_unix_values))

        # Draggable progress bar for radar timestamps
        self.radar_progress = DraggableProgressBar(control_frame, pcd_ts_values, min_val=min_ts, max_val=max_ts, bg="lightgray", update_fn=self.update_visualizations)
        self.radar_progress.pack(fill=tk.X, padx=5, pady=5)
        
        # Draggable progress bar for skeleton timestamps
        self.skel_progress = DraggableProgressBar(control_frame, skel_unix_values, min_val=min_ts, max_val=max_ts, bg="lightgray", update_fn=self.update_visualizations)
        self.skel_progress.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(
            control_frame,
            text="Hotkeys: A/D move radar, Q/E move skeleton.",
        ).pack(pady=2)

        # Button to apply calibration
        ttk.Button(control_frame, text="Apply Calibration", command=self.apply_calibration).pack(pady=5)

        # Global key bindings for quick manual calibration.
        self.bind("<KeyPress-a>", lambda _e: self.radar_progress.step(-1))
        self.bind("<KeyPress-d>", lambda _e: self.radar_progress.step(1))
        self.bind("<KeyPress-q>", lambda _e: self.skel_progress.step(-1))
        self.bind("<KeyPress-e>", lambda _e: self.skel_progress.step(1))
        self.bind("<KeyPress-A>", lambda _e: self.radar_progress.step(-1))
        self.bind("<KeyPress-D>", lambda _e: self.radar_progress.step(1))
        self.bind("<KeyPress-Q>", lambda _e: self.skel_progress.step(-1))
        self.bind("<KeyPress-E>", lambda _e: self.skel_progress.step(1))
        self.focus_set()
        
        # Initially update visualizations based on the first timestamp
        self.update_visualizations()
    
    def update_visualizations(self):
        # Get the currently selected timestamp from each progress bar.
        radar_ts = self.radar_progress.get_current_value()
        skel_ts = self.skel_progress.get_current_value()
        print(f"Radar TS: {radar_ts}, Skeleton TS: {skel_ts}")
        
        # Assume these methods exist to extract a frame given a timestamp.
        radar_frame_points: List["SimplePoint5D"] = self.episode.get_pcd_frame_by('ts', radar_ts, encapsulate=True)
        skeleton_frame = self.episode.get_skel_frame_by('unix_ms', skel_ts, encapsulate=True)
        
        # Update the figures.
        self.radar_plot.update(radar_frame_points)
        self.skel_plot.update(skeleton_frame)
        self.combined_plot.update(skeleton_frame, radar_frame_points)
    
    def apply_calibration(self):
        radar_ts = self.radar_progress.get_current_value()
        skel_ts = self.skel_progress.get_current_value()
        # Return only the offset; alignment will apply this later.
        self.result_offset_ms = int(round(radar_ts - skel_ts))
        self.destroy()  # close the window after applying calibration
    
class DraggableProgressBar(tk.Canvas):
    def __init__(self, parent, timestamps, width=1200, height=30, update_fn=lambda: None,
                 min_val=None, max_val=None, **kwargs):
        super().__init__(parent, width=width, height=height, **kwargs)
        self.timestamps = sorted(timestamps)
        self.width = width
        self.height = height
        self.current_value = self.timestamps[0]
        self.min_val = min_val or self.timestamps[0]
        self.max_val = max_val or self.timestamps[-1]
        self.marker = None
        self.update_fn = update_fn
        self._update_job = None

        # Bind mouse events
        self.bind("<Button-1>", self.on_click)
        self.bind("<B1-Motion>", self.on_drag)
        # Bind arrow keys; they will work once this widget has focus.
        self.bind("<Left>", self.on_left_key)
        self.bind("<Right>", self.on_right_key)

        self.draw_bar()

    def schedule_update(self):
        # Cancel any pending update to avoid redundant refresh calls.
        if self._update_job is not None:
            self.after_cancel(self._update_job)
        # Schedule update_fn to be called after a short delay.
        self._update_job = self.after(50, self.update_fn)
        
    def draw_bar(self):
        self.delete("all")
        # Draw the progress bar background.
        self.create_rectangle(0, self.height/3, self.width, 2*self.height/3, fill="gray", outline="")

        # Draw markers for each timestamp (normalized position).
        for ts in self.timestamps:
            if ts < self.min_val or ts > self.max_val:
                continue
            x = (ts - self.min_val) / (self.max_val - self.min_val) * self.width
            self.create_line(x, 0, x, self.height, fill="black")

        # Create the draggable marker (initially at the first timestamp).
        self.marker = self.create_oval(0, 0, 10, 10, fill="red")
        self.tag_bind(self.marker, "<Button-1>", self.on_click)
        self.tag_bind(self.marker, "<B1-Motion>", self.on_drag)
        self.update_marker_position()

    def update_marker_position(self):
        x = (self.current_value - self.min_val) / (self.max_val - self.min_val) * self.width
        # Center the marker vertically.
        self.coords(self.marker, x-5, self.height/2-5, x+5, self.height/2+5)

    def on_click(self, event):
        # Set focus so that arrow keys are handled by this widget.
        # Set focus so that arrow keys are handled by this widget.
        self.focus_set()
        self.move_marker(event.x)

    def on_drag(self, event):
        # Set focus on drag as well.
        self.focus_set()
        self.move_marker(event.x)

    def move_marker(self, x):
        # Calculate the corresponding timestamp and snap to the nearest one.
        ts = self.min_val + (x / self.width) * (self.max_val - self.min_val)
        closest_ts = min(self.timestamps, key=lambda t: abs(t - ts))
        self.current_value = closest_ts
        self.update_marker_position()
        self.schedule_update()

    def on_left_key(self, event):
        # Move marker to the previous timestamp.
        idx = self.timestamps.index(self.current_value)
        if idx > 0:
            self.current_value = self.timestamps[idx - 1]
            self.update_marker_position()
            self.schedule_update()

    def on_right_key(self, event):
        # Move marker to the next timestamp.
        idx = self.timestamps.index(self.current_value)
        if idx < len(self.timestamps) - 1:
            self.current_value = self.timestamps[idx + 1]
            self.update_marker_position()
            self.schedule_update()

    def get_current_value(self):
        return self.current_value

    def step(self, delta: int):
        if not self.timestamps:
            return
        idx = self.timestamps.index(self.current_value)
        idx = max(0, min(len(self.timestamps) - 1, idx + int(delta)))
        self.current_value = self.timestamps[idx]
        self.update_marker_position()
        self.schedule_update()
