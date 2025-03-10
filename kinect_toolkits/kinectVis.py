import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from kinect_toolkits.kinectData import Skeleton
from kinect_toolkits.transforms import kinect_coord_to_radar



class SkeletonFigure:
    def __init__(self, *args, **kwargs):
        self.figure = Figure(*args, **kwargs)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self._setup_axes()
        # Pre-create a scatter plot object (empty for now)
        self.scatter = self.ax.scatter([], [], [], c='b', marker='o')
        # Dictionary to hold line objects for each connection
        self.lines = {} # NOTE: If SkeletonFigure inherits from Figure, this will be considered as Figure children and cause errors.
    
    def _setup_axes(self):
        """Configure the 3D axis for the skeleton display."""
        self.ax.set_title("3D Skeleton Visualization")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.grid(True)
        self.ax.set_xlim(2, -2)
        self.ax.set_ylim(2, 0)
        self.ax.set_zlim(-2, 2)
        
    def update_skeleton(self, skeleton: Skeleton):
        # Cache coordinate conversion results to avoid duplicate work.
        coords = {}
        for key, keypoint in skeleton.keypoints.items():
            coords[key] = kinect_coord_to_radar(keypoint.x, keypoint.y, keypoint.z)
        
        # Update scatter plot data.
        xs, ys, zs = zip(*coords.values())
        # Note: For 3D scatter, you may need to remove and redraw if set_data methods are limited.
        self.scatter._offsets3d = (xs, ys, zs)
        
        # Update or create line objects for connections.
        for key, keypoint in skeleton.keypoints.items():
            for conn in keypoint.connections:
                if conn in skeleton.keypoints:
                    line_key = tuple(sorted((key.value, conn.value)))
                    k1x, k1y, k1z = coords[key]
                    k2x, k2y, k2z = coords[conn]
                    
                    if line_key in self.lines:
                        # Update the existing line
                        line_obj = self.lines[line_key]
                        line_obj.set_data([k1x, k2x], [k1y, k2y])
                        line_obj.set_3d_properties([k1z, k2z])
                    else:
                        # Create a new line and store it.
                        line_obj = self.ax.plot([k1x, k2x], [k1y, k2y], [k1z, k2z], 'r-')[0]
                        print(type(line_obj))
                        self.lines[line_key] = line_obj

class SkeletonFigureFrame(tk.Frame):
    def __init__(self, master=None, figure_cfg: dict={}, **kwargs):
        super().__init__(master, **kwargs)
        figure_cfg.setdefault('figsize', (5, 5))
        figure_cfg.setdefault('dpi', 100)
        self.figure = SkeletonFigure(**figure_cfg)
        
        self.canvas = FigureCanvasTkAgg(self.figure.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def update(self, skeleton: Skeleton):
        self.figure.update_skeleton(skeleton)
        self.canvas.draw()
    
    def reset(self):
        self.figure.ax.clear()
        self.canvas.draw()