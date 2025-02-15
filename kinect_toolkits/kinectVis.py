import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from kinect_toolkits.kinectData import Skeleton
from kinect_toolkits.transforms import kinect_coord_to_radar

class SkeletonFigure(Figure):
    def __init__(self, *args, **kwargs):
        """
        Subclass of matplotlib.figure.Figure for 3D skeleton visualization.
        """
        super().__init__(*args, **kwargs)
        # Create a 3D axis for drawing the skeleton.
        self.ax = self.add_subplot(111, projection='3d')
        self._setup_axes()

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
        self.ax.clear()
        self._setup_axes()

        for keypoint in skeleton.keypoints.values():
            kx, ky, kz = kinect_coord_to_radar(keypoint.x, keypoint.y, keypoint.z)
            self.ax.scatter(kx, ky, kz, c='b', marker='o')
            
        for keypoint in skeleton.keypoints.values():
            for conn in keypoint.connections:
                if conn in skeleton.keypoints:
                    kp2 = skeleton.keypoints[conn]
                    k1x, k1y, k1z = kinect_coord_to_radar(keypoint.x, keypoint.y, keypoint.z)
                    k2x, k2y, k2z = kinect_coord_to_radar(kp2.x, kp2.y, kp2.z)
                    xs = [k1x, k2x]
                    ys = [k1y, k2y]
                    zs = [k1z, k2z]
                    self.ax.plot(xs, ys, zs, 'r-')

class SkeletonFigureFrame(tk.Frame):
    def __init__(self, master=None, figure_cfg: dict={}, **kwargs):
        super().__init__(master, **kwargs)
        figure_cfg.setdefault('figsize', (5, 5))
        figure_cfg.setdefault('dpi', 100)
        self.figure = SkeletonFigure(**figure_cfg)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def update(self, skeleton: Skeleton):
        self.figure.update_skeleton(skeleton)
        self.canvas.draw()
    
    def reset(self):
        self.figure.ax.clear()
        self.figure._setup_axes()
        self.canvas.draw()