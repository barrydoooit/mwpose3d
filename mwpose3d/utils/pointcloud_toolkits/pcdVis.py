from typing import Iterable
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from mwpose3d.utils.pointcloud_toolkits.structures import SimplePoint3D
import logging
logger = logging.getLogger(__name__)



class PointCloudFigure(Figure):
    def __init__(self, figure_cfg: dict):
        super().__init__(**figure_cfg)
        self.ax: Axes3D = self.add_subplot(111, projection='3d')
        self._setup_axes()
        # Pre-create the scatter object with empty data.
        self.scatter = self.ax.scatter([], [], [], c='b', marker='o')
        
    def _setup_axes(self):
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(2, -2)
        self.ax.set_ylim(2, 0)  # Reverse Y axis
        self.ax.set_zlim(-2, 2)
    
    def update_points(self, points: Iterable[SimplePoint3D]):
        # Collect coordinates from the points.
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        zs = [point.z for point in points]
        
        # Update the scatter object without clearing the axes.
        self.scatter._offsets3d = (xs, ys, zs)


class PointCloudFigureFrame(tk.Frame):
    def __init__(self, parent: tk.Widget,
                 figure_cfg: dict):
        super().__init__(parent)
        logger.warning(f'Visualizing point cloud with Tkinter is slow. Please consider use the Pyqt6-based visualizer in mwCore')
        self.figure = PointCloudFigure(figure_cfg)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        # self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        # self.toolbar.update()
        # self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)
    
    def update(self, points: Iterable[SimplePoint3D]):
        self.figure.update_points(points)
        self.canvas.draw()
    
    def reset(self):
        self.figure.ax.clear()
        self.canvas.draw()