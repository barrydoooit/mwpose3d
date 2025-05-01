from typing import Iterable, Iterator, List, Optional, Union
from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton)
from PySide2.QtCore import QTimer, Qt
import numpy as np
import sys
import pyqtgraph.opengl as gl
from .plot_3d import Plot3D
import os
from PySide2 import QtCore
pyside2_plugin_path = os.path.join(
    os.path.dirname(QtCore.__file__),
    "plugins",
    "platforms",
)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyside2_plugin_path




class PointCloudOfflineVisualizer(QMainWindow):
    def __init__(self, 
                 point_clouds: Union[List[np.ndarray], Iterator[np.ndarray]],
                 total_frames: Optional[int] = None,
                 play_fps: float = 5,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Offline Point Cloud Visualizer")
        self.resize(800, 600)
        
        if isinstance(point_clouds, list):
            self._is_list = True
            self.cache: List[np.ndarray] = point_clouds
            self.iterator: Optional[Iterator[np.ndarray]] = None
            self.num_frames = len(point_clouds)
        elif isinstance(point_clouds, Iterable):
            self._is_list = False
            self.cache: List[np.ndarray] = []
            self.iterator: Iterator[np.ndarray] = iter(point_clouds)
            self.num_frames = total_frames
        
        self.current_frame = 0
        self.play_fps = play_fps
        self.playing = False
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(int(1000 / self.play_fps))
        
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        self.plot3d = Plot3D()
        layout.addWidget(self.plot3d.plot_3d)
         
        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.play_btn = QPushButton("PLAY")
        self.prev_btn = QPushButton("PREV")
        self.next_btn = QPushButton("NEXT")
        controls.addWidget(self.play_btn)
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.next_btn)
        
        self.play_btn.clicked.connect(self.toggle_play)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        
        self.play_key = Qt.Key_Space
        self.prev_key = Qt.Key_A
        self.next_key = Qt.Key_D
        
    def initilaize(self):
        self._load_frame(0)
        self.update_display()
    
    def show(self):
        super().show()
        self.initilaize()
        
    def _load_frame(self, index: int) -> None:
        """
        Ensure the frame at `index` is loaded into cache. For iterators, fetch until that frame.
        """
        if self._is_list:
            return
        # If we've already cached this frame, do nothing
        if index < len(self.cache):
            return
        # Fetch frames up to the requested index
        try:
            while len(self.cache) <= index:
                frame = next(self.iterator)
                self.cache.append(frame)
        except StopIteration:
            # Reached end of iterator
            # Optionally loop: reset iterator if length unknown
            if self.num_frames:
                # If we know total, wrap around
                self.current_frame = 0
                return
            else:
                # Stop playback
                self.playing = False
                self.timer.stop()
                
    def keyPressEvent(self, event):
        if event.key() == self.play_key:
            self.toggle_play()
        elif event.key() == self.prev_key and not self.playing:
            self.prev_frame()
        elif event.key() == self.next_key and not self.playing:
            self.next_frame()
        super().keyPressEvent(event)
    
    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_btn.setText("PAUSE")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.timer.start()
        else:
            self.play_btn.setText("PLAY")
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.timer.stop()
    
    def next_frame(self):
        target = self.current_frame + 1
        # For lists, wrap by modulo
        if self._is_list and self.num_frames:
            self.current_frame = target % self.num_frames
        else:
            self.current_frame = target
        self._load_frame(self.current_frame)
        # Clamp if past end of cache and no wrap
        if self.current_frame >= len(self.cache):
            self.current_frame = len(self.cache) - 1
        self.update_display()
    
    def prev_frame(self):
        if self._is_list and self.num_frames:
            self.current_frame = (self.current_frame - 1) % self.num_frames
        else:
            self.current_frame = max(0, self.current_frame - 1)
        self.update_display()
    
    def update_display(self):
        cloud = self.cache[self.current_frame]
        # Ensure shape Nx3
        if cloud.ndim == 2 and cloud.shape[1] >= 3:
            pts = cloud[:, :3]
        else:
            pts = np.zeros((0, 3))
        self.plot3d.scatter.setData(pos=pts)
        title = f"Frame {self.current_frame + 1}"
        if self.num_frames:
            title += f"/{self.num_frames}"
        self.setWindowTitle(title)

class PointCloudOfflineVisualizerSK(PointCloudOfflineVisualizer):
    def __init__(self,
                 point_clouds: Union[List[np.ndarray], Iterator[np.ndarray]],
                 skeletons: Union[List[np.ndarray], Iterator[np.ndarray]],
                 total_frames: Optional[int] = None,
                 play_fps: float = 5,
                 parent=None):
        if isinstance(skeletons, list):
            self._skel_is_list = True
            self._skel_cache = skeletons
            self._skel_iter = None
        else:
            self._skel_is_list = False
            self._skel_cache = []
            self._skel_iter = iter(skeletons)
        super().__init__(point_clouds, total_frames, play_fps, parent)

        self._skel_scatter = gl.GLScatterPlotItem(size=10, color=(1, 0, 0, 1))
        self.plot3d.plot_3d.addItem(self._skel_scatter)
    
    def _load_frame(self, index: int) -> None:
        super()._load_frame(index)
        if not self._skel_is_list:
            while len(self._skel_cache) <= index:
                try:
                    self._skel_cache.append(next(self._skel_iter))
                except StopIteration:
                    break
    
    def update_display(self):
        super().update_display()
        if self.current_frame < len(self._skel_cache):
            skel_flat = self._skel_cache[self.current_frame]
            joints = np.array(skel_flat).reshape(-1, 3)
            self._skel_scatter.setData(pos=joints)
        else:
            self._skel_scatter.setData(pos=np.zeros((0, 3)))