from typing import List, Optional, Sequence, Union

import numpy as np
from mwcore.visualization import OnlinePointCloudVisualizer
from mwpose3d.registry import VISUALIZERS
from PySide6.QtCore import Slot
import pyqtgraph.opengl as gl
import logging

logger = logging.getLogger(__name__)


COLOR_DEFAULT = (0, 1, 0, 1)   # green
COLOR_POINTING = (0, 0, 1, 1)  # blue


@VISUALIZERS.register_module()
class OnlineSkeletonVisualizer(OnlinePointCloudVisualizer):
    def __init__(self,
                 parent=None,
                 on_close: Optional[callable] = None,
                 joint_cnxn: Optional[Sequence[Sequence[int]]] = None,
                 joint_indices: Optional[Sequence[int]] = None,
                 highlight_pointing: bool = False):
        super().__init__(parent, on_close)
        self.cnxn_matrix = joint_cnxn if joint_cnxn is not None else None
        self.joint_indices = list(joint_indices) if joint_indices is not None else None
        self._joint_set = set(self.joint_indices) if self.joint_indices is not None else None
        self._skel_scatter = gl.GLScatterPlotItem(size=5, color=COLOR_DEFAULT)
        if self.cnxn_matrix is not None:
            assert self.joint_indices is not None
            self.plot3d.plot_3d.addItem(self._skel_scatter)
            self._skel_lines: List[gl.GLLinePlotItem] = []

        # Pointing detection
        self._pointing_detector = None
        if highlight_pointing:
            from mwpose3d.utils.pointing_detector import PointingDetector
            self._pointing_detector = PointingDetector()
    
    def update_skeleton(self,
                        skeleton: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]):
        flat = np.asarray(skeleton, dtype=float).flatten()
        joints = flat.reshape(-1, 3)

        if self.joint_indices is not None and joints.shape[0] != len(self.joint_indices):
            logger.debug("Skeleton joint count does not match expected count. "
                           f"Expected {len(self.joint_indices)}, got {joints.shape[0]}.")
            joints = joints[self.joint_indices]

        # Determine color based on pointing detection
        if self._pointing_detector is not None:
            is_pointing = self._pointing_detector.detect_frame(flat)
            color = COLOR_POINTING if is_pointing else COLOR_DEFAULT
        else:
            color = COLOR_DEFAULT

        self._skel_scatter.setData(pos=joints, color=color)

        if self.cnxn_matrix is None:
            return

        line_idx = 0
        for start, end in self.cnxn_matrix:
            if start not in self._joint_set or end not in self._joint_set:
                continue
            i = self.joint_indices.index(start)
            j = self.joint_indices.index(end)
            pts = np.vstack((joints[i], joints[j]))
            if line_idx < len(self._skel_lines):
                self._skel_lines[line_idx].setData(pos=pts, color=color)
            else:
                line_item = gl.GLLinePlotItem(
                    pos=pts,
                    color=color,
                    width=2,
                    antialias=True,
                    mode='lines'
                )
                self.plot3d.plot_3d.addItem(line_item)
                self._skel_lines.append(line_item)
            line_idx += 1
