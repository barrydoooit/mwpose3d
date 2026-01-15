from typing import List, Optional, Sequence, Union

import numpy as np
try:
    from mwcore.visualization import OnlinePointCloudVisualizer
except ImportError:
    pass
from mwpose3d.registry import VISUALIZERS
import pyqtgraph.opengl as gl
import logging

logger = logging.getLogger(__name__)


@VISUALIZERS.register_module()
class OnlineSkeletonVisualizer(OnlinePointCloudVisualizer):
    def __init__(self,
                 parent=None,
                 on_close: Optional[callable] = None,
                 joint_cnxn: Optional[Sequence[Sequence[int]]] = None,
                 joint_indices: Optional[Sequence[int]] = None,
                 skeleton_offset: Optional[Sequence[float]] = None):
        super().__init__(parent, on_close)
        self.cnxn_matrix = joint_cnxn if joint_cnxn is not None else None
        self.joint_indices = list(joint_indices) if joint_indices is not None else None
        self._joint_set = set(self.joint_indices) if self.joint_indices is not None else None
        self.skeleton_offset = np.array(skeleton_offset, dtype=float) if skeleton_offset is not None else None
        self._skel_scatter = gl.GLScatterPlotItem(size=5, color=(0, 1, 0, 1))
        if self.cnxn_matrix is not None:
            assert self.joint_indices is not None
            self.plot3d.plot_3d.addItem(self._skel_scatter)
            self._skel_lines: List[gl.GLLinePlotItem] = []
    
    def update_skeleton(self,
                        skeleton: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]):
        flat = np.asarray(skeleton, dtype=float).flatten()
        joints = flat.reshape(-1, 3)

        # Apply offset to transform skeleton back to original coordinates
        if self.skeleton_offset is not None:
            joints = joints + self.skeleton_offset

        if self.joint_indices is not None and joints.shape[0] != len(self.joint_indices):
            logger.debug("Skeleton joint count does not match expected count. "
                           f"Expected {len(self.joint_indices)}, got {joints.shape[0]}.")
            joints = joints[self.joint_indices]
        
        self._skel_scatter.setData(pos=joints)

        if self.cnxn_matrix is None:
            return
        
        for line in self._skel_lines:
            self.plot3d.plot_3d.removeItem(line)
        self._skel_lines.clear()

        for start, end in self.cnxn_matrix:
            if start not in self._joint_set or end not in self._joint_set:
                continue
            i = self.joint_indices.index(start)
            j = self.joint_indices.index(end)
            pts = np.vstack((joints[i], joints[j]))
            line_item = gl.GLLinePlotItem(
                pos=pts,
                color=(1,1,1,1),
                width=2,
                antialias=True,
                mode='lines'
            )
            self.plot3d.plot_3d.addItem(line_item)
            self._skel_lines.append(line_item)
