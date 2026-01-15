from typing import List, Optional, Sequence, Union

import numpy as np
try:
    from mwcore.visualization import OnlinePointCloudVisualizer
except ImportError:
    pass
from mwpose3d.registry import VISUALIZERS
# from PySide6.QtCore import Slot
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
                 skeleton_offset: Optional[Sequence[float]] = None,
                 use_pcd_centroid: bool = True):
        super().__init__(parent, on_close)
        self.cnxn_matrix = joint_cnxn if joint_cnxn is not None else None
        self.joint_indices = list(joint_indices) if joint_indices is not None else None
        self._joint_set = set(self.joint_indices) if self.joint_indices is not None else None
        self.skeleton_offset = np.array(skeleton_offset, dtype=float) if skeleton_offset is not None else None
        self.use_pcd_centroid = use_pcd_centroid
        self._pcd_centroid = np.zeros(3)  # Track point cloud centroid
        self._skel_scatter = gl.GLScatterPlotItem(size=5, color=(0, 1, 0, 1))
        if self.cnxn_matrix is not None:
            assert self.joint_indices is not None
            self.plot3d.plot_3d.addItem(self._skel_scatter)
            self._skel_lines: List[gl.GLLinePlotItem] = []
    
    def on_new_cloud(self, cloud_data):
        """Override to capture point cloud centroid for skeleton positioning."""
        # Extract XYZ from cloud data and compute centroid
        if hasattr(cloud_data, 'xyz'):
            pts = cloud_data.xyz
        elif isinstance(cloud_data, np.ndarray) and cloud_data.ndim == 2 and cloud_data.shape[1] >= 3:
            pts = cloud_data[:, :3]
        else:
            pts = None
        
        if pts is not None and len(pts) > 0:
            self._pcd_centroid = np.mean(pts, axis=0)
        
        # Call parent implementation
        super().on_new_cloud(cloud_data)
    
    def update_skeleton(self,
                        skeleton: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]):
        # #region agent log
        import json, time; open('/Users/joaquin/Desktop/delft/mwpose3d/.cursor/debug.log','a').write(json.dumps({"hypothesisId":"H3","location":"skel_online.py:update_skeleton","message":"skeleton_received","data":{"input_type":type(skeleton).__name__,"input_len":len(skeleton) if hasattr(skeleton,'__len__') else None,"joint_indices":self.joint_indices},"timestamp":int(time.time()*1000)})+'\n')
        # #endregion
        flat = np.asarray(skeleton, dtype=float).flatten()
        joints = flat.reshape(-1, 3)

        # Apply offset to transform skeleton to point cloud coordinates
        if self.use_pcd_centroid:
            # Use dynamic point cloud centroid for X/Z, keep Y from fixed offset
            offset = self._pcd_centroid.copy()
            if self.skeleton_offset is not None:
                offset[1] = self.skeleton_offset[1]  # Keep fixed Y offset (height)
            joints = joints + offset
        elif self.skeleton_offset is not None:
            joints = joints + self.skeleton_offset

        # #region agent log
        open('/Users/joaquin/Desktop/delft/mwpose3d/.cursor/debug.log','a').write(json.dumps({"hypothesisId":"H3,H5,H8","location":"skel_online.py:update_skeleton:joints","message":"joints_reshaped","data":{"joints_shape":list(joints.shape),"joints_sample":joints[:3].tolist() if len(joints)>=3 else joints.tolist(),"expected_joints":len(self.joint_indices) if self.joint_indices else None,"pcd_centroid":self._pcd_centroid.tolist(),"use_pcd_centroid":self.use_pcd_centroid},"timestamp":int(time.time()*1000)})+'\n')
        # #endregion

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
