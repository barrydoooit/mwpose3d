from typing import Tuple

import numpy as np
from .base import TRANSFORM, BaseTransform


@TRANSFORM.register_module()
class CoordinateTransform(BaseTransform):
    def __init__(self,
                 radar_tilt: float = 5,
                 kinect_tilt: float = 5,  
                 pcd_tran: Tuple[float, float, float] = (0, -2, 0),
                 skel_tran: Tuple[float, float, float] = (-0.37, 0, -2)):
        self.radar_tilt = radar_tilt
        self.kinect_tilt = kinect_tilt
        self.pcd_tran = pcd_tran
        self.skel_tran = skel_tran

    @staticmethod
    def get_rotation_matrix(tilt: float) -> np.ndarray:
        """
        Returns the 3x3 rotation matrix to correct for sensor tilt.
        Tilt is given in degrees. Since the sensor is physically tilted (leaning forward),
        we undo that tilt by rotating by -tilt (about the x-axis).
        """
        angle_rad = np.deg2rad(-tilt)  # negative to counteract the sensor's forward lean
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        # Rotation about x-axis:
        R = np.array([
            [1,      0,       0],
            [0, cos_val, -sin_val],
            [0, sin_val,  cos_val]
        ])
        return R

    @staticmethod
    def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Applies a rotation (R) and translation (t) to a set of 3D points.
        Points is expected to be an (N,3) array.
        """
        return (R @ points.T).T + t

    def transform(self, input: dict):
        """
        Applies the transformation to both radar point clouds (pcd_frames) and Kinect skeletons (skel_frames).

        For radar (pcd_frames):
          - Apply tilt correction using the radar's tilt.
          - Then add the radar translation (pcd_tran) which is in radar coordinates.

        For Kinect skeletons (skel_frames):
          - Reshape each 1D skeleton array into (-1,3) keypoints.
          - Apply tilt correction using the Kinect's tilt.
          - Apply the Kinect translation (skel_tran) in Kinect coordinates.
          - Convert the keypoints from the Kinect coordinate system to the radar coordinate system.
        """
        pcd_frames: Tuple[np.ndarray] = input['pcd_frames']
        skel_frames: Tuple[np.ndarray] = input['skel_frames']

        # --- Process radar point cloud frames ---
        radar_R = self.get_rotation_matrix(self.radar_tilt)
        pcd_tran_array = np.array(self.pcd_tran)
        transformed_pcd_frames = []
        for frame in pcd_frames:
            # Each row: first three columns are (x,y,z)
            points = frame[:, :3]
            transformed_points = self.transform_points(points, radar_R, pcd_tran_array)
            # If additional values exist in the row, preserve them.
            if frame.shape[1] > 3:
                extra = frame[:, 3:]
                transformed_frame = np.hstack([transformed_points, extra])
            else:
                transformed_frame = transformed_points
            transformed_pcd_frames.append(transformed_frame)
        input['pcd_frames'] = tuple(transformed_pcd_frames)

        # --- Process Kinect skeleton frames ---
        # Conversion matrix to map Kinect coordinates to radar coordinates.
        # Kinect: x (right-to-left), y (down-to-up), z (back-to-front)
        # Radar:  x (left-to-right), y (back-to-front), z (down-to-up)
        # Mapping: radar_x = -kinect_x, radar_y = kinect_z, radar_z = kinect_y.
        conversion = np.array([
            [-1, 0, 0],
            [ 0, 0, 1],
            [ 0, 1, 0]
        ])

        kinect_R = self.get_rotation_matrix(self.kinect_tilt)
        skel_tran_array = np.array(self.skel_tran)
        transformed_skel_frames = []
        for frame in skel_frames:
            # Reshape the 1D skeleton array to an (N,3) array of keypoints.
            keypoints = frame[:len(frame) // 3 * 3].reshape(-1, 3)
            # Apply tilt correction in Kinect native coordinates.
            tilted = (kinect_R @ keypoints.T).T
            # Apply Kinect translation in the Kinect coordinate system.
            translated = tilted + skel_tran_array
            # Convert from Kinect's coordinate system to the radar's coordinate system.
            converted = (conversion @ translated.T).T
            # Flatten back to 1D if needed.
            transformed_frame = converted.flatten()
            transformed_skel_frames.append(transformed_frame)
        input['skel_frames'] = tuple(transformed_skel_frames)

        return input