import numpy as np
import apps.lateral_tracking.constants as const
import math
import time
from filterpy.kalman import KalmanFilter
from ..utils import (
    apply_DBscan,
    RingBuffer,
)
from typing import List
from enum import Enum

ACTIVE = 1
INACTIVE = 0

class Status(Enum):
    STATIC = 0,
    DYNAMIC = 1

STATIC = True
DYNAMIC = False


class BatchedData(RingBuffer):
    """
    A class to manage and combine frames into a batch.

    Attributes:
    ----------
    - effective_data (numpy.ndarray): An array to store effective data frames.

    Methods:
    -------
    - empty(): Reset the buffer and create an empty effective_data array.
    - add_frame(new_data: numpy.ndarray): Add a new frame of data to the buffer.
    - clear(): Clear the buffer and reset effective_data.
    - change_buffer_size(new_size): Change the size of the buffer.
    - pop_frame(): Remove the oldest frame from the buffer.
    """

    def __init__(self):
        super().__init__(const.FB_FRAMES_BATCH + 1, init_val=np.empty((0, 8)))
        self.effective_data = np.empty((0, 8))

    def add_frame(self, new_data: np.array):
        """
        Add a new frame of data to the buffer.
        """
        while len(self.buffer) >= self.size:
            self.pop_frame()

        super().append(new_data)
        self.effective_data = np.concatenate(list(self.buffer), axis=0)

    def clear(self):
        """
        Clear the buffer and reset effective_data.
        """
        self.buffer.clear()
        self.effective_data = np.array([])

    def change_buffer_size(self, new_size):
        """
        Change the size of the buffer.
        """
        self.size = new_size

    def pop_frame(self):
        """
        Remove the oldest frame from the buffer.
        """
        if len(self.buffer) > 0:
            self.buffer.popleft()


class KalmanState(KalmanFilter):
    """
    A class representing the state of a Kalman filter for motion tracking.

    Attributes:
    ----------
    - centroid: The centroid of the track used for initializing this Kalman filter instance.

    Methods:
    -------
    - __init__(centroid: np.ndarray): Initialize the Kalman filter with default parameters based on the centroid.
    """

    def __init__(self, centroid: np.ndarray):
        super().__init__(
            dim_x=const.MOTION_MODEL.KF_DIM[0], dim_z=const.MOTION_MODEL.KF_DIM[1]
        )

        self.F = const.MOTION_MODEL.KF_F(1)
        self.H = const.MOTION_MODEL.KF_H
        self.Q = const.MOTION_MODEL.KF_Q_DISCR(1)
        self.R = np.eye(const.MOTION_MODEL.KF_DIM[1]) * const.KF_R_STD**2
        self.x = np.array([const.MOTION_MODEL.STATE_VEC(centroid)]).T
        self.P = np.eye(const.MOTION_MODEL.KF_DIM[0]) * const.KF_P_INIT


class PointCluster:
    """
    A class representing a cluster of 3D points and its attributes.

    Attributes:
    ----------
    - pointcloud (numpy.ndarray): An array of 3D points in the form (x, y, z, x', y', z', r', s).
    - point_num (int): The number of points in the cluster.
    - centroid (numpy.ndarray): The centroid of the cluster.
    - min_vals (numpy.ndarray): The minimum values in each dimension of the pointcloud.
    - max_vals (numpy.ndarray): The maximum values in each dimension of the pointcloud.
    - status (bool): The cluster movement status (STATIC: True, DYNAMIC: False)

    Methods:
    -------
    - __init__(pointcloud: numpy.ndarray):
        Initialize PointCluster with a given pointcloud.

    """

    def __init__(self, pointcloud: np.array):
        """
        Initialize PointCluster with a given pointcloud.
        """

        # NOTE: the input is now a list of 8 entries
        self.pointcloud = pointcloud
        self.point_num = pointcloud.shape[0]
        self.centroid = np.mean(pointcloud[:, :6], axis=0)
        self.min_vals = np.min(pointcloud[:, :6], axis=0)
        self.max_vals = np.max(pointcloud[:, :6], axis=0)

