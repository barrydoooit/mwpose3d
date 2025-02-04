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

from .Tracking import BatchedData, KalmanState, PointCluster, Status



class ClusterTrack:
    """
    A class representing a tracked cluster with a Kalman filter for motion estimation.

    Parameters
    ----------
    cluster : PointCluster
        The initial point cluster associated with the track.

    Attributes
    ----------
    N_est : int
        Estimated number of points in the cluster.
    spread_est : numpy.ndarray
        Estimated spread of measurements in each dimension.
    group_disp_est : numpy.ndarray
        Estimated group dispersion matrix.
    cluster : PointCluster
        PointCluster associated with the track.
    batch : BatchedData
        The collection of overlaying previous frames
    state : KalmanState
        KalmanState instance for motion estimation.
    status : int (INACTIVE or ACTIVE)
        Current status of the track.
    num_points_associated_last : int
        Number of points associated with the track in the last frame.
    num_dynamic_points_associated_last : int
        Number of dynamic points associated with the track in the last frame.
    track_status : Status
        The status of the track (STATIC or DYNAMIC).
    color : numpy.ndarray
        Random color assigned to the track for visualization (for visualization purposes).

    Methods
    -------
    compute_cartesian_velocity()
        Compute the cartesian velocity of the track using the a priori state, with respect to the x and y dimensions.
    
    __get_num_dynamic_points_associated(pointcloud)
        Get the number of dynamic points associated with the track.

    predict_state(dt)
        Predict the state of the Kalman filter based on the time multiplier.

    _estimate_point_num()
        Estimate the number of points in the cluster.

    _estimate_measurement_spread()
        Estimate the spread of measurements in each dimension.

    _estimate_group_disp_matrix()
        Estimate the group dispersion matrix.

    _get_D()
        Calculate and get the dispersion matrix for the track.

    associate_pointcloud(pointcloud)
        Associate a new pointcloud with the track.

    get_Rm()
        Get the measurement covariance matrix.

    _get_Rc()
        Get the combined covariance matrix.

    update_state()
        Update the state of the Kalman filter based on the associated pointcloud.

    seek_inner_clusters()
        Seek inner clusters within the current track.

    """

    def __init__(self, cluster: PointCluster):
        self.N_est = 0
        self.spread_est = np.zeros(const.MOTION_MODEL.KF_DIM[1])
        self.group_disp_est = (
            np.eye(const.MOTION_MODEL.KF_DIM[1]) * const.KF_GROUP_DISP_EST_INIT
        )
        self.cluster = cluster
        self.batch = BatchedData()
        self.state = KalmanState(cluster.centroid)

        self.num_points_associated_last = cluster.point_num
        self.num_dynamic_points_associated_last = self._get_num_dynamic_points_associated(cluster.pointcloud)

        self.track_status = Status.DYNAMIC if self.num_dynamic_points_associated_last > const.NUM_DYNAMIC_POINTS_THRESHOLD else Status.STATIC

        self.color = np.random.rand(
            3,
        )
        
    def compute_cartesian_velocity(self):
        """
        Compute the cartesian velocity of the track using the a priory state, with respect to the x and y dimensions.
        """
        return math.sqrt(np.sum((self.state.x_prior[3:5] ** 2)))
    
    def _get_num_dynamic_points_associated(self, pointcloud: np.array):
        """
        Get the number of dynamic points associated with the track.

        A point is considered dynamic if its Doppler value is greater than the DOPPLER_THRESHOLD.
        """
        return 0 if not len(pointcloud) else np.sum(pointcloud[:, 6] > const.DOPPLER_THRESHOLD)

    def _estimate_point_num(self):
        """
        Estimate the expected number of points in the cluster.
        """
        if const.KF_ENABLE_EST:
            # TODO: Instead of self.cluster.point_num, use my_good_points
            if self.cluster.point_num > self.N_est:
                self.N_est = self.cluster.point_num
            else:
                # Weighted average between the current number of points and the estimated number of points
                self.N_est = (
                    1 - const.KF_A_N
                ) * self.N_est + const.KF_A_N * self.cluster.point_num
        else:
            self.N_est = max(const.KF_EST_POINTNUM, self.cluster.point_num)

    def _estimate_measurement_spread(self):
        """
        Estimate the spread of measurements in each dimension.
        """
        if self.cluster.point_num > 1:
            for m in range(len(self.cluster.min_vals)):
                # Difference between max and min values in the cluster (in one dimension)
                spread = self.cluster.max_vals[m] - self.cluster.min_vals[m]

                # Unbiased spread estimation - the more points we have, the tighter the spread we create is
                spread = (
                    # TODO: Use my_good_points instead of self.cluster.point_num
                    spread * (self.cluster.point_num + 1) / (self.cluster.point_num - 1)
                )

                # Map the spread to a range between 1 and 2 times between the configured spread limits
                spread = min(2 * const.KF_SPREAD_LIM[m], spread)
                spread = max(const.KF_SPREAD_LIM[m], spread)

                if spread > self.spread_est[m]:
                    # This would most likely be the case when we have few samples
                    self.spread_est[m] = spread
                else:
                    # Weighed average between calculated spread and the previous spread estimation
                    self.spread_est[m] = (1.0 - const.KF_A_SPR) * self.spread_est[
                        m
                    ] + const.KF_A_SPR * spread

    def _get_D(self):
        """
        Calculate and get the dispersion matrix for the current cluster.

        Returns
        -------
        numpy.ndarray
            Dispersion matrix for the cluster.
        """
        dimension = const.MOTION_MODEL.KF_DIM[1]
        pointcloud = self.cluster.pointcloud
        centroid = self.cluster.centroid
        disp = np.zeros((dimension, dimension), dtype="float")

        for i in range(dimension):
            for j in range(dimension):
                disp[i, j] = np.mean(
                    (pointcloud[:, i] - centroid[i]) * (pointcloud[:, j] - centroid[j])
                )

        return disp

    def _estimate_group_disp_matrix(self):
        """
        Estimate the group dispersion matrix.
        """
        a = self.cluster.point_num / self.N_est
        self.group_disp_est = (1 - a) * self.group_disp_est + a * self._get_D()

    def _get_Rc(self):
        """
        Get the combined covariance matrix.

        Returns
        -------
        numpy.ndarray
            Combined covariance matrix for the cluster.
        """
        N = self.cluster.point_num
        N_est = self.N_est
        return (self.get_Rm() / N) + (
            (N_est - N) / ((N_est - 1) * N)
        ) * self.group_disp_est

    def associate_pointcloud(self, pointcloud: np.array):
        """
        Associate a point cloud with the track.

        Parameters
        ----------
        pointcloud : np.array
            2D NumPy array representing the point cloud.

        Notes
        -----
        This method performs the following steps:
        1. Updates the number of points associated with the track.
        2. Updates the number of dynamic points associated with the track.
        3. In case there are point associated with this track, perform the other steps.
        4. Initializes a PointCluster with the given point cloud.
        5. Adds the point-cluster to the track's frames batch.
        """

        # Update the number of points and dynamic associated with the track.
        self.num_points_associated_last = len(pointcloud)
        self.num_dynamic_points_associated_last = self._get_num_dynamic_points_associated(pointcloud)

        print('points associated with the track -- ', len(pointcloud))
        print('dynamic points associated with the track -- ', self.num_dynamic_points_associated_last)

        # If there are points associated with this track, update the track.
        if len(pointcloud):
            self.cluster = PointCluster(pointcloud)
            self.batch.add_frame(self.cluster.pointcloud)

    def get_Rm(self):
        """
        Get the measurement covariance matrix

        Returns
        -------
        numpy.ndarray
            Measurement covariance matrix for the cluster.
        """
        return np.diag(((self.spread_est / 2) ** 2))

    def predict_state(self, dt: float):
        """
        Predict the state of the Kalman filter based on the time multiplier.

        Parameters
        ----------
        dt : float
            Time multiplier for the prediction.
        """
        if self.track_status is Status.DYNAMIC:
            self.state.predict(
                F=const.MOTION_MODEL.KF_F(dt),
                Q=const.MOTION_MODEL.KF_Q_DISCR(dt),
            )
        
    def update_state(self):
        """
        Update the track.
        """
        # TODO: Calculate my_good_points - dynamic (Doppler more than 0) and unique (association with only one track)
        vel = self.compute_cartesian_velocity()
        if not self.num_points_associated_last:
            if self.track_status is Status.DYNAMIC:
                if vel < const.MIN_VELOCITY_STOP_NO_POINTS:
                    # If the track is dynamic and no points are associated, force zero velocity.
                    self.state.x[3:6] = 0
                    # If the track is dynamic and no points are associated, transition to STATIC.
                    self.track_status = Status.STATIC
                else: 
                    self._move_target()
            else:
                # If the track is static and no points are associated, do not update the state.
                return
        elif self.num_dynamic_points_associated_last < const.NUM_DYNAMIC_POINTS_THRESHOLD + 1:
            if self.track_status is Status.STATIC:
                # TODO: Update confidence.
                return
            else:
                if vel < const.MIN_VELOCITY_STOP_NO_DYNAMIC_POINTS:
                    # If the track is dynamic and no dynamic points are associated, force zero velocity.
                    self.state.x[3:6] = 0
                    # If the track is dynamic and no dynamic points are associated, transition to STATIC.
                    self.track_status = Status.STATIC 
                    # TODO: If there are many STATIC points, increase confidence.
                elif vel < const.MIN_VELOCITY_SLOW_DOWN:
                    # If the track is dynamic and no dynamic points are associated, decrease the velocity.
                    self.state.x[3:6] *= 0.5
                    self._move_target()
                else:
                    # TODO: Increase confidence.
                    self._move_target() 
        elif self.num_dynamic_points_associated_last > const.NUM_DYNAMIC_POINTS_THRESHOLD:
            self._estimate_point_num()
            self._estimate_measurement_spread()
            self._estimate_group_disp_matrix()

            self._move_target()

    def _move_target(self):
        """
        Move the target.
        """
        self.track_status = Status.DYNAMIC
        z = np.array(self.cluster.centroid)
        self.state.update(z, R=self._get_Rc())

        variance = z[:1] - self.state.x[:1, 0]
        self.state.x[:1, 0] += variance * 0.4

