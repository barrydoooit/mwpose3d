from typing import List

from .Tracking import BatchedData, PointCluster
from .cluster import ClusterTrack

import numpy as np
import apps.lateral_tracking.constants as const
import time
from filterpy.kalman import KalmanFilter
from ..utils import (
    apply_DBscan,
)

class TrackBuffer:
    """
    A class representing a buffer for managing and updating the multiple ClusterTracks of the scene.

    Attributes
    ----------
    effective_tracks : List[ClusterTrack]
        List of currently active (non-INACTIVE) tracks in the buffer.
    next_track_id : int
        The id int that will be given to the next active track.
    dt : float
        Time multiplier used for predicting states. Indicates the time passed since the previous
        valid observed frame.
    t : float
        Current time when the TrackBuffer is instantiated / updated.

    Methods
    -------
    update_ef_tracks()
        Update the list of effective tracks (excluding INACTIVE tracks).

    has_active_tracks()
        Check if there are active tracks in the buffer.

    _calc_dist_fun(full_set)
        Calculate the Mahalanobis distance matrix for gating.

    _add_tracks(new_clusters)
        Add new tracks to the buffer.

    _predict_all()
        Predict the state of all effective tracks.

    _update_all()
        Update the state of all effective tracks.

    _get_gated_clouds(full_set)
        Gate the pointcloud and return gated and unassigned clouds.

    _associate_points_to_tracks(full_set)
        Associate points to existing tracks and handle inner cluster separation.

    track(pointcloud, batch)
        Perform the tracking process including prediction, association, status update, and clustering.

    estimate_posture(model)
        Estimate the posture of each track in the buffer using a CNN model.

    """

    def __init__(self):
        """
        Initialize TrackBuffer with empty lists for tracks and effective tracks.
        """
        self.effective_tracks: List[ClusterTrack] = []
        self.next_track_id = 0
        self.dt = 0
        self.t = time.time()

    def _find_closest_track(self, full_set: np.array):
        """
        Calculate the Mahalanobis distance matrix for gating.

        Parameters
        ----------
        full_set : np.ndarray
            Full set of points.

        Returns
        -------
        np.ndarray
            An array representing the associated track (entry) for each point (index).
            If no track is associated with a point, the entry is set to None.
        """
        bidding_score = np.empty((full_set.shape[0], len(self.effective_tracks)))
        associated_track_for = np.full(full_set.shape[0], None, dtype=object)

        for j, track in enumerate(self.effective_tracks):
            H_i = np.dot(const.MOTION_MODEL.KF_H, track.state.x_prior).flatten()
            # Group residual covariance matrix
            C_g_j = track.state.P_prior[:6, :6] + track.get_Rm() + track.group_disp_est

            for i, point in enumerate(full_set):
                # Innovation for each measurement
                y_ij = np.array(point[:6]) - H_i

                # Mahalanobis Distance (squared)
                d_squared = np.dot(np.dot(y_ij.T, np.linalg.inv(C_g_j)), y_ij)

                # bidding score (squared)
                bidding_score[i][j] = np.log(np.abs(np.linalg.det(C_g_j))) + d_squared

                # Perform Gate threshold check
                if bidding_score[i][j] < const.TR_GATE:
                    # Just choose the closest mahalanobis distance
                    if associated_track_for[i] is None:
                        associated_track_for[i] = j
                    else:
                        if (
                            bidding_score[i][j]
                            < bidding_score[i][int(associated_track_for[i])]
                        ):
                            associated_track_for[i] = j

        return associated_track_for

    def _add_tracks(self, new_clusters):
        """
        Add new tracks to the buffer.

        Parameters
        ----------
        new_clusters : list
            List of new clusters to be added as tracks.
        """
        for new_cluster in new_clusters:
            new_track = ClusterTrack(PointCluster(np.array(new_cluster)))
            # new_track.id = self.next_track_id
            self.next_track_id += 1
            self.effective_tracks.append(new_track)

    def _predict_all(self):
        """
        Predict the state of all effective tracks.
        """
        for track in self.effective_tracks:
            # TODO: Maybe, accumulate dt for this track in case it is not updated.
            track.predict_state(self.dt)

    def _update_all(self):
        """
        Update the state of all effective tracks.
        """
        for track in self.effective_tracks:
            # TODO: Update only when the track is active (static or dynamic). Delete FREE tracks.
            track.update_state()

    def _get_gated_clouds(self, full_set: np.array):
        """
        Split the pointcloud according to the formed gates and return gated and unassigned clouds.

        Parameters
        ----------
        full_set : np.array
            Full set of points.

        Returns
        -------
        tuple
            Tuple containing unassigned points and clustered clouds.
        """
        unassigned = np.empty((0, 8), dtype="float")
        clusters = [[] for _ in range(len(self.effective_tracks))]
        # Simple matrix has len = len(full_set) and has the index of the chosen track.
        point_to_closest_track_assignment = self._find_closest_track(full_set)

        for i, point in enumerate(full_set):
            if point_to_closest_track_assignment[i] is None:
                unassigned = np.append(unassigned, [point], axis=0)
            else:
                clusters[point_to_closest_track_assignment[i]].append(point)

        return unassigned, clusters

    def _assign_points_to_tracks_and_get_unassigned(self, full_set: np.array):
        """
        Associate points to existing tracks.

        Parameters
        ----------
        full_set : np.array
            Full set of sensed points.

        Returns
        -------
        np.ndarray
            Unassigned points.
        """
        unassigned, clouds = self._get_gated_clouds(full_set)

        for j, track in enumerate(self.effective_tracks):
            track.associate_pointcloud(np.array(clouds[j]))

        return unassigned

    def track(self, pointcloud, batch: BatchedData, isBetweenFrame: bool = False):
        """
        Perform the tracking process including prediction, association, maintenance, update, and clustering.

        Parameters
        ----------
        pointcloud : np.array
            Pointcloud data.
        batch : BatchedData
            BatchedData instance for managing frames.

        Returns
        -------
        None
        """
        # Prediction Step
        self._predict_all()

        # Association Step
        unassigned = self._assign_points_to_tracks_and_get_unassigned(pointcloud)

        # Update Step
        self._update_all()

        # TODO: Move Allocation step before maintenance.

        # TODO: Maintenance Step

        # Clustering of the remainder points Step
        new_clusters = []
        batch.add_frame(unassigned)

        if (
            len(batch.effective_data) > 0
            and len(self.effective_tracks) < const.TR_MAX_TRACKS
        ):
            new_clusters = apply_DBscan(batch.effective_data)

            if len(new_clusters) > 0:
                batch.clear()

            # Create new track for every new cluster
            self._add_tracks(new_clusters)
