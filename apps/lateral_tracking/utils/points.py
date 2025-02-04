import math
import numpy as np
from sklearn.cluster import DBSCAN
import apps.lateral_tracking.constants as const





def calc_projection_points(x_origin, y_origin, z_origin):
    """
    Calculate the screen projection of a point based on its distance from a reference point.

    Parameters
    ----------
    x_origin : float
        The reference point coordinate along the x axis.

    y_origin : float
        The reference point coordinate along the y axis.

    z_origin : float
        The reference point coordinate along the z (vertical) axis.

    Returns
    -------
    tuple of floats
        The (x,z) coordinates of the point where the line that connects the reference point
        and the sensitive component of the system, cuts the screen.

    """

    x_dist = x_origin - const.M_X
    y_dist = y_origin - const.M_Y
    z_dist = z_origin - const.M_Z

    if x_dist == 0:
        x_proj = x_origin
    else:
        x1 = -const.M_Y / (y_dist / x_dist)
        x_proj = x1 + const.M_X

    if z_dist == 0:
        z_proj = z_origin
    else:
        z1 = -const.M_Y / (y_dist / z_dist)
        z_proj = z1 + const.M_Z

    return x_proj, z_proj


def altered_EuclideanDist(p1, p2):
    """
    Calculate an altered Euclidean distance between two points in 3D space.

    This distance metric incorporates modifications to better suit the characteristics of cylinder-shaped point clouds,
    especially those representing the human silhouette. It achieves this by applying the following adjustments:

    1. **Vertical Weighting**: Reduces the impact of the vertical distance by using a constant `const.DB_Z_WEIGHT`.
    This is beneficial for improved clustering of cylinder-shaped point clouds.

    2. **Inverse Proportional Weighting**: Introduces a weight to the result inversely proportional to the points' y-axis values.
    This ensures that the distance outputs are lower when the point cloud is further away from the sensor and thus, more sparse.

    Returns
    -------
    float
        The adjusted Euclidean distance between the two points.
    """
    # NOTE: The z-axis has less weight in the distance metric since the sillouette of a person is tall and thin.
    # Also, the further away from the sensor the more sparse the points, so we need a weighing factor
    weight = 1 - ((p1[1] + p2[1]) / 2) * const.DB_RANGE_WEIGHT
    return weight * (
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
        + const.DB_Z_WEIGHT * ((p1[2] - p2[2]) ** 2)
    )


def apply_DBscan(pointcloud, eps=const.DB_EPS, min_samples=const.DB_MIN_SAMPLES_MIN):
    """
    Apply DBSCAN clustering to a 3D point cloud using an altered Euclidean distance metric.

    Parameters
    ----------
    pointcloud : array-like
        The 3D point cloud represented as a list or NumPy array.

    eps : float, optional
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        Default is const.DB_EPS.

    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        Default is const.DB_MIN_SAMPLES.

    Returns
    -------
    list
        A list of clustered point clouds, where each cluster is represented as a list of points.
    """
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=altered_EuclideanDist,
    )

    labels = dbscan.fit_predict(pointcloud)

    # label of -1 means noice so we exclude it
    filtered_labels = set(labels) - {-1}

    # Assign points to clusters
    clustered_points = {label: [] for label in filtered_labels}
    for i, label in enumerate(labels):
        if label != -1:
            clustered_points[label].append(pointcloud[i])

    # Return a list of clustered pointclouds
    clusters = list(clustered_points.values())
    return clusters


def point_transform_to_standard_axis(input):
    """
    Transform 3D point coordinates and velocities to a standard axis.

    The transformation includes translation and rotation to bring the input point into a standard coordinate system.

    Parameters
    ----------
    input : array-like
        Input point represented as a 6-element array or list, where the first three elements are coordinates (x, y, z),
        and the last three elements are velocities along the corresponding axes.

    Returns
    -------
    np.array
        Transformed point with coordinates and velocities in the standard axis system.
    """
    # Translation Matrix (T)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, const.S_HEIGHT], [0, 0, 0, 1]])

    # Rotation Matrix (R_inv)
    ang_rad = np.radians(const.S_TILT)
    R_inv = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(ang_rad), -np.sin(ang_rad), 0],
            [0, np.sin(ang_rad), np.cos(ang_rad), 0],
            [0, 0, 0, 1],
        ]
    )

    coordinates = np.concatenate((input[:3], [1]))
    velocities = np.concatenate((input[3:], [0]))
    transformed_coords = np.dot(T, np.dot(R_inv, coordinates))
    transformed_velocities = np.dot(T, np.dot(R_inv, velocities))

    return np.array(
        [
            transformed_coords[0],
            transformed_coords[1],
            transformed_coords[2],
            transformed_velocities[0],
            transformed_velocities[1],
            transformed_velocities[2],
        ]
    )


def normalize_data(detObj):
    """
    Preprocesses the point cloud data from the sensor.

    This function filters the input point cloud, converts radial to Cartesian velocity,
    and transforms the coordinates to the standard vertical-horizontal plane axis system.

    Parameters
    ----------
    detObj : dict
        Dictionary containing the raw detection data with keys:
        - "x": x-coordinate
        - "y": y-coordinate
        - "z": z-coordinate
        - "doppler": Doppler velocity
        - "peakVal": Signal Intensity

    Returns
    -------
    np.ndarray
        Preprocessed data in the standard vertical-horizontal plane axis system.
        Columns:
        - x-coordinate
        - y-coordinate
        - z-coordinate
        - Cartesian velocity along the x-axis
        - Cartesian velocity along the y-axis
        - Cartesian velocity along the z-axis
        - doppler
        - peakval
    """

    input_data = np.vstack(
        (detObj["x"], detObj["y"], detObj["z"], detObj["doppler"], detObj["peakVal"])
    ).T
    ef_data = np.empty((0, 8), dtype="float")

    for index in range(len(input_data)):

        # Transform the radial velocity into Cartesian
        r = math.sqrt(
            input_data[index, 0] ** 2
            + input_data[index, 1] ** 2
            + input_data[index, 2] ** 2
        )
        if r == 0:
            vx = 0
            vy = input_data[index, 3]
            vz = 0
        else:
            if (
                input_data[index, 0] is None
                or input_data[index, 1] is None
                or input_data[index, 2] is None
                or input_data[index, 3] is None
            ):
                print(f"Error: {input_data[index, :]}")

            vx = input_data[index, 3] * input_data[index, 0] / r
            vy = input_data[index, 3] * input_data[index, 1] / r
            vz = input_data[index, 3] * input_data[index, 2] / r

        # Translate points to new coordinate system
        transformed_point = point_transform_to_standard_axis(
            np.array(
                [
                    input_data[index, 0],
                    input_data[index, 1],
                    input_data[index, 2],
                    vx,
                    vy,
                    vz,
                ]
            )
        )

        transformed_point = np.append(
            transformed_point, (input_data[index, 3], input_data[index, 4])
        )

        # Perform scene constraints filtering
        if (
            transformed_point[2] <= 2.5
            and transformed_point[2] > 0
            and transformed_point[1] > 0
        ):
            ef_data = np.append(
                ef_data,
                [transformed_point],
                axis=0,
            )

    return ef_data