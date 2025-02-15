import os
from typing import Optional
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyquaternion as pyq
from pyquaternion import Quaternion
from io import StringIO
from scipy import signal
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from kinect_toolkits.kinectData import Keypoint, KeypointType, Skeleton, SkeletonExtras, Connectivity

def orientation_matrix(q0, q1, q2, q3):
   # based on https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
   r11 = 2 * (q0 ** 2 + q1 ** 2) - 1
   r12 = 2 * (q1 * q2 - q0 * q3)
   r13 = 2 * (q1 * q3 + q0 * q2)
   r21 = 2 * (q1 * q2 + q0 * q3)
   r22 = 2 * (q0 ** 2 + q2 ** 2) - 1
   r23 = 2 * (q2 * q3 - q0 * q1)
   r31 = 2 * (q1 * q3 - q0 * q2)
   r32 = 2 * (q2 * q3 + q0 * q1)
   r33 = 2 * (q0 ** 2 + q3 ** 2) - 1
   return r11, r12, r13, r21, r22, r23, r31, r32, r33
def compute_relative_orientation(seg, cal):
   '''
   Calculating the relative orientation between two matrices. This is used for the initial normalization
   procedure using the standing calibration
   '''
   R_11 = np.array([])
   R_12 = np.array([])
   R_13 = np.array([])
   R_21 = np.array([])
   R_22 = np.array([])
   R_23 = np.array([])
   R_31 = np.array([])
   R_32 = np.array([])
   R_33 = np.array([])
   for i in range(seg.shape[0]):
       segment = np.asmatrix([
           [np.array(seg['o11'])[i], np.array(seg['o12'])[i], np.array(seg['o13'])[i]],
           [np.array(seg['o21'])[i], np.array(seg['o22'])[i], np.array(seg['o23'])[i]],
           [np.array(seg['o31'])[i], np.array(seg['o32'])[i], np.array(seg['o33'])[i]]
       ])
       segment_cal = np.asmatrix([
           [np.array(cal['o11'])[i], np.array(cal['o12'])[i], np.array(cal['o13'])[i]],
           [np.array(cal['o21'])[i], np.array(cal['o22'])[i], np.array(cal['o23'])[i]],
           [np.array(cal['o31'])[i], np.array(cal['o32'])[i], np.array(cal['o33'])[i]]
       ])
       # normalization
       r = np.matmul(segment, segment_cal.T)
       new_orientations = np.asarray(r).reshape(-1)
       R_11 = np.append(R_11, new_orientations[0])
       R_12 = np.append(R_12, new_orientations[1])
       R_13 = np.append(R_13, new_orientations[2])
       R_21 = np.append(R_21, new_orientations[3])
       R_22 = np.append(R_22, new_orientations[4])
       R_23 = np.append(R_23, new_orientations[5])
       R_31 = np.append(R_31, new_orientations[6])
       R_32 = np.append(R_32, new_orientations[7])
       R_33 = np.append(R_33, new_orientations[8])
   return R_11, R_12, R_13, R_21, R_22, R_23, R_31, R_32, R_33
def compute_joint_angle(df, child, parent):
   c = df[df['jointType'] == child]
   p = df[df['jointType'] == parent]
   ml = np.array([])
   ap = np.array([])
   v = np.array([])
   # Compute Rotation Matrix Components
   for i in range(c.shape[0]):
       segment = np.asmatrix([
           [np.array(c['n_o11'])[i], np.array(c['n_o12'])[i], np.array(c['n_o13'])[i]],
           [np.array(c['n_o21'])[i], np.array(c['n_o22'])[i], np.array(c['n_o23'])[i]],
           [np.array(c['n_o31'])[i], np.array(c['n_o32'])[i], np.array(c['n_o33'])[i]]
       ])
       reference_segment = np.asmatrix([
           [np.array(p['n_o11'])[i], np.array(p['n_o12'])[i], np.array(p['n_o13'])[i]],
           [np.array(p['n_o21'])[i], np.array(p['n_o22'])[i], np.array(p['n_o23'])[i]],
           [np.array(p['n_o31'])[i], np.array(p['n_o32'])[i], np.array(p['n_o33'])[i]]
       ])
       # transformation of segment to reference segment
       r = np.matmul(reference_segment.T, segment)
       # decomposition to Euler angles
       rotations = R.from_matrix(r).as_euler('xyz', degrees=True)
       ml = np.append(ml, rotations[0])
       ap = np.append(ap, rotations[1])
       v = np.append(v, rotations[2])
   return ml, ap, v
def resample_df(d, new_freq=30, method='linear'):
    # Resamples data at 30Hz unless otherwise specified
    joints_without_quats = [3, 15, 19, 21, 22, 23, 24]
    resampled_df = pd.DataFrame(
        columns=['# timestamp', 'jointType', 'orientation.X', 'orientation.Y', 'orientation.Z',
                 'orientation.W', 'position.X', 'position.Y', 'position.Z'])
    new_df = pd.DataFrame()
    for i in d['jointType'].unique():
        current_df = d.loc[d['jointType'] == i].copy()
        old_times = np.array(current_df['# timestamp'])
        new_times = np.arange(min(current_df['# timestamp']), max(current_df['# timestamp']), 1 / new_freq)
        o_x = np.array(current_df['orientation.X'])
        o_y = np.array(current_df['orientation.Y'])
        o_z = np.array(current_df['orientation.Z'])
        o_w = np.array(current_df['orientation.W'])
        p_x = np.array(current_df['position.X'])
        p_y = np.array(current_df['position.Y'])
        p_z = np.array(current_df['position.Z'])
        if i in joints_without_quats:
            orientation_x = np.repeat(0.0, len(new_times))
            orientation_y = np.repeat(0.0, len(new_times))
            orientation_z = np.repeat(0.0, len(new_times))
            orientation_w = np.repeat(0.0, len(new_times))
        else:
            if method == "linear":
                orientation_x = np.interp(new_times, old_times, o_x)
                orientation_y = np.interp(new_times, old_times, o_y)
                orientation_z = np.interp(new_times, old_times, o_z)
                orientation_w = np.interp(new_times, old_times, o_w)
            elif method == 'slerp':
                quats = []
                for t in range(len(old_times)):
                    quats.append([o_x[t], o_y[t], o_z[t], o_w[t]])
                # Create rotation object
                quats_object = R.from_quat(quats)
                # Spherical Linear Interpolation
                slerp = Slerp(np.array(current_df['# timestamp']), quats_object)
                interp_rots = slerp(new_times)
                new_quats = interp_rots.as_quat()
                # Create new orientation objects
                orientation_x = np.array([item[0] for item in new_quats])
                orientation_y = np.array([item[1] for item in new_quats])
                orientation_z = np.array([item[2] for item in new_quats])
                orientation_w = np.array([item[3] for item in new_quats])
            else:
                raise ValueError("Method must be either linear or spherical (slerp) interpolation.")
        position_x = signal.resample(p_x, num=int(max(current_df['# timestamp']) * new_freq))
        position_y = signal.resample(p_y, num=int(max(current_df['# timestamp']) * new_freq))
        position_z = signal.resample(p_z, num=int(max(current_df['# timestamp']) * new_freq))
        new_df['# timestamp'] = pd.Series(new_times)
        new_df['jointType'] = pd.Series(np.repeat(i, len(new_times)))
        new_df['orientation.X'] = pd.Series(orientation_x)
        new_df['orientation.Y'] = pd.Series(orientation_y)
        new_df['orientation.Z'] = pd.Series(orientation_z)
        new_df['orientation.W'] = pd.Series(orientation_w)
        new_df['position.X'] = pd.Series(position_x)
        new_df['position.Y'] = pd.Series(position_y)
        new_df['position.Z'] = pd.Series(position_z)
        resampled_df = resampled_df.append(new_df, ignore_index=True)
    return resampled_df
def smooth_rotations(o_x, o_y, o_z, o_w):
    o_x = np.array(o_x)
    o_y = np.array(o_y)
    o_z = np.array(o_z)
    o_w = np.array(o_w)
    trajNoisy = []
    for i in range(len(o_x)):
        trajNoisy.append([o_x[i], o_y[i], o_z[i], o_w[i]])
    trajNoisy = np.array(trajNoisy)
    # This code was adapted from https://ww2.mathworks.cn/help/nav/ug/lowpass-filter-orientation-using-quaternion-slerp.html
    # As explained in the link above, "The interpolation parameter to slerp is in the closed-interval [0,1], so the output of dist
    # must be re-normalized to this range. However, the full range of [0,1] for the interpolation parameter gives poor performance,
    # so it is limited to a smaller range hrange centered at hbias."
    hrange = 0.4
    hbias = 0.4
    low = max(min(hbias - (hrange / 2), 1), 0)
    high = max(min(hbias + (hrange / 2), 1), 0)
    hrangeLimited = high - low
    # initial filter state is the quaternion at frame 0
    y = trajNoisy[0]
    qout = []
    for i in range(1, len(trajNoisy)):
        x = trajNoisy[i]
        # x = mathutils.Quaternion(x)
        # y = mathutils.Quaternion(y)
        # d = x.rotation_difference(y).angle
        x = pyq.Quaternion(x)
        y = pyq.Quaternion(y)
        d = (x.conjugate * y).angle
        # Renormalize dist output to the range [low, high]
        hlpf = (d / np.pi) * hrangeLimited + low
        # y = y.slerp(x, hlpf)
        y = Quaternion.slerp(y, x, hlpf).elements
        qout.append(np.array(y))
    # because a frame of data is lost during this process, I've (arbitrarily) decided to append an extra quaternion at the end of the trial
    # that is identical to the n-1th frame. This keeps the length consistent (so there is no issues with merging later) and should not
    # negatively impact the data since the last frame is rarely of interest (and the data collector can decide to collect for a split second
    # after their trial of interest has completed to attenuate any of these "errors" that may propogate in the analyses)
    qout.append(qout[int(len(qout) - 1)])
    orientation_x = [item[0] for item in qout]
    orientation_y = [item[1] for item in qout]
    orientation_z = [item[2] for item in qout]
    orientation_w = [item[3] for item in qout]
    return orientation_x, orientation_y, orientation_z, orientation_w
def smooth_quaternions(d):
    for i in d['jointType'].unique():
        current_df = d.loc[d['jointType'] == i].copy()
        current_df['orientation.X'], current_df['orientation.Y'], current_df['orientation.Z'], current_df[
            'orientation.W'] = smooth_rotations(current_df['orientation.X'], current_df['orientation.Y'],
                                                 current_df['orientation.Z'], current_df['orientation.W'])
        d[d['jointType'] == i] = current_df
    return d
def compute_segment_angle(df, SEGMENT):
    s = df[df['jointType'] == SEGMENT]
    ml = np.array([])
    ap = np.array([])
    v = np.array([])
    # Compute Rotation Matrix Components
    for i in range(s.shape[0]):
        segment = np.asmatrix([
            [np.array(s['n_o11'])[i], np.array(s['n_o12'])[i], np.array(s['n_o13'])[i]],
            [np.array(s['n_o21'])[i], np.array(s['n_o22'])[i], np.array(s['n_o23'])[i]],
            [np.array(s['n_o31'])[i], np.array(s['n_o32'])[i], np.array(s['n_o33'])[i]]
        ])
        # decomposition to Euler angles
        rotations = R.from_matrix(segment).as_euler('xyz', degrees=True)
        ml = np.append(ml, rotations[0])
        ap = np.append(ap, rotations[1])
        v = np.append(v, rotations[2])
    return ml, ap, v

def compute_joint_angle_single(child_norm: np.ndarray, parent_norm: np.ndarray) -> np.ndarray:
    """
    Compute the relative orientation between a child joint and a parent joint.
    Both child_norm and parent_norm are 3x3 normalized orientation matrices.
    Returns the Euler angles (in degrees) using the 'xyz'convention.
    """
    # Relative rotation from parent to child
    rel_R = parent_norm.T @ child_norm
    return R.from_matrix(rel_R).as_euler('xyz', degrees=True)

def compute_segment_angle_single(norm: np.ndarray) -> float:
    """
    Compute the segment angle (using the first Euler angle from the 'xyz'decomposition)
    from a normalized orientation matrix.
    """
    return R.from_matrix(norm).as_euler('xyz', degrees=True)[0]


def process_record(df: pd.DataFrame) -> list:
    """
    Process a dataframe of raw Kinect data and return a list of Skeleton objects.
    Each row of df is assumed to have the following columns:
        "# timestamp", " jointType", " orientation.X", " orientation.Y", " orientation.Z",
        " orientation.W", " position.X", " position.Y", " position.Z"
    The function performs re-orientation, (optional) resampling and smoothing, computes
    the orientation matrix for each joint, normalizes each joint’s orientation using a
    calibration (here, the first frame is taken as calibration), computes joint angles,
    and builds a Skeleton object for each unique timestamp.
    """
    # --- Step 1. Reorient the local coordinate systems ---
    # (These rules were taken from your original code)
    df.columns = df.columns.str.strip()
    df_reoriented = df.copy()
    
    # For hips:
    # Joint 16 (HIP_RIGHT)
    mask = df_reoriented['jointType'] == 16
    df_reoriented.loc[mask, 'orientation.X'] = df.loc[mask, 'orientation.Z']
    df_reoriented.loc[mask, 'orientation.Y'] = df.loc[mask, 'orientation.X']
    df_reoriented.loc[mask, 'orientation.Z'] = df.loc[mask, 'orientation.Y']
    # Joint 12 (HIP_LEFT)
    mask = df_reoriented['jointType'] == 12
    df_reoriented.loc[mask, 'orientation.X'] = df.loc[mask, 'orientation.Z']
    df_reoriented.loc[mask, 'orientation.Y'] = df.loc[mask, 'orientation.X'] * -1
    df_reoriented.loc[mask, 'orientation.Z'] = df.loc[mask, 'orientation.Y'] * -1

    # For knees:
    # Joint 17 (KNEE_RIGHT)
    mask = df_reoriented['jointType'] == 17
    df_reoriented.loc[mask, 'orientation.X'] = df.loc[mask, 'orientation.X'] * -1
    df_reoriented.loc[mask, 'orientation.Y'] = df.loc[mask, 'orientation.Y'] * -1
    # Joint 13 (KNEE_LEFT)
    mask = df_reoriented['jointType'] == 13
    df_reoriented.loc[mask, 'orientation.Y'] = df.loc[mask, 'orientation.Y'] * -1
    df_reoriented.loc[mask, 'orientation.Z'] = df.loc[mask, 'orientation.Z'] * -1

    # For ankles:
    # Joint 18 (ANKLE_RIGHT)
    mask = df_reoriented['jointType'] == 18
    df_reoriented.loc[mask, 'orientation.X'] = df.loc[mask, 'orientation.X'] * -1
    df_reoriented.loc[mask, 'orientation.Y'] = df.loc[mask, 'orientation.Y'] * -1
    # Joint 14 (ANKLE_LEFT)
    mask = df_reoriented['jointType'] == 14
    df_reoriented.loc[mask, 'orientation.Y'] = df.loc[mask, 'orientation.Y'] * -1
    df_reoriented.loc[mask, 'orientation.Z'] = df.loc[mask, 'orientation.Z'] * -1

    # --- (Optional) Step 2. Resample and smooth rotations ---
    # You might call your resample_df() and smooth_quaternions() functions here.
    # For brevity, we assume that df_reoriented is already at the desired sampling rate and is smoothed.

    # --- Step 3. Compute the orientation matrix for each row ---
    # (We assume your orientation_matrix function is defined as shown in your code.)
    o11, o12, o13, o21, o22, o23, o31, o32, o33 = orientation_matrix(
        df_reoriented['orientation.W'],
        df_reoriented['orientation.X'],
        df_reoriented['orientation.Y'],
        df_reoriented['orientation.Z']
    )
    df_reoriented['o11'] = o11
    df_reoriented['o12'] = o12
    df_reoriented['o13'] = o13
    df_reoriented['o21'] = o21
    df_reoriented['o22'] = o22
    df_reoriented['o23'] = o23
    df_reoriented['o31'] = o31
    df_reoriented['o32'] = o32
    df_reoriented['o33'] = o33

    # --- Step 4. Use the first frame as the calibration pose ---
    # Group by timestamp (each frame is assumed to be identified by a unique "# timestamp")
    grouped_frames = df_reoriented.groupby('# timestamp')
    first_timestamp = df_reoriented['# timestamp'].min()
    calibration_frame = grouped_frames.get_group(first_timestamp)
    # Build a dict mapping joint type to its calibration orientation matrix (3x3)
    calibration_orientations = {}
    for _, row in calibration_frame.iterrows():
        joint = int(row['jointType'])
        calibration_orientations[joint] = np.array([
            [row['o11'], row['o12'], row['o13']],
            [row['o21'], row['o22'], row['o23']],
            [row['o31'], row['o32'], row['o33']]
        ])

    # --- Step 5. Build Skeleton objects for each frame ---
    skeletons = []

    # Define a helper to get the normalized orientation for a given row:
    def get_normalized_orientation(row):
        joint = int(row['jointType'])
        R_current = np.array([
            [row['o11'], row['o12'], row['o13']],
            [row['o21'], row['o22'], row['o23']],
            [row['o31'], row['o32'], row['o33']]
        ])
        # If no calibration exists for this joint, default to the identity.
        R_cal = calibration_orientations.get(joint, np.eye(3))
        return R_current @ R_cal.T

    # Loop over each frame (each unique timestamp)
    for timestamp, frame_df in grouped_frames:
        keypoints = {}
        norm_orientations = {}  # to store normalized 3x3 matrices for extra computation

        # For each joint (row) in the current frame:
        for _, row in frame_df.iterrows():
            joint_int = int(row['jointType'])
            # Create a Keypoint using your KeypointType enum:
            try:
                kp = Keypoint(
                    keypoint_type=KeypointType(joint_int),
                    x=row['position.X'],
                    y=row['position.Y'],
                    z=row['position.Z'],
                    connections=Connectivity.get(KeypointType(joint_int), [])
                )
            except ValueError:
                continue
            keypoints[KeypointType(joint_int)] = kp

            # Compute the normalized orientation matrix for this joint.
            norm_orientations[joint_int] = get_normalized_orientation(row)

        # --- Step 6. Compute extra joint angles for this frame ---
        # (Here we follow the sign conventions you used in your final dataframe.)
        # Right hip (parent: HIP_RIGHT=16, child: KNEE_RIGHT=17)
        if 16 in norm_orientations and 17 in norm_orientations:
            r_hip_angles = compute_joint_angle_single(
                norm_orientations[17],
                norm_orientations[16]
            )
            r_hipFlexion = r_hip_angles[0]
            r_hipAbduction = -r_hip_angles[1]
            r_hipV = -r_hip_angles[2]
        else:
            r_hipFlexion = r_hipAbduction = r_hipV = 0.0

        # Left hip (parent: HIP_LEFT=12, child: KNEE_LEFT=13)
        if 12 in norm_orientations and 13 in norm_orientations:
            l_hip_angles = compute_joint_angle_single(
                norm_orientations[13],
                norm_orientations[12]
            )
            l_hipFlexion = -l_hip_angles[0]
            l_hipAbduction = l_hip_angles[1]
            l_hipV = -l_hip_angles[2]
        else:
            l_hipFlexion = l_hipAbduction = l_hipV = 0.0

        # Right knee (parent: KNEE_RIGHT=17, child: ANKLE_RIGHT=18)
        if 17 in norm_orientations and 18 in norm_orientations:
            r_knee_angles = compute_joint_angle_single(
                norm_orientations[18],
                norm_orientations[17]
            )
            r_kneeFlexion = -r_knee_angles[0]
            r_kneeAdduction = r_knee_angles[1]
            r_kneeV = -r_knee_angles[2]
        else:
            r_kneeFlexion = r_kneeAdduction = r_kneeV = 0.0

        # Left knee (parent: KNEE_LEFT=13, child: ANKLE_LEFT=14)
        if 13 in norm_orientations and 14 in norm_orientations:
            l_knee_angles = compute_joint_angle_single(
                norm_orientations[14],
                norm_orientations[13]
            )
            l_kneeFlexion = l_knee_angles[0]
            l_kneeAdduction = -l_knee_angles[1]
            l_kneeV = l_knee_angles[2]
        else:
            l_kneeFlexion = l_kneeAdduction = l_kneeV = 0.0

        # Pelvis rotation (using HIP_RIGHT=16 as representative)
        if 16 in norm_orientations:
            pelvis_rotation = compute_segment_angle_single(norm_orientations[16])
        else:
            pelvis_rotation = 0.0

        # Right thigh rotation (using KNEE_RIGHT=17)
        if 17 in norm_orientations:
            r_thigh_rotation = compute_segment_angle_single(norm_orientations[17])
        else:
            r_thigh_rotation = 0.0

        # Left thigh rotation (using KNEE_LEFT=13; note sign correction)
        if 13 in norm_orientations:
            l_thigh_rotation = -compute_segment_angle_single(norm_orientations[13])
        else:
            l_thigh_rotation = 0.0

        # Right shank rotation (using ANKLE_RIGHT=18)
        if 18 in norm_orientations:
            r_shank_rotation = compute_segment_angle_single(norm_orientations[18])
        else:
            r_shank_rotation = 0.0

        # Left shank rotation (using ANKLE_LEFT=14; note sign correction)
        if 14 in norm_orientations:
            l_shank_rotation = -compute_segment_angle_single(norm_orientations[14])
        else:
            l_shank_rotation = 0.0

        extras = SkeletonExtras(
            r_hipFlexion=r_hipFlexion,
            l_hipFlexion=l_hipFlexion,
            r_hipAbduction=r_hipAbduction,
            l_hipAbduction=l_hipAbduction,
            r_hipV=r_hipV,
            l_hipV=l_hipV,
            r_kneeFlexion=r_kneeFlexion,
            l_kneeFlexion=l_kneeFlexion,
            r_kneeAdduction=r_kneeAdduction,
            l_kneeAdduction=l_kneeAdduction,
            r_kneeV=r_kneeV,
            l_kneeV=l_kneeV,
            pelvis_rotation=pelvis_rotation,
            r_thigh_rotation=r_thigh_rotation,
            l_thigh_rotation=l_thigh_rotation,
            r_shank_rotation=r_shank_rotation,
            l_shank_rotation=l_shank_rotation
        )

        # Create the Skeleton object for the frame
        skeleton = Skeleton(timestamp=timestamp, keypoints=keypoints, extras=extras)
        skeletons.append(skeleton)
    return skeletons
        
def tail(filename: str, n_lines: int, jump_bytes: int = 4096):
    with open(filename, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        lines_found= []
        remaining_bytes = b""
        pointer_location = file_size
        
        while pointer_location > 0 and len(lines_found) < n_lines:
            jump = min(pointer_location, jump_bytes)
            pointer_location -= jump
            f.seek(pointer_location)
            chunk = f.read(jump) + remaining_bytes
            lines = chunk.split(b"\n")
            remaining_bytes = lines[0]
            lines_found = lines[1:] + lines_found
        lines_found = lines_found[-n_lines:]
    
    return [line.decode('utf-8') for line in lines_found]

def get_last_record(csv_file_path: str) -> Optional[Skeleton]:
    with open(csv_file_path, 'r') as f:
        header = f.readline()
    header = ",".join(map(str.strip, header.split(","))) + "\n"
    
    record_lines = tail(csv_file_path, 50)
    if len(record_lines) < 25:
        return None
    csv_string = header + "".join(line + "\n" for line in record_lines)
    df = pd.read_csv(StringIO(csv_string))
    candidate_frame = None
    total_rows = len(df)
    for start in range(total_rows - 25, -1, -1):
        candidate = df.iloc[start:start + 25]
        try:
            # Ensure that jointType values are integers.
            joint_types = candidate['jointType'].astype(int).tolist()
        except Exception:
            joint_types = candidate['jointType'].tolist()
        if joint_types == list(range(25)):
            if candidate.dtypes.eq('object').any():
                continue
            candidate_frame = candidate
            break  # we found the most recent complete frame
    
    if candidate_frame is None:
        return None
    return process_record(candidate_frame)[-1]
    