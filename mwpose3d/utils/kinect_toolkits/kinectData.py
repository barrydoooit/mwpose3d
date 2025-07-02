import csv
import io
from enum import Enum
from typing import List, Dict, Literal, Optional, Sequence, Tuple

import pandas as pd



class KeypointType(Enum):
    SPINE_BASE = 0
    SPINE_MID = 1
    NECK = 2
    HEAD = 3
    SHOULDER_LEFT = 4
    ELBOW_LEFT = 5
    WRIST_LEFT = 6
    HAND_LEFT = 7
    SHOULDER_RIGHT = 8
    ELBOW_RIGHT = 9
    WRIST_RIGHT = 10
    HAND_RIGHT = 11
    HIP_LEFT = 12
    KNEE_LEFT = 13
    ANKLE_LEFT = 14
    FOOT_LEFT = 15
    HIP_RIGHT = 16
    KNEE_RIGHT = 17
    ANKLE_RIGHT = 18
    FOOT_RIGHT = 19
    SPINE_SHOULDER = 20
    HAND_TIP_LEFT = 21
    THUMB_LEFT = 22
    HAND_TIP_RIGHT = 23
    THUMB_RIGHT = 24

USED_KEYPOINTS = tuple(range(20))
ALL_KEYPOINTS = tuple(range(25))

LEFT_KEYPOINTS = (KeypointType.SHOULDER_LEFT, KeypointType.ELBOW_LEFT, KeypointType.WRIST_LEFT, KeypointType.HAND_LEFT,
                  KeypointType.HIP_LEFT, KeypointType.KNEE_LEFT, KeypointType.ANKLE_LEFT, KeypointType.FOOT_LEFT)
RIGHT_KEYPOINTS = (KeypointType.SHOULDER_RIGHT, KeypointType.ELBOW_RIGHT, KeypointType.WRIST_RIGHT, KeypointType.HAND_RIGHT,
                   KeypointType.HIP_RIGHT, KeypointType.KNEE_RIGHT, KeypointType.ANKLE_RIGHT, KeypointType.FOOT_RIGHT)

class Keypoint:
    def __init__(self, keypoint_type: KeypointType, x: float, y: float, z: float, 
                 connections: Optional[List[KeypointType]] = None):
        """
        :param keypoint_type: Enum value representing the joint type.
        :param x: X coordinate.
        :param y: Y coordinate.
        :param z: Z coordinate.
        :param connections: A list of KeypointType values that this keypoint connects to.
        """
        self.keypoint_type = keypoint_type
        self.x = x
        self.y = y
        self.z = z
        self.connections = connections if connections is not None else []

class SkeletonExtras:
    def __init__(self, 
                 r_hipFlexion: float, l_hipFlexion: float,
                 r_hipAbduction: float, l_hipAbduction: float,
                 r_hipV: float, l_hipV: float,
                 r_kneeFlexion: float, l_kneeFlexion: float,
                 r_kneeAdduction: float, l_kneeAdduction: float,
                 r_kneeV: float, l_kneeV: float,
                 pelvis_rotation: float,
                 r_thigh_rotation: float, l_thigh_rotation: float,
                 r_shank_rotation: float, l_shank_rotation: float):
        self.r_hipFlexion = r_hipFlexion
        self.l_hipFlexion = l_hipFlexion
        self.r_hipAbduction = r_hipAbduction
        self.l_hipAbduction = l_hipAbduction
        self.r_hipV = r_hipV
        self.l_hipV = l_hipV
        self.r_kneeFlexion = r_kneeFlexion
        self.l_kneeFlexion = l_kneeFlexion
        self.r_kneeAdduction = r_kneeAdduction
        self.l_kneeAdduction = l_kneeAdduction
        self.r_kneeV = r_kneeV
        self.l_kneeV = l_kneeV
        self.pelvis_rotation = pelvis_rotation
        self.r_thigh_rotation = r_thigh_rotation
        self.l_thigh_rotation = l_thigh_rotation
        self.r_shank_rotation = r_shank_rotation
        self.l_shank_rotation = l_shank_rotation

class Skeleton:
    def __init__(self, timestamp: float,
                 unix_ms: int,
                 keypoints: Dict[KeypointType, Keypoint],
                 extras: Optional[SkeletonExtras] = None,
                 used_points: List[int] = USED_KEYPOINTS):
        self.timestamp = timestamp
        self.unix_ms = unix_ms
        self.keypoints = keypoints
        self.extras = extras
        self.used_points = used_points
    
    def transpose_(self, *order: int, with_extra: bool = False) -> 'Skeleton':
        if len(order) == 1 and isinstance(order[0], Sequence):
            order = order[0],
        if len(order) != 3 or set(order) != {0, 1, 2}:
            raise ValueError("Order must be a sequence of three unique indices (0, 1, 2).")
        for kp in self.keypoints.values():
            coords = (kp.x, kp.y, kp.z)
            kp.x, kp.y, kp.z = coords[order[0]], coords[order[1]], coords[order[2]]
        
        if self.extras and with_extra:
            raise NotImplementedError("Transposing extras is not implemented.")

        return self

    def flatten(self) -> Tuple[list[str], list[float]]:
        headers = ['timestamp', 'unix_ms'] + \
                    [item for kp in self.used_points for item in \
                     (f'{KeypointType(kp).name.lower()}_x', f'{KeypointType(kp).name.lower()}_y', f'{KeypointType(kp).name.lower()}_z')]
        flat_data = []
        for kp_type_val in self.used_points:
            kp_type = KeypointType(kp_type_val)
            kp = self.keypoints.get(kp_type)
            if kp:
                flat_data.extend([kp.x, kp.y, kp.z])
            else:
                flat_data.extend([0.0, 0.0, 0.0])
        return headers, [self.timestamp, self.unix_ms] + flat_data

    @classmethod
    def from_dataframe(cls, row: pd.Series, used_points: List[int] = None, extras: Optional[SkeletonExtras] = None):
        kps = {}
        used_points = used_points or USED_KEYPOINTS
        for kp_type_val in used_points:
            kp_type = KeypointType(kp_type_val)
            kp_type_name_lower = kp_type.name.lower()
            x, y, z = row[f'{kp_type_name_lower}_x'], row[f'{kp_type_name_lower}_y'], row[f'{kp_type_name_lower}_z']
            connections = Connectivity.get(kp_type, [])
            kps[kp_type] = Keypoint(kp_type, x, y, z, connections)
        return Skeleton(row['timestamp'], row['unix_ms'], kps, used_points=used_points, extras=extras)

    @classmethod
    def from_sequence(cls, sequence: list[float], used_points: List[int] = None, order: Literal['xyz', 'xzy'] = 'xyz',
                      timestamp: float = 0.0, unix_ms: int = 0, extras: Optional[SkeletonExtras] = None):
        kps = {}
        swap_order = order == 'xzy'
        used_points = used_points or USED_KEYPOINTS
        for i, kp_type_val in enumerate(used_points):
            kp_type = KeypointType(kp_type_val)
            x, y, z = sequence[i*3:i*3+3]
            if swap_order:
                y, z = z, y
            connections = Connectivity.get(kp_type, [])
            kps[kp_type] = Keypoint(kp_type, x, y, z, connections)
        return Skeleton(timestamp, unix_ms, kps, used_points=used_points, extras=extras)
    
Connectivity = {
    KeypointType.SPINE_BASE: [KeypointType.SPINE_MID],
    KeypointType.SPINE_MID: [KeypointType.SPINE_BASE, KeypointType.NECK],
    KeypointType.NECK: [KeypointType.SPINE_MID, KeypointType.HEAD, KeypointType.SHOULDER_LEFT, KeypointType.SHOULDER_RIGHT],
    KeypointType.HEAD: [KeypointType.NECK],
    KeypointType.SHOULDER_LEFT: [KeypointType.NECK, KeypointType.ELBOW_LEFT],
    KeypointType.ELBOW_LEFT: [KeypointType.SHOULDER_LEFT, KeypointType.WRIST_LEFT],
    KeypointType.WRIST_LEFT: [KeypointType.ELBOW_LEFT, KeypointType.HAND_LEFT],
    KeypointType.HAND_LEFT: [KeypointType.WRIST_LEFT],
    KeypointType.SHOULDER_RIGHT: [KeypointType.NECK, KeypointType.ELBOW_RIGHT],
    KeypointType.ELBOW_RIGHT: [KeypointType.SHOULDER_RIGHT, KeypointType.WRIST_RIGHT],
    KeypointType.WRIST_RIGHT: [KeypointType.ELBOW_RIGHT, KeypointType.HAND_RIGHT],
    KeypointType.HAND_RIGHT: [KeypointType.WRIST_RIGHT],
    KeypointType.HIP_LEFT: [KeypointType.KNEE_LEFT],
    KeypointType.KNEE_LEFT: [KeypointType.HIP_LEFT, KeypointType.ANKLE_LEFT],
    KeypointType.ANKLE_LEFT: [KeypointType.KNEE_LEFT, KeypointType.FOOT_LEFT],
    KeypointType.FOOT_LEFT: [KeypointType.ANKLE_LEFT],
    KeypointType.HIP_RIGHT: [KeypointType.KNEE_RIGHT],
    KeypointType.KNEE_RIGHT: [KeypointType.HIP_RIGHT, KeypointType.ANKLE_RIGHT],
    KeypointType.ANKLE_RIGHT: [KeypointType.KNEE_RIGHT, KeypointType.FOOT_RIGHT],
    KeypointType.FOOT_RIGHT: [KeypointType.ANKLE_RIGHT],
}

ConnectivityUni = {key: [neighbor for neighbor in neighbors if key.value < neighbor.value]
                   for key, neighbors in Connectivity.items()
                   if any(key.value < neighbor.value for neighbor in neighbors)}
ConnectivityVal = {key.value: [neighbor.value for neighbor in neighbors]
                     for key, neighbors in Connectivity.items()}
ConnectivityValUni = {key.value: [neighbor.value for neighbor in neighbors]
                     for key, neighbors in ConnectivityUni.items()}
    