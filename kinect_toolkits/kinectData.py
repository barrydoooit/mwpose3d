import csv
import io
from enum import Enum
from typing import List, Dict, Optional



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

USED_KEYPOINTS = tuple(range(20))
ALL_KEYPOINTS = tuple(range(25))


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
                 extras: Optional[SkeletonExtras] = None):
        self.timestamp = timestamp
        self.unix_ms = unix_ms
        self.keypoints = keypoints
        self.extras = extras


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