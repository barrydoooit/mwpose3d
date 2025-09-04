from statistics import median
from typing import Any, Dict, List, Tuple
import numpy as np
from mwpose3d.runner.hooks.pre_inference_hook import PreInferenceHook
from mwpose3d.runner.inference_engine import InferenceEngine

from ..base import BaseTransform, OnlineEnabled
from mwpose3d.registry import TRANSFORMS



def _get_joint_xyz(skel_frame: np.ndarray, joint_idx: int, joint_map: Dict[int, int]) -> np.ndarray:
    b = 3 * joint_map.get(joint_idx)
    v = skel_frame[b:b+3]
    return v
    

def _vec_xy(v: np.ndarray) -> np.ndarray:
    return np.array([v[0], v[1]], dtype=np.float32)


def _unit_xy(vxy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(vxy)
    return vxy / (n + eps)


def _clamp_norm_xy(vxy: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return vxy
    n = float(np.linalg.norm(vxy))
    if n <= max_norm:
        return vxy
    return vxy * (max_norm / (n + 1e-12))


def _median_safe(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size == 0:
        return np.nan
    return float(np.median(arr))


def _angle_deg(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < eps or nb < eps: 
        return 180.0
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))

@OnlineEnabled
@TRANSFORMS.register_module()
class TrackingCentroidCalibration(BaseTransform):
    def __init__(self,
                 method: str = "identity",
                 method_cfg: Dict[str, Any] | None = None,
                 online_mode: bool = False):
        super().__init__(online_mode)
        self.method_cfg = dict(method_cfg or {})

        # Simple if/else assignment instead of registry
        if method == "identity":
            self.calib = self._calib_identity
        elif method == "suppress":
            self.calib = self._calib_suppress
        else:
            raise ValueError(f"Unknown calibration method '{method}'")
        
    def transform(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            preds: Tuple[np.ndarray, ...] = input[PreInferenceHook.PREINFERENCE_RESULTS]
        except KeyError:
            return input
        input["track_centroid"] = self.calib(input, input["track_centroid"], preds, self.method_cfg)
        return input

    def transform_online(self, input: Dict[str, Any]) -> Dict[str, Any]:
        track_centroid = input["track_centroid"][:-1]  # Exclude current frame
        skel_pred_history: List[np.ndarray] = InferenceEngine.get_current_instance().get_pred_history()
        if len(skel_pred_history) < len(track_centroid):
            return input
        input["track_centroid"] = tuple(
            list(self.calib(input, track_centroid, tuple(skel_pred_history), self.method_cfg)) + [self._estim_missing_centroid(track_centroid)])
        return input

    def _estim_missing_centroid(self, centroids: Tuple[np.ndarray, ...], n_frames: int = 5) -> np.ndarray:
        if n_frames is not None:
            centroids = centroids[-n_frames:]
        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]

        x_last = median(xs)
        y_last = median(ys)
        dxs = [xs[i] - xs[i-1] for i in range(1, len(xs))]
        dys = [ys[i] - ys[i-1] for i in range(1, len(ys))]
        vx, vy = median(dxs), median(dys)
        return np.array([x_last + vx, y_last + vy, 0], dtype=np.float32)

    
    @staticmethod
    def _calib_identity(input: Dict[str, Any],
                        centroids: Tuple[np.ndarray, ...],
                        preds: Tuple[np.ndarray, ...],
                        method_cfg: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
        return centroids

    @staticmethod
    def _calib_suppress_get_K(centroids: Tuple[np.ndarray, ...], skel_frames: Tuple[np.ndarray, ...], 
                              joint_map: Dict[int, int], spine_idx: int, l_wrist_idx:int, r_wrist_idx: int,
                              range_gate_on_trh: float, range_gate_off_trh: float,
                              kappa: float, tau0: float, vector_pair_angle_max_degdiff: float,
                              K_default: float, K_min: float, K_max: float) -> float:
        ratios: List[float] = []
        gateL_prev = False
        gateR_prev = False

        T = min(len(centroids), len(skel_frames))
        for t in range(T):
            centroid = centroids[t]
            skel_frame = skel_frames[t]

            J_S = _get_joint_xyz(skel_frame, spine_idx, joint_map)
            J_LW = _get_joint_xyz(skel_frame, l_wrist_idx, joint_map)
            J_RW = _get_joint_xyz(skel_frame, r_wrist_idx, joint_map)
            wL = _vec_xy(J_LW - J_S)
            wR = _vec_xy(J_RW - J_S)
            rL = float(np.linalg.norm(wL))
            rR = float(np.linalg.norm(wR))

            gateL = (rL > range_gate_on_trh) or (gateL_prev and rL > range_gate_off_trh)
            gateR = (rR > range_gate_on_trh) or (gateR_prev and rR > range_gate_off_trh)
            gate = gateL or gateR

            if not gate:
                pass
            else:
                if gateL and gateR:
                    w_vec = 0.5 * (wL + wR)
                elif gateL:
                    w_vec = wL
                else:
                    w_vec = wR
                
                w_norm = float(np.linalg.norm(w_vec))
                if w_norm < 0.2:
                    gateL_prev, gateR_prev = gateL, gateR
                    continue
                w_vec_u = _unit_xy(w_vec)

                c_vec = _vec_xy(centroid - J_S)
                c_vec_prj = float(np.dot(c_vec, w_vec_u))
                c_vec_trh = kappa * w_norm + tau0
                #CASE 1:
                if c_vec_prj >= c_vec_trh:
                    if _angle_deg(c_vec, w_vec) < vector_pair_angle_max_degdiff:
                        ratios.append(c_vec_prj / (w_norm + 1e-6))
            gateL_prev, gateR_prev = gateL, gateR
        if ratios:
            K_hat = _median_safe(ratios)
            if np.isfinite(K_hat):
                return float(np.clip(K_hat, K_min, K_max))
        return K_default

    @staticmethod
    def _calib_suppress(input: Dict[str, Any],
                        centroids: Tuple[np.ndarray, ...],
                        preds: Tuple[np.ndarray, ...],
                        method_cfg: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
        keypoints_involved = method_cfg.get("keypoints_involved")
        joint_map = {jid: i for i, jid in enumerate(keypoints_involved)}
        spine_idx, l_shoulder_idx, r_shoulder_idx, l_wrist_idx, r_wrist_idx = \
            tuple(method_cfg.get(k) for k in ["spine_idx", "l_shoulder_idx", "r_shoulder_idx", "l_wrist_idx", "r_wrist_idx"])
        
        range_gate_on_trh = float(method_cfg.get("range_gate_on_trh", 0.5))
        range_gate_off_trh = float(method_cfg.get("range_gate_off_trh", 0.4))
        K_default = float(method_cfg.get("K_default", 0.5))
        K_min = float(method_cfg.get("K_min", 0.2))
        K_max = float(method_cfg.get("K_max", 0.8))

        kappa, tau0 = float(method_cfg.get("kappa", 0.35)), float(method_cfg.get("tau0", 0.01))
        case2_suppress_ratio = float(method_cfg.get("case2_suppress_ratio", 0.75))
        case3_suppress_ratio = float(method_cfg.get("case3_suppress_ratio", 0.95))
        case3_calib_strength = float(method_cfg.get("case3_calib_strength", 0.4))
        max_translate = float(method_cfg.get("max_translate", 0.3))
        
        vector_pair_angle_max_degdiff = float(method_cfg.get("vector_pair_angle_max_degdiff", 15.0))

        K = TrackingCentroidCalibration._calib_suppress_get_K(
            centroids, preds, joint_map, spine_idx, l_wrist_idx, r_wrist_idx,
            range_gate_on_trh, range_gate_off_trh,
            kappa, tau0, vector_pair_angle_max_degdiff,
            K_default, K_min, K_max)

        out: List[np.ndarray] = []
        gateL_prev = False
        gateR_prev = False
        centroid_prev = None

        T = min(len(centroids), len(preds))
        for t in range(T):
            centroid = centroids[t]
            skel_frame = preds[t]

            J_S, J_LS, J_RS, J_LW, J_RW = (
                _get_joint_xyz(skel_frame, idx, joint_map)
                for idx in (spine_idx, l_shoulder_idx, r_shoulder_idx, l_wrist_idx, r_wrist_idx)
            )
            wL = _vec_xy(J_LW - J_S)
            wR = _vec_xy(J_RW - J_S)
            rL = float(np.linalg.norm(wL))
            rR = float(np.linalg.norm(wR))

            gateL = (rL > range_gate_on_trh) or (gateL_prev and rL > range_gate_off_trh)
            gateR = (rR > range_gate_on_trh) or (gateR_prev and rR > range_gate_off_trh)
            gate = gateL or gateR

            if centroid_prev is None:
                centroid_prev =  centroid
            
            translation_xy = np.zeros(2, dtype=np.float32)
            suppress = 0.0
            
            if gate:
                segment_start = not (gateL_prev or gateR_prev)
                if gateL and gateR:
                    w_vec = 0.5 * (wL + wR)
                elif gateL:
                    w_vec = wL
                else:
                    w_vec = wR
                
                w_norm = float(np.linalg.norm(w_vec))
                w_vec_u = _unit_xy(w_vec)

                c_vec = _vec_xy(centroid - J_S)
                c_vec_prj = float(np.dot(c_vec, w_vec_u))
                shift = K * w_norm
                c_vec_trh = kappa * w_norm + tau0

                #CASE 1:
                if c_vec_prj >= c_vec_trh:
                    suppress = 0.0
                elif abs(c_vec_prj) < c_vec_trh:
                    suppress = case2_suppress_ratio
                    translation_xy = _clamp_norm_xy(-K * w_vec, max_translate)
                else:
                    suppress = case3_suppress_ratio
                    dir_c = _unit_xy(c_vec)
                    translation_xy = _clamp_norm_xy(case3_calib_strength * np.linalg.norm(c_vec) * dir_c, max_translate)
                
                centroid = centroid + np.array([translation_xy[0], translation_xy[1], 0], dtype=np.float32)
            out.append(centroid)
            gateL_prev, gateR_prev = gateL, gateR
        return tuple(out)




