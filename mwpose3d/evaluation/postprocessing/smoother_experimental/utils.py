import torch
from dataclasses import dataclass

# ---------- small torch-only helpers ----------
def _norm(v, eps=1e-8): return torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
def _unit(v, eps=1e-8): return v / _norm(v, eps)

def _axis_angle_from_negz(d):
    d = _unit(d)
    ref = torch.tensor([0.0, 0.0, -1.0], device=d.device, dtype=d.dtype)
    c = (ref * d).sum().clamp(-1, 1)
    angle = torch.arccos(c)
    axis = torch.cross(ref, d)
    an = _norm(axis)[..., 0]
    axis = torch.where((an > 1e-8).unsqueeze(-1), axis / an.unsqueeze(-1), torch.zeros_like(axis))
    return axis * angle

def _rotate_negz(theta_vec):
    ref = torch.tensor([0.0, 0.0, -1.0], device=theta_vec.device, dtype=theta_vec.dtype)
    ang = _norm(theta_vec)[..., 0]
    if ang < 1e-8: return ref.clone()
    axis = theta_vec / ang
    ca = torch.cos(ang); sa = torch.sin(ang)
    dot = (axis * ref).sum()
    cross = torch.cross(axis, ref)
    return ref * ca + cross * sa + axis * dot * (1 - ca)

# ---------- Lockable EMAs (with peek/update and frozen output when locked) ----------
class EMA1:
    def __init__(self, lam, return_locked_output=True):
        self.lam = lam
        self.return_locked_output = return_locked_output
        self.y = None               # running EMA
        self.locked = False
        self.lock_value = None      # frozen output while locked
    # DO NOT update when deciding; just read previous EMA
    def peek(self):
        return self.y
    # Update EMA state AFTER decisions
    def update(self, x):
        self.y = x if self.y is None else (1 - self.lam) * self.y + self.lam * x
        return self.y
    # For convenience (not used in lock decisions)
    def step(self, x):
        self.update(x)
        return self.output()
    def set_locked(self, locked: bool):
        if locked and not self.locked:
            self.lock_value = None if self.y is None else self.y.clone().detach()
        self.locked = locked
    def output(self):
        if self.return_locked_output and self.locked and self.lock_value is not None:
            return self.lock_value
        return self.y

class EMA3:
    def __init__(self, lam, return_locked_output=True):
        self.lam = lam
        self.return_locked_output = return_locked_output
        self.y = None
        self.locked = False
        self.lock_value = None
    def peek(self):
        return self.y
    def update(self, x3):
        self.y = x3 if self.y is None else (1 - self.lam) * self.y + self.lam * x3
        return self.y
    def step(self, x3):
        self.update(x3)
        return self.output()
    def set_locked(self, locked: bool):
        if locked and not self.locked:
            self.lock_value = None if self.y is None else self.y.clone().detach()
        self.locked = locked
    def output(self):
        if self.return_locked_output and self.locked and self.lock_value is not None:
            return self.lock_value
        return self.y

# ---------- Spine EMA lock counter ----------
class EmaLock:
    def __init__(self, radius, k_in=2, k_out=2):
        self.radius, self.k_in, self.k_out = radius, k_in, k_out
        self.locked = False
        self._in_cnt = 0; self._out_cnt = 0
    def update(self, inside):
        if self.locked:
            if inside: self._out_cnt = 0
            else:
                self._out_cnt += 1
                if self._out_cnt >= self.k_out:
                    self.locked = False; self._in_cnt = 0; self._out_cnt = 0
        else:
            if inside:
                self._in_cnt += 1
                if self._in_cnt >= self.k_in:
                    self.locked = True; self._in_cnt = 0; self._out_cnt = 0
            else:
                self._in_cnt = 0
        return self.locked

# ---------- Optional CV for spine position ----------
class CV3:
    def __init__(self, dt, violate_tol, restart_frames=2):
        self.dt, self.tol, self.restart_frames = dt, violate_tol, restart_frames
        self.p = None; self.v = None
        self.prev_raw = None; self.prev_res = None
        self.violate_cnt = 0; self.active = False
    def cancel(self): self.active = False; self.prev_res = None; self.violate_cnt = 0
    def start_at(self, raw):
        self.v = (raw - self.prev_raw) / self.dt if self.prev_raw is not None else torch.zeros_like(raw)
        self.p = raw.clone(); self.active = True; self.prev_res = None; self.violate_cnt = 0
    def step(self, raw):
        if not self.active: self.prev_raw = raw.clone(); return raw
        p_pred = self.p + self.v * self.dt
        res = raw - p_pred
        if self.prev_res is not None:
            same_dir = (res * self.prev_res).sum() > 0
            self.violate_cnt = self.violate_cnt + 1 if (same_dir and torch.linalg.norm(res) > self.tol) else 0
        self.prev_res = res.clone()
        if self.violate_cnt >= self.restart_frames:
            self.start_at(raw); out = raw
        else:
            self.p = p_pred; out = p_pred
        self.prev_raw = raw.clone()
        return out

# ---------- Angle EMA lock band (uses peek-then-update) ----------
class AngleEmaLock:
    """
    Keep EMA of axis-angle. Decide 'inside' using the **previous EMA** (peek),
    then update EMA, then update lock, and freeze output when locked.
    """
    def __init__(self, ema: EMA3, tol_rad, unlock_out_frames=2, lock_in_frames=1, follow_raw_when_unlocked=True):
        self.ema = ema
        self.tol = tol_rad
        self.unlock_out_frames = unlock_out_frames
        self.lock_in_frames = lock_in_frames
        self.follow_raw_when_unlocked = follow_raw_when_unlocked
        self.out_cnt = 0; self.in_cnt = 0
        self._initialized = False

    def step(self, th_raw):
        # First-time init: seed EMA with raw, lock immediately (frozen)
        if not self._initialized:
            self.ema.update(th_raw)
            self.ema.set_locked(True)
            self._initialized = True
            return self.ema.output(), True

        # 1) DECIDE using PREVIOUS EMA (no update yet)
        th_ema_prev = self.ema.peek()
        inside = torch.linalg.norm(th_raw - th_ema_prev) <= self.tol

        # 2) UPDATE COUNTERS / LOCK STATE (but do not freeze value yet)
        was_locked = self.ema.locked
        if self.ema.locked:
            if inside:
                self.out_cnt = 0
            else:
                self.out_cnt += 1
                if self.out_cnt >= self.unlock_out_frames:
                    self.ema.set_locked(False)  # unlock (output will no longer be frozen)
                    self.out_cnt = 0
        else:
            if inside:
                self.in_cnt += 1
                if self.in_cnt >= self.lock_in_frames:
                    # We'll lock AFTER we update EMA (step 3), so freeze to the *updated* EMA
                    self.in_cnt = 0
                    # mark that we need to lock after update
                    will_lock = True
                else:
                    will_lock = False
            else:
                self.in_cnt = 0
                will_lock = False

        # 3) NOW UPDATE THE EMA WITH CURRENT SAMPLE
        self.ema.update(th_raw)

        # 4) APPLY PENDING LOCK (freeze to updated EMA)
        if not self.ema.locked:
            # If we were unlocked and met lock condition, freeze now
            if 'will_lock' in locals() and will_lock:
                self.ema.set_locked(True)

        # 5) OUTPUT
        if self.ema.locked:
            return self.ema.output(), True
        else:
            return (th_raw if self.follow_raw_when_unlocked else self.ema.y), False

# ---------- Config ----------
@dataclass
class PoseSmootherConfig:
    dt: float
    # toggles
    use_spine_cv: bool = False
    follow_raw_angle_when_unlocked: bool = False
    # EMA rates
    shoulder_offset_lambda: float = 0.2
    length_lambda: float = 0.05
    spine_ema_lambda: float = 0.5
    angle_ema_lambda: float = 0.5
    # Spine lock via EMA radius
    spine_lock_radius_m: float = 0.05
    spine_lock_in_frames: int = 2
    spine_lock_out_frames: int = 1
    # CV (spine) restart rules
    cv_restart_frames: int = 2
    cv_pos_violate_tol_m: float = 0.05
    # Angle lock band
    angle_tol_deg: float = 9.0
    angle_unlock_out_frames: int = 2
    angle_lock_in_frames: int = 1

# ---------- Main smoother ----------
class PoseSmoother:
    """
    filter(frame_flat): torch.tensor [N*3] -> [N*3]
    Set joint indices:
      idx_spine, idx_mid, idx_neck,
      idx_shoulder_L/R, idx_elbow_L/R, idx_wrist_L/R
    """
    def __init__(self, N_joints, cfg: PoseSmootherConfig, device='cuda', dtype=torch.float32):
        self.cfg = cfg
        self.device = torch.device(device); self.dtype = dtype
        self.N = N_joints

        # indices (caller may overwrite)
        self.idx_spine = 0
        self.idx_mid   = 1
        self.idx_neck  = 2
        self.idx_shoulder_L = 3; self.idx_shoulder_R = 4
        self.idx_elbow_L = 5;    self.idx_elbow_R = 6
        self.idx_wrist_L = 7;    self.idx_wrist_R = 8

        # EMAs (lockable)
        self.ema_spine_pos = EMA3(cfg.spine_ema_lambda, return_locked_output=True)  # frozen output when locked
        self.ema_shL_off = EMA3(cfg.shoulder_offset_lambda, return_locked_output=False)
        self.ema_shR_off = EMA3(cfg.shoulder_offset_lambda, return_locked_output=False)

        self.ema_len_sp_mid = EMA1(cfg.length_lambda, return_locked_output=False)
        self.ema_len_mid_ne = EMA1(cfg.length_lambda, return_locked_output=False)
        self.ema_len_uL = EMA1(cfg.length_lambda, return_locked_output=False)
        self.ema_len_fL = EMA1(cfg.length_lambda, return_locked_output=False)
        self.ema_len_uR = EMA1(cfg.length_lambda, return_locked_output=False)
        self.ema_len_fR = EMA1(cfg.length_lambda, return_locked_output=False)

        # Spine lock + optional CV
        self.lock_spine = EmaLock(cfg.spine_lock_radius_m, cfg.spine_lock_in_frames, cfg.spine_lock_out_frames)
        self.cv_spine = CV3(cfg.dt, cfg.cv_pos_violate_tol_m, cfg.cv_restart_frames)

        # Angle EMA locks (one per bone)
        ang_tol = cfg.angle_tol_deg * 3.14159265 / 180
        def make_angle_lock():
            ema = EMA3(cfg.angle_ema_lambda, return_locked_output=True)  # frozen when locked
            return AngleEmaLock(ema, ang_tol, cfg.angle_unlock_out_frames, cfg.angle_lock_in_frames,
                                cfg.follow_raw_angle_when_unlocked)
        self.a_sp_mid = make_angle_lock()
        self.a_mid_ne = make_angle_lock()
        self.a_uL = make_angle_lock(); self.a_fL = make_angle_lock()
        self.a_uR = make_angle_lock(); self.a_fR = make_angle_lock()

    def set_indices(self, idx_spine, idx_mid, idx_neck,
                    idx_shoulder_L, idx_shoulder_R, idx_elbow_L, idx_elbow_R, idx_wrist_L, idx_wrist_R):
        self.idx_spine = idx_spine
        self.idx_mid = idx_mid
        self.idx_neck = idx_neck
        self.idx_shoulder_L = idx_shoulder_L
        self.idx_shoulder_R = idx_shoulder_R
        self.idx_elbow_L = idx_elbow_L
        self.idx_elbow_R = idx_elbow_R
        self.idx_wrist_L = idx_wrist_L
        self.idx_wrist_R = idx_wrist_R

    def _get(self, frame, idx): return frame[idx, :]
    def _set(self, frame, idx, val): frame[idx, :] = val; return frame

    def _angle_step(self, parent_raw, child_raw, a_lock: AngleEmaLock):
        d_raw = _unit(child_raw - parent_raw)
        th_raw = _axis_angle_from_negz(d_raw)
        th_out, _ = a_lock.step(th_raw)   # uses previous EMA for inside/outside, then updates/freeze
        return _rotate_negz(th_out)

    def filter(self, frame_flat: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        x = frame_flat.view(self.N, 3).to(self.device, self.dtype)

        # raw joints
        p_sp  = self._get(x, self.idx_spine)
        p_md  = self._get(x, self.idx_mid)
        p_nk  = self._get(x, self.idx_neck)
        p_shL = self._get(x, self.idx_shoulder_L); p_shR = self._get(x, self.idx_shoulder_R)
        p_elL = self._get(x, self.idx_elbow_L);    p_elR = self._get(x, self.idx_elbow_R)
        p_wrL = self._get(x, self.idx_wrist_L);    p_wrR = self._get(x, self.idx_wrist_R)

        # ---------- 1) EMAs upfront (lengths & shoulder offsets) ----------
        L_sp_mid = self.ema_len_sp_mid.step(torch.linalg.norm(p_md - p_sp))
        L_mid_ne = self.ema_len_mid_ne.step(torch.linalg.norm(p_nk - p_md))
        L_uL = self.ema_len_uL.step(torch.linalg.norm(p_elL - p_shL))
        L_fL = self.ema_len_fL.step(torch.linalg.norm(p_wrL - p_elL))
        L_uR = self.ema_len_uR.step(torch.linalg.norm(p_elR - p_shR))
        L_fR = self.ema_len_fR.step(torch.linalg.norm(p_wrR - p_elR))

        r_shL_ema = self.ema_shL_off.step(p_shL - p_nk)
        r_shR_ema = self.ema_shR_off.step(p_shR - p_nk)

        # ---------- 2) Spine: DECIDE using previous EMA, then update & (un)lock ----------
        sp_ema_prev = self.ema_spine_pos.peek()
        if sp_ema_prev is None:
            # First frame: seed EMA and lock immediately
            self.ema_spine_pos.update(p_sp)
            self.ema_spine_pos.set_locked(True)
            p_sp_hat = self.ema_spine_pos.output()
        else:
            inside = torch.linalg.norm(p_sp - sp_ema_prev) <= cfg.spine_lock_radius_m
            now_locked = self.lock_spine.update(inside)
            # update EMA AFTER decision
            self.ema_spine_pos.update(p_sp)
            # freeze if locked (captures updated EMA)
            self.ema_spine_pos.set_locked(now_locked)

            if now_locked or not cfg.use_spine_cv:
                self.cv_spine.cancel()
                p_sp_hat = self.ema_spine_pos.output()
            else:
                if not self.cv_spine.active: self.cv_spine.start_at(p_sp)
                p_sp_hat = self.cv_spine.step(p_sp)

        # ---------- 3) Torso angles (EMA band with previous EMA) ----------
        d_sp_mid = self._angle_step(p_sp, p_md, self.a_sp_mid)
        d_mid_ne = self._angle_step(p_md, p_nk, self.a_mid_ne)

        # ---------- 4) Rebuild torso ----------
        p_md_hat = p_sp_hat + d_sp_mid * L_sp_mid
        p_nk_hat = p_md_hat + d_mid_ne * L_mid_ne

        # ---------- 5) Shoulders from EMA offsets + neck ----------
        p_shL_hat = p_nk_hat + r_shL_ema
        p_shR_hat = p_nk_hat + r_shR_ema

        # ---------- 6) Arms (angles via EMA band with previous EMA) ----------
        d_uL = self._angle_step(p_shL, p_elL, self.a_uL)
        d_fL = self._angle_step(p_elL, p_wrL, self.a_fL)
        d_uR = self._angle_step(p_shR, p_elR, self.a_uR)
        d_fR = self._angle_step(p_elR, p_wrR, self.a_fR)

        p_elL_hat = p_shL_hat + d_uL * L_uL
        p_wrL_hat = p_elL_hat + d_fL * L_fL
        p_elR_hat = p_shR_hat + d_uR * L_uR
        p_wrR_hat = p_elR_hat + d_fR * L_fR

        # ---------- 7) Assemble ----------
        out = x.clone()
        out[self.idx_spine, :]      = p_sp_hat
        out[self.idx_mid,   :]      = p_md_hat
        out[self.idx_neck,  :]      = p_nk_hat
        out[self.idx_shoulder_L, :] = p_shL_hat
        out[self.idx_shoulder_R, :] = p_shR_hat
        out[self.idx_elbow_L,   :]  = p_elL_hat
        out[self.idx_elbow_R,   :]  = p_elR_hat
        out[self.idx_wrist_L,   :]  = p_wrL_hat
        out[self.idx_wrist_R,   :]  = p_wrR_hat

        return out.view(-1)
