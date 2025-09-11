#include "pose_smoother.h"

// =================== device helpers ===================

__device__ inline f3 f3_clamp_min_norm(const f3& v, float eps=1e-8f){
    float n = f3_norm(v);
    if(n < eps) return f3(0,0,0);
    return f3_mul(v, 1.0f/n);
}

__device__ inline f3 axis_angle_from_negz(const f3& d_in){
    f3 d = f3_clamp_min_norm(d_in);
    f3 ref = f3(0,0,-1);
    float c = f3_dot(ref, d);
    c = fmaxf(-1.f, fminf(1.f, c));
    float angle = acosf(c);
    f3 axis = f3_cross(ref, d);
    float an = f3_norm(axis);
    if(an < 1e-8f) return f3(0,0,0);
    float inv = 1.0f / an;
    return f3(axis.x*inv*angle, axis.y*inv*angle, axis.z*inv*angle);
}

__device__ inline f3 rotate_negz(const f3& theta){
    f3 ref = f3(0,0,-1);
    float ang = f3_norm(theta);
    if(ang < 1e-8f) return ref;
    f3 axis = f3(theta.x/ang, theta.y/ang, theta.z/ang);
    float ca = cosf(ang), sa = sinf(ang);
    float dot = f3_dot(axis, ref);
    f3 cross = f3_cross(axis, ref);
    // Rodrigues
    f3 term1 = f3_mul(ref, ca);
    f3 term2 = f3_mul(cross, sa);
    f3 term3 = f3_mul(axis, dot*(1.f - ca));
    return f3_add(f3_add(term1, term2), term3);
}

__device__ inline float f3_dist(const f3&a, const f3&b){
    float dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
    return sqrtf(dx*dx+dy*dy+dz*dz);
}

// ===== EMA1 =====
__device__ inline float ema1_output(const EMA1State& s){
    if (s.return_locked_output && s.locked && s.has_y) return s.lock_value;
    return s.y;
}
__device__ inline void ema1_update(EMA1State& s, float x){
    if(!s.has_y){ s.y = x; s.has_y = 1; }
    else{
        float a = 1.f - s.lam;
        s.y = a*s.y + s.lam*x;
    }
}
__device__ inline void ema1_set_locked(EMA1State& s, int lock_now){
    if(lock_now && !s.locked && s.has_y){
        s.lock_value = s.y;
    }
    s.locked = lock_now;
}

// ===== EMA3: running y + static center =====
__device__ inline f3 ema3_output(const EMA3State& s){
    if (s.return_locked_output && s.locked && s.has_y) return s.lock_value;
    return s.y;
}
__device__ inline f3 ema3_peek_center(const EMA3State& s){
    return (s.has_center ? s.center : s.y);
}
__device__ inline void ema3_update(EMA3State& s, const f3& x){
    if(!s.has_y){ s.y = x; s.has_y = 1; }
    else{
        float a = 1.f - s.lam;
        s.y = f3(a*s.y.x + s.lam*x.x, a*s.y.y + s.lam*x.y, a*s.y.z + s.lam*x.z);
    }
}
__device__ inline void ema3_follow_center(EMA3State& s){
    if(s.has_y){ s.center = s.y; s.has_center = 1; }
}
__device__ inline void ema3_set_locked(EMA3State& s, int lock_now){
    if(lock_now && !s.locked && s.has_y){
        s.lock_value = s.y; // freeze output
        s.center = s.y;     // freeze band center
        s.has_center = 1;
    }
    s.locked = lock_now;
}

// ===== EmaLockState =====
__device__ inline void emalock_update(EmaLockState& L, bool inside){
    if(L.locked){
        if(inside){ L.out_cnt = 0; }
        else{
            L.out_cnt++;
            if(L.out_cnt >= L.k_out){
                L.locked = 0; L.in_cnt = 0; L.out_cnt = 0;
            }
        }
    } else {
        if(inside){
            L.in_cnt++;
            if(L.in_cnt >= L.k_in){
                L.locked = 1; L.in_cnt = 0; L.out_cnt = 0;
            }
        } else {
            L.in_cnt = 0;
        }
    }
}

// ===== CV3 =====
__device__ inline void cv3_cancel(CV3State& s){
    s.active = 0; s.has_prev_res = 0; s.violate_cnt = 0;
}
__device__ inline void cv3_start_at(CV3State& s, const f3& raw){
    if(s.has_prev_raw){
        s.v = f3((raw.x - s.prev_raw.x)/s.dt, (raw.y - s.prev_raw.y)/s.dt, (raw.z - s.prev_raw.z)/s.dt);
    } else {
        s.v = f3(0,0,0);
    }
    s.p = raw; s.active = 1; s.has_prev_res = 0; s.violate_cnt = 0;
}
__device__ inline f3 cv3_step(CV3State& s, const f3& raw){
    if(!s.active){ s.prev_raw = raw; s.has_prev_raw = 1; return raw; }
    f3 p_pred = f3_add(s.p, f3_mul(s.v, s.dt));
    f3 res = f3_sub(raw, p_pred);
    if(s.has_prev_res){
        float same = f3_dot(res, s.prev_res);
        float rnorm = f3_norm(res);
        if(same > 0 && rnorm > s.tol) s.violate_cnt++;
        else s.violate_cnt = 0;
    }
    s.prev_res = res; s.has_prev_res = 1;
    f3 out;
    if(s.violate_cnt >= s.restart_frames){
        cv3_start_at(s, raw);
        out = raw;
    } else {
        s.p = p_pred;
        out = p_pred;
    }
    s.prev_raw = raw; s.has_prev_raw = 1;
    return out;
}

// ===== Angle lock using static center =====
__device__ inline f3 angle_lock_step(AngleEmaLockState& A, const f3& th_raw){
    if(!A.initialized){
        ema3_update(A.ema, th_raw);
        ema3_set_locked(A.ema, 1);
        A.initialized = 1;
        return ema3_output(A.ema);
    }

    // decision using previous center
    f3 center_prev = ema3_peek_center(A.ema);
    bool inside = (f3_dist(th_raw, center_prev) <= A.tol);

    bool will_lock = false;
    bool will_unlock = false;

    if(A.ema.locked){
        if(inside){ A.out_cnt = 0; }
        else {
            A.out_cnt++;
            if(A.out_cnt >= A.unlock_out_frames){
                will_unlock = true; A.out_cnt = 0;
            }
        }
    } else {
        if(inside){
            A.in_cnt++;
            if(A.in_cnt >= A.lock_in_frames){ will_lock = true; A.in_cnt = 0; }
        } else {
            A.in_cnt = 0;
        }
    }

    // update running EMA AFTER decision
    ema3_update(A.ema, th_raw);

    // apply pending changes using UPDATED EMA
    if(will_unlock){
        ema3_set_locked(A.ema, 0); // unlock; center will follow next step
        ema3_follow_center(A.ema);
    } else if(will_lock){
        ema3_set_locked(A.ema, 1); // lock now to UPDATED EMA
    } else {
        if(!A.ema.locked) ema3_follow_center(A.ema);
    }

    // output
    if(A.ema.locked) return ema3_output(A.ema);
    return (A.follow_raw_when_unlocked ? th_raw : A.ema.y);
}

// =================== kernels ===================

__global__ void init_state_kernel(PoseSmootherState* S, int N, ConfigDevice cfg){
    S->cfg = cfg;
    S->N = N;

    // default indices
    S->idx_spine = 0; S->idx_mid = 1; S->idx_neck = 2;
    S->idx_shL = 3; S->idx_shR = 4; S->idx_elL = 5; S->idx_elR = 6; S->idx_wrL = 7; S->idx_wrR = 8;

    // EMAs lambdas
    S->ema_spine_pos.lam = cfg.spine_ema_lambda;  S->ema_spine_pos.return_locked_output = 1;
    S->ema_spine_pos.has_y = 0; S->ema_spine_pos.locked = 0; S->ema_spine_pos.has_center = 0;

    S->ema_shL_off.lam = cfg.shoulder_offset_lambda; S->ema_shL_off.return_locked_output = 0;
    S->ema_shL_off.has_y = 0; S->ema_shL_off.locked = 0; S->ema_shL_off.has_center = 0;

    S->ema_shR_off.lam = cfg.shoulder_offset_lambda; S->ema_shR_off.return_locked_output = 0;
    S->ema_shR_off.has_y = 0; S->ema_shR_off.locked = 0; S->ema_shR_off.has_center = 0;

    S->ema_len_sp_mid.lam = cfg.length_lambda; S->ema_len_sp_mid.has_y=0; S->ema_len_sp_mid.locked=0; S->ema_len_sp_mid.return_locked_output=0;
    S->ema_len_mid_ne.lam = cfg.length_lambda; S->ema_len_mid_ne.has_y=0; S->ema_len_mid_ne.locked=0; S->ema_len_mid_ne.return_locked_output=0;
    S->ema_len_uL.lam = cfg.length_lambda; S->ema_len_uL.has_y=0; S->ema_len_uL.locked=0; S->ema_len_uL.return_locked_output=0;
    S->ema_len_fL.lam = cfg.length_lambda; S->ema_len_fL.has_y=0; S->ema_len_fL.locked=0; S->ema_len_fL.return_locked_output=0;
    S->ema_len_uR.lam = cfg.length_lambda; S->ema_len_uR.has_y=0; S->ema_len_uR.locked=0; S->ema_len_uR.return_locked_output=0;
    S->ema_len_fR.lam = cfg.length_lambda; S->ema_len_fR.has_y=0; S->ema_len_fR.locked=0; S->ema_len_fR.return_locked_output=0;

    // spine lock
    S->lock_spine.radius = cfg.spine_lock_radius_m;
    S->lock_spine.k_in = cfg.spine_lock_in_frames;
    S->lock_spine.k_out = cfg.spine_lock_out_frames;
    S->lock_spine.locked = 0; S->lock_spine.in_cnt=0; S->lock_spine.out_cnt=0;

    // CV (spine)
    S->cv_spine.dt = cfg.dt; S->cv_spine.tol = cfg.cv_pos_violate_tol_m; S->cv_spine.restart_frames = cfg.cv_restart_frames;
    S->cv_spine.active=0; S->cv_spine.has_prev_raw=0; S->cv_spine.has_prev_res=0; S->cv_spine.violate_cnt=0;

    // angle locks
    auto init_angle = [&](AngleEmaLockState& A){
        A.ema.lam = cfg.angle_ema_lambda; A.ema.return_locked_output = 1;
        A.ema.has_y=0; A.ema.locked=0; A.ema.has_center=0;
        A.tol = cfg.angle_tol_rad;
        A.unlock_out_frames = cfg.angle_unlock_out_frames;
        A.lock_in_frames = cfg.angle_lock_in_frames;
        A.follow_raw_when_unlocked = cfg.follow_raw_angle_when_unlocked;
        A.out_cnt=0; A.in_cnt=0; A.initialized=0;
    };
    init_angle(S->a_sp_mid);
    init_angle(S->a_mid_ne);
    init_angle(S->a_uL);
    init_angle(S->a_fL);
    init_angle(S->a_uR);
    init_angle(S->a_fR);
}

__global__ void set_indices_kernel(PoseSmootherState* S,
                                   int idx_spine, int idx_mid, int idx_neck,
                                   int idx_shL, int idx_shR, int idx_elL, int idx_elR, int idx_wrL, int idx_wrR){
    S->idx_spine = idx_spine; S->idx_mid=idx_mid; S->idx_neck=idx_neck;
    S->idx_shL=idx_shL; S->idx_shR=idx_shR; S->idx_elL=idx_elL; S->idx_elR=idx_elR; S->idx_wrL=idx_wrL; S->idx_wrR=idx_wrR;
}

__global__ void reset_kernel(PoseSmootherState* S){
    // reset dynamics (keep cfg and indices)
    S->ema_spine_pos.has_y=0; S->ema_spine_pos.locked=0; S->ema_spine_pos.has_center=0;
    S->ema_shL_off.has_y=0;   S->ema_shL_off.locked=0;   S->ema_shL_off.has_center=0;
    S->ema_shR_off.has_y=0;   S->ema_shR_off.locked=0;   S->ema_shR_off.has_center=0;

    S->ema_len_sp_mid.has_y=0; S->ema_len_sp_mid.locked=0;
    S->ema_len_mid_ne.has_y=0; S->ema_len_mid_ne.locked=0;
    S->ema_len_uL.has_y=0;     S->ema_len_uL.locked=0;
    S->ema_len_fL.has_y=0;     S->ema_len_fL.locked=0;
    S->ema_len_uR.has_y=0;     S->ema_len_uR.locked=0;
    S->ema_len_fR.has_y=0;     S->ema_len_fR.locked=0;

    S->lock_spine.locked=0; S->lock_spine.in_cnt=0; S->lock_spine.out_cnt=0;

    S->cv_spine.active=0; S->cv_spine.has_prev_raw=0; S->cv_spine.has_prev_res=0; S->cv_spine.violate_cnt=0;

    auto reset_A = [&](AngleEmaLockState& A){
        A.ema.has_y=0; A.ema.locked=0; A.ema.has_center=0;
        A.out_cnt=0; A.in_cnt=0; A.initialized=0;
    };
    reset_A(S->a_sp_mid);
    reset_A(S->a_mid_ne);
    reset_A(S->a_uL);
    reset_A(S->a_fL);
    reset_A(S->a_uR);
    reset_A(S->a_fR);
}

__global__ void filter_kernel(PoseSmootherState* S, const float* in_flat, float* out_flat){
    const int N = S->N;
    // read indices
    int iSp = S->idx_spine, iMd=S->idx_mid, iNk=S->idx_neck;
    int iShL=S->idx_shL, iShR=S->idx_shR, iElL=S->idx_elL, iElR=S->idx_elR, iWrL=S->idx_wrL, iWrR=S->idx_wrR;

    auto load = [&](int idx)->f3{
        int k = idx*3;
        return f3(in_flat[k+0], in_flat[k+1], in_flat[k+2]);
    };
    auto store = [&](int idx, const f3& v){
        int k = idx*3;
        out_flat[k+0]=v.x; out_flat[k+1]=v.y; out_flat[k+2]=v.z;
    };

    // raw joints
    f3 p_sp  = load(iSp);
    f3 p_md  = load(iMd);
    f3 p_nk  = load(iNk);
    f3 p_shL = load(iShL), p_shR = load(iShR);
    f3 p_elL = load(iElL), p_elR = load(iElR);
    f3 p_wrL = load(iWrL), p_wrR = load(iWrR);

    // 1) EMAs: lengths + shoulder offsets
    float L_sp_mid_raw = f3_norm(f3_sub(p_md, p_sp));
    float L_mid_ne_raw = f3_norm(f3_sub(p_nk, p_md));
    float L_uL_raw = f3_norm(f3_sub(p_elL, p_shL));
    float L_fL_raw = f3_norm(f3_sub(p_wrL, p_elL));
    float L_uR_raw = f3_norm(f3_sub(p_elR, p_shR));
    float L_fR_raw = f3_norm(f3_sub(p_wrR, p_elR));

    ema1_update(S->ema_len_sp_mid, L_sp_mid_raw);
    ema1_update(S->ema_len_mid_ne, L_mid_ne_raw);
    ema1_update(S->ema_len_uL, L_uL_raw);
    ema1_update(S->ema_len_fL, L_fL_raw);
    ema1_update(S->ema_len_uR, L_uR_raw);
    ema1_update(S->ema_len_fR, L_fR_raw);

    float L_sp_mid = ema1_output(S->ema_len_sp_mid);
    float L_mid_ne = ema1_output(S->ema_len_mid_ne);
    float L_uL = ema1_output(S->ema_len_uL);
    float L_fL = ema1_output(S->ema_len_fL);
    float L_uR = ema1_output(S->ema_len_uR);
    float L_fR = ema1_output(S->ema_len_fR);

    // shoulder offsets EMA (no locking)
    f3 r_shL = f3_sub(p_shL, p_nk);
    f3 r_shR = f3_sub(p_shR, p_nk);
    ema3_update(S->ema_shL_off, r_shL);
    ema3_update(S->ema_shR_off, r_shR);
    f3 r_shL_ema = S->ema_shL_off.y;
    f3 r_shR_ema = S->ema_shR_off.y;

    // 2) Spine: STATIC center decision, then update/freeze/follow
    bool first = (S->ema_spine_pos.has_y == 0);
    f3 p_sp_hat;

    if(first){
        ema3_update(S->ema_spine_pos, p_sp);
        ema3_set_locked(S->ema_spine_pos, 1); // also freezes center
        S->lock_spine.locked = 1; S->lock_spine.in_cnt=0; S->lock_spine.out_cnt=0;
        p_sp_hat = ema3_output(S->ema_spine_pos);
        cv3_cancel(S->cv_spine);
    } else {
        f3 center_prev = ema3_peek_center(S->ema_spine_pos);
        bool inside = (f3_dist(p_sp, center_prev) <= S->cfg.spine_lock_radius_m);

        // decide and set will_lock / will_unlock flags (do NOT change S->lock_spine.locked yet)
        bool will_lock_sp = false;
        bool will_unlock_sp = false;
        if(S->lock_spine.locked){
            if(inside){ S->lock_spine.out_cnt = 0; }
            else {
                S->lock_spine.out_cnt++;
                if(S->lock_spine.out_cnt >= S->lock_spine.k_out){
                    will_unlock_sp = true; S->lock_spine.out_cnt = 0;
                }
            }
        } else {
            if(inside){
                S->lock_spine.in_cnt++;
                if(S->lock_spine.in_cnt >= S->lock_spine.k_in){
                    will_lock_sp = true; S->lock_spine.in_cnt = 0;
                }
            } else {
                S->lock_spine.in_cnt = 0;
            }
        }

        // update EMA after decision
        ema3_update(S->ema_spine_pos, p_sp);

        // apply pending state changes ON UPDATED EMA
        if(will_unlock_sp){
            S->lock_spine.locked = 0;
            ema3_set_locked(S->ema_spine_pos, 0);
            ema3_follow_center(S->ema_spine_pos);
        } else if(will_lock_sp){
            S->lock_spine.locked = 1;
            ema3_set_locked(S->ema_spine_pos, 1);
        } else {
            if(S->lock_spine.locked){
                // remain locked: nothing to do
            } else {
                ema3_follow_center(S->ema_spine_pos);
            }
        }

        // produce output (CV optional)
        if(S->lock_spine.locked || !S->cfg.use_spine_cv){
            cv3_cancel(S->cv_spine);
            p_sp_hat = ema3_output(S->ema_spine_pos);
        } else {
            if(!S->cv_spine.active) cv3_start_at(S->cv_spine, p_sp);
            p_sp_hat = cv3_step(S->cv_spine, p_sp);
        }
    }

    // 3) Torso angles via static-center angle locks
    f3 d_sp_mid = rotate_negz( angle_lock_step(S->a_sp_mid, axis_angle_from_negz(f3_sub(p_md, p_sp))) );
    f3 d_mid_ne = rotate_negz( angle_lock_step(S->a_mid_ne, axis_angle_from_negz(f3_sub(p_nk, p_md))) );

    // 4) Rebuild torso
    f3 p_md_hat = f3_add(p_sp_hat, f3_mul(d_sp_mid, L_sp_mid));
    f3 p_nk_hat = f3_add(p_md_hat, f3_mul(d_mid_ne, L_mid_ne));

    // 5) Shoulders from EMA offsets + neck
    f3 p_shL_hat = f3_add(p_nk_hat, r_shL_ema);
    f3 p_shR_hat = f3_add(p_nk_hat, r_shR_ema);

    // 6) Arms: angles via static-center locks
    f3 d_uL = rotate_negz( angle_lock_step(S->a_uL, axis_angle_from_negz(f3_sub(p_elL, p_shL))) );
    f3 d_fL = rotate_negz( angle_lock_step(S->a_fL, axis_angle_from_negz(f3_sub(p_wrL, p_elL))) );
    f3 d_uR = rotate_negz( angle_lock_step(S->a_uR, axis_angle_from_negz(f3_sub(p_elR, p_shR))) );
    f3 d_fR = rotate_negz( angle_lock_step(S->a_fR, axis_angle_from_negz(f3_sub(p_wrR, p_elR))) );

    f3 p_elL_hat = f3_add(p_shL_hat, f3_mul(d_uL, L_uL));
    f3 p_wrL_hat = f3_add(p_elL_hat, f3_mul(d_fL, L_fL));
    f3 p_elR_hat = f3_add(p_shR_hat, f3_mul(d_uR, L_uR));
    f3 p_wrR_hat = f3_add(p_elR_hat, f3_mul(d_fR, L_fR));

    // 7) write out
    for(int i=0;i<N;i++){
        // start with raw copied to output
        int k = i*3; out_flat[k]=in_flat[k]; out_flat[k+1]=in_flat[k+1]; out_flat[k+2]=in_flat[k+2];
    }
    store(iSp,  p_sp_hat);
    store(iMd,  p_md_hat);
    store(iNk,  p_nk_hat);
    store(iShL, p_shL_hat);
    store(iShR, p_shR_hat);
    store(iElL, p_elL_hat);
    store(iElR, p_elR_hat);
    store(iWrL, p_wrL_hat);
    store(iWrR, p_wrR_hat);
}

// =================== host entry points ===================

void init_state_cuda(torch::Tensor state_buf, int N, const ConfigDevice& cfg){
    auto* S = reinterpret_cast<PoseSmootherState*>(state_buf.data_ptr<uint8_t>());
    init_state_kernel<<<1,1>>>(S, N, cfg);
    cudaDeviceSynchronize();
}

void set_indices_cuda(torch::Tensor state_buf,
                      int idx_spine, int idx_mid, int idx_neck,
                      int idx_shL, int idx_shR, int idx_elL, int idx_elR, int idx_wrL, int idx_wrR){
    auto* S = reinterpret_cast<PoseSmootherState*>(state_buf.data_ptr<uint8_t>());
    set_indices_kernel<<<1,1>>>(S, idx_spine, idx_mid, idx_neck, idx_shL, idx_shR, idx_elL, idx_elR, idx_wrL, idx_wrR);
    cudaDeviceSynchronize();
}

torch::Tensor filter_cuda(torch::Tensor state_buf, torch::Tensor frame_flat){
    const auto Nbytes = frame_flat.nbytes();
    auto out = torch::empty_like(frame_flat);
    auto* S = reinterpret_cast<PoseSmootherState*>(state_buf.data_ptr<uint8_t>());
    const float* in_ptr  = frame_flat.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    filter_kernel<<<1,1>>>(S, in_ptr, out_ptr);
    cudaDeviceSynchronize();
    return out;
}

torch::Tensor filter_sequence_cuda(torch::Tensor state_buf, torch::Tensor frames_Tx3N){
    TORCH_CHECK(frames_Tx3N.dim()==2, "frames must be [T, N*3]");
    const int64_t T = frames_Tx3N.size(0);
    auto out = torch::empty_like(frames_Tx3N);
    for(int64_t t=0; t<T; ++t){
        auto in_t  = frames_Tx3N[t];
        auto out_t = filter_cuda(state_buf, in_t);
        out[t].copy_(out_t);
    }
    return out;
}

void reset_cuda(torch::Tensor state_buf){
    auto* S = reinterpret_cast<PoseSmootherState*>(state_buf.data_ptr<uint8_t>());
    reset_kernel<<<1,1>>>(S);
    cudaDeviceSynchronize();
}
