#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// ===== Small math helper =====
struct f3 {
    float x, y, z;
    __host__ __device__ f3() : x(0), y(0), z(0) {}
    __host__ __device__ f3(float X, float Y, float Z): x(X), y(Y), z(Z) {}
};

__host__ __device__ inline f3 f3_add(const f3&a, const f3&b){ return f3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline f3 f3_sub(const f3&a, const f3&b){ return f3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ inline f3 f3_mul(const f3&a, float s){ return f3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ inline float f3_dot(const f3&a, const f3&b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline f3 f3_cross(const f3&a, const f3&b){
    return f3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__host__ __device__ inline float f3_norm(const f3&v){ return sqrtf(f3_dot(v,v)); }
__host__ __device__ inline f3 f3_unit(const f3&v){
    float n = f3_norm(v);
    if(n < 1e-8f) return f3(0,0,0);
    float inv = 1.0f / n; return f3(v.x*inv, v.y*inv, v.z*inv);
}

// ===== EMA states =====
struct EMA1State {
    float lam = 0.f;
    int32_t return_locked_output = 0;
    int32_t has_y = 0;
    int32_t locked = 0;
    float y = 0.f;
    float lock_value = 0.f;
};

struct EMA3State {
    float lam = 0.f;
    int32_t return_locked_output = 1;
    int32_t has_y = 0;
    int32_t locked = 0;
    f3 y = f3(0,0,0);
    f3 lock_value = f3(0,0,0);
    // --- NEW: static band center (frozen while locked, follows y when unlocked)
    int32_t has_center = 0;
    f3 center = f3(0,0,0);
};

struct EmaLockState {
    float radius = 0.03f;
    int32_t k_in = 2;
    int32_t k_out = 2;
    int32_t locked = 0;
    int32_t in_cnt = 0;
    int32_t out_cnt = 0;
};

struct CV3State {
    float dt = 1/60.f;
    float tol = 0.02f;
    int32_t restart_frames = 2;

    int32_t active = 0;
    int32_t has_prev_raw = 0;
    int32_t has_prev_res = 0;

    f3 p = f3(0,0,0);
    f3 v = f3(0,0,0);
    f3 prev_raw = f3(0,0,0);
    f3 prev_res = f3(0,0,0);
    int32_t violate_cnt = 0;
};

struct AngleEmaLockState {
    EMA3State ema;
    float tol = 15.f * 3.14159265f / 180.f;
    int32_t unlock_out_frames = 2;
    int32_t lock_in_frames = 1;
    int32_t follow_raw_when_unlocked = 1;

    int32_t out_cnt = 0;
    int32_t in_cnt = 0;
    int32_t initialized = 0;
};

// ===== Config carried on device =====
struct ConfigDevice {
    float dt;
    int32_t use_spine_cv;
    int32_t follow_raw_angle_when_unlocked;

    float shoulder_offset_lambda;
    float length_lambda;
    float spine_ema_lambda;
    float angle_ema_lambda;

    float spine_lock_radius_m;
    int32_t spine_lock_in_frames;
    int32_t spine_lock_out_frames;

    int32_t cv_restart_frames;
    float cv_pos_violate_tol_m;

    float angle_tol_rad;
    int32_t angle_unlock_out_frames;
    int32_t angle_lock_in_frames;
};

// ===== Whole persistent state on device =====
struct PoseSmootherState {
    ConfigDevice cfg;
    int32_t N;

    // indices
    int32_t idx_spine, idx_mid, idx_neck;
    int32_t idx_shL, idx_shR, idx_elL, idx_elR, idx_wrL, idx_wrR;

    // EMAs (running state)
    EMA3State ema_spine_pos;
    EMA3State ema_shL_off;
    EMA3State ema_shR_off;

    EMA1State ema_len_sp_mid;
    EMA1State ema_len_mid_ne;
    EMA1State ema_len_uL;
    EMA1State ema_len_fL;
    EMA1State ema_len_uR;
    EMA1State ema_len_fR;

    // spine lock + CV
    EmaLockState lock_spine;
    CV3State cv_spine;

    // angle locks
    AngleEmaLockState a_sp_mid;
    AngleEmaLockState a_mid_ne;
    AngleEmaLockState a_uL;
    AngleEmaLockState a_fL;
    AngleEmaLockState a_uR;
    AngleEmaLockState a_fR;
};

// API (host-callable)
void init_state_cuda(torch::Tensor state_buf, int N, const ConfigDevice& cfg);
void set_indices_cuda(torch::Tensor state_buf,
                      int idx_spine, int idx_mid, int idx_neck,
                      int idx_shL, int idx_shR, int idx_elL, int idx_elR, int idx_wrL, int idx_wrR);
torch::Tensor filter_cuda(torch::Tensor state_buf, torch::Tensor frame_flat);
torch::Tensor filter_sequence_cuda(torch::Tensor state_buf, torch::Tensor frames_Tx3N);
void reset_cuda(torch::Tensor state_buf);
