#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// Reuse f3 and helpers from the original header
#include "pose_smoother.h"

// ===== Config carried on device (Gaussian EMA) =====
struct GaussianEMAConfigDevice {
    float dt;               // not strictly required here, but kept for parity
    float mu_pos;           // meters; center of Gaussian for position diff
    float delta_pos;        // meters; stddev of Gaussian for position diff
    float mu_ang;           // radians; center of Gaussian for angle diff
    float delta_ang;        // radians; stddev of Gaussian for angle diff
    float lam_min;          // clamp
    float lam_max;          // clamp
    int32_t combine_rule;   // 0=min, 1=product, 2=weighted sum
    float combine_alpha;    // for weighted rule: lam = a*lam_pos + (1-a)*lam_ang
};

// ===== Tiny EMA state (3D) =====
struct EMA3Lite {
    int32_t has_y = 0;
    f3      y     = f3(0,0,0);
};

// ===== Previous direction for a segment (unit vector) =====
struct DirState {
    int32_t has  = 0;
    f3      dir  = f3(0,0,0);
};

// ===== Whole persistent state on device (Gaussian EMA) =====
struct PoseSmootherGaussianState {
    GaussianEMAConfigDevice cfg;
    int32_t N;

    // indices (same meaning/order as original)
    int32_t idx_spine, idx_mid, idx_neck;
    int32_t idx_shL, idx_shR, idx_elL, idx_elR, idx_wrL, idx_wrR;

    // EMA for the 9 driven joints
    EMA3Lite ema_sp, ema_mid, ema_ne, ema_shL, ema_shR, ema_elL, ema_elR, ema_wrL, ema_wrR;

    // Previous raw positions (for per-joint displacement)
    int32_t has_prev_raw = 0;
    f3 prev_sp, prev_mid, prev_ne, prev_shL, prev_shR, prev_elL, prev_elR, prev_wrL, prev_wrR;

    // Previous segment directions (for angle difference)
    DirState dir_sp_mid;
    DirState dir_mid_ne;
    DirState dir_uL;  // shoulderL -> elbowL
    DirState dir_fL;  // elbowL -> wristL
    DirState dir_uR;  // shoulderR -> elbowR
    DirState dir_fR;  // elbowR -> wristR
};

// ---- API (host-callable) ----
void init_state_gaussian_cuda(torch::Tensor state_buf, int N, const GaussianEMAConfigDevice& cfg);
void set_indices_gaussian_cuda(torch::Tensor state_buf,
                               int idx_spine, int idx_mid, int idx_neck,
                               int idx_shL,  int idx_shR, int idx_elL, int idx_elR,
                               int idx_wrL,  int idx_wrR);
torch::Tensor filter_gaussian_cuda(torch::Tensor state_buf, torch::Tensor frame_flat);
torch::Tensor filter_sequence_gaussian_cuda(torch::Tensor state_buf, torch::Tensor frames_Tx3N);
void reset_gaussian_cuda(torch::Tensor state_buf);
