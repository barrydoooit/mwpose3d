#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// Reuse f3 and helpers from the original header
#include "pose_smoother.h"

// ===== Config carried on device (Gaussian EMA) =====
struct GaussianEMAConfigDevice {
    float dt;

    // Translation gaussian params per joint TYPE
    // arrays indexed by joint type (see comment above)
    float mu_trans[6];
    float delta_trans[6];

    // Rotation gaussian params per joint TYPE (angles in radians)
    float mu_rot[6];
    float delta_rot[6];

    // lambda clamps & combination rule (shared)
    float lam_min;
    float lam_max;
    int32_t combine_rule;   // 0=min, 1=product, 2=weighted
    float combine_alpha;    // used when rule==2

    // If true, apply positional smoothing (EMA) in relative coordinates to parent
    // (spine is still treated as absolute; this flag doesn't affect spine).
    int32_t use_relative_pos_except_spine;
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
