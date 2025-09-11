#include "gaussian_ema.h"
#include <cmath>

static __device__ __forceinline__ float clampf(float x, float a, float b){
    return x < a ? a : (x > b ? b : x);
}

static __device__ __forceinline__ float gaussian_val(float x, float mu, float delta){
    if(delta <= 1e-9f) return (x == mu) ? 1.f : 0.f;
    float z = (x - mu) / delta;
    return __expf(-0.5f * z * z);
}

static __device__ __forceinline__ int idx3(int j){ return 3*j; }

static __device__ __forceinline__ f3 load_xyz(const float* flat, int j){
    int i = idx3(j);
    return f3(flat[i+0], flat[i+1], flat[i+2]);
}

static __device__ __forceinline__ void store_xyz(float* flat, int j, const f3& v){
    int i = idx3(j);
    flat[i+0] = v.x; flat[i+1] = v.y; flat[i+2] = v.z;
}

static __device__ float combine_lambda(const GaussianEMAConfigDevice& cfg, float lam_pos, float lam_ang){
    lam_pos = clampf(lam_pos, 0.f, 1.f);
    lam_ang = clampf(lam_ang, 0.f, 1.f);
    float lam = 0.f;
    if (cfg.combine_rule == 0){
        lam = lam_pos < lam_ang ? lam_pos : lam_ang;  // min
    } else if (cfg.combine_rule == 1){
        lam = lam_pos * lam_ang;                      // product
    } else {
        float a = clampf(cfg.combine_alpha, 0.f, 1.f);
        lam = a * lam_pos + (1.f - a) * lam_ang;     // weighted
    }
    return clampf(lam, cfg.lam_min, cfg.lam_max);
}

static __device__ f3 ema_update(EMA3Lite& e, const f3& x, float lam){
    if(!e.has_y){
        e.y = x; e.has_y = 1;
        return x;
    }
    float one_m = 1.f - lam;
    e.y = f3_add(f3_mul(e.y, lam), f3_mul(x, one_m));
    return e.y;
}

static __device__ float angle_between(const f3& a, const f3& b){
    float da = f3_norm(a), db = f3_norm(b);
    if(da < 1e-9f || db < 1e-9f) return 0.f;
    float c = f3_dot(f3_mul(a, 1.f/da), f3_mul(b, 1.f/db));
    c = clampf(c, -1.f, 1.f);
    return acosf(c);
}

static __device__ void update_dir(DirState& ds, const f3& d){
    float n = f3_norm(d);
    if(n < 1e-9f) return;
    ds.dir = f3_mul(d, 1.f/n);
    ds.has = 1;
}

static __global__ void kernel_init(PoseSmootherGaussianState* st, int N, GaussianEMAConfigDevice cfg){
    st->cfg = cfg;
    st->N = N;
    st->idx_spine = 0; st->idx_mid = 1; st->idx_neck = 2;
    st->idx_shL = 3;   st->idx_shR = 4;
    st->idx_elL = 5;   st->idx_elR = 6;
    st->idx_wrL = 7;   st->idx_wrR = 8;

    st->ema_sp = EMA3Lite(); st->ema_mid = EMA3Lite(); st->ema_ne = EMA3Lite();
    st->ema_shL = EMA3Lite(); st->ema_shR = EMA3Lite();
    st->ema_elL = EMA3Lite(); st->ema_elR = EMA3Lite();
    st->ema_wrL = EMA3Lite(); st->ema_wrR = EMA3Lite();

    st->has_prev_raw = 0;
    st->dir_sp_mid = DirState(); st->dir_mid_ne = DirState();
    st->dir_uL = DirState();     st->dir_fL   = DirState();
    st->dir_uR = DirState();     st->dir_fR   = DirState();
}

static __global__ void kernel_set_indices(PoseSmootherGaussianState* st,
    int idx_spine, int idx_mid, int idx_neck,
    int idx_shL,  int idx_shR, int idx_elL, int idx_elR,
    int idx_wrL,  int idx_wrR)
{
    st->idx_spine = idx_spine; st->idx_mid = idx_mid; st->idx_neck = idx_neck;
    st->idx_shL = idx_shL;     st->idx_shR = idx_shR;
    st->idx_elL = idx_elL;     st->idx_elR = idx_elR;
    st->idx_wrL = idx_wrL;     st->idx_wrR = idx_wrR;
}

static __global__ void kernel_reset(PoseSmootherGaussianState* st){
    // Re-initialize state inline (no device-side kernel launches)
    GaussianEMAConfigDevice cfg = st->cfg;
    int N = st->N;

    st->idx_spine = 0; st->idx_mid = 1; st->idx_neck = 2;
    st->idx_shL = 3;   st->idx_shR = 4;
    st->idx_elL = 5;   st->idx_elR = 6;
    st->idx_wrL = 7;   st->idx_wrR = 8;

    st->ema_sp = EMA3Lite(); st->ema_mid = EMA3Lite(); st->ema_ne = EMA3Lite();
    st->ema_shL = EMA3Lite(); st->ema_shR = EMA3Lite();
    st->ema_elL = EMA3Lite(); st->ema_elR = EMA3Lite();
    st->ema_wrL = EMA3Lite(); st->ema_wrR = EMA3Lite();

    st->has_prev_raw = 0;
    st->dir_sp_mid = DirState(); st->dir_mid_ne = DirState();
    st->dir_uL = DirState();     st->dir_fL   = DirState();
    st->dir_uR = DirState();     st->dir_fR   = DirState();

    st->cfg = cfg; st->N = N;
}

static __global__ void kernel_filter(PoseSmootherGaussianState* st, const float* in, float* out){
    // default: copy-through
    for(int i=0;i<st->N*3;i++) out[i] = in[i];

    // Load current raw positions
    const int is = st->idx_spine, im = st->idx_mid, inx = st->idx_neck;
    const int iSL= st->idx_shL,   iSR= st->idx_shR;
    const int iEL= st->idx_elL,   iER= st->idx_elR;
    const int iWL= st->idx_wrL,   iWR= st->idx_wrR;

    f3 p_sp = load_xyz(in, is);
    f3 p_mid= load_xyz(in, im);
    f3 p_ne = load_xyz(in, inx);
    f3 p_shL= load_xyz(in, iSL);
    f3 p_shR= load_xyz(in, iSR);
    f3 p_elL= load_xyz(in, iEL);
    f3 p_elR= load_xyz(in, iER);
    f3 p_wrL= load_xyz(in, iWL);
    f3 p_wrR= load_xyz(in, iWR);

    // Per-joint displacement deltas (meters)
    float d_sp=0, d_mid=0, d_ne=0, d_shL=0, d_shR=0, d_elL=0, d_elR=0, d_wrL=0, d_wrR=0;
    if(st->has_prev_raw){
        d_sp  = f3_norm(f3_sub(p_sp,  st->prev_sp));
        d_mid = f3_norm(f3_sub(p_mid, st->prev_mid));
        d_ne  = f3_norm(f3_sub(p_ne,  st->prev_ne));
        d_shL = f3_norm(f3_sub(p_shL, st->prev_shL));
        d_shR = f3_norm(f3_sub(p_shR, st->prev_shR));
        d_elL = f3_norm(f3_sub(p_elL, st->prev_elL));
        d_elR = f3_norm(f3_sub(p_elR, st->prev_elR));
        d_wrL = f3_norm(f3_sub(p_wrL, st->prev_wrL));
        d_wrR = f3_norm(f3_sub(p_wrR, st->prev_wrR));
    }

    // Segment directions
    f3 d_sp_mid = f3_sub(p_mid, p_sp);
    f3 d_mid_ne = f3_sub(p_ne,  p_mid);
    f3 d_uL     = f3_sub(p_elL, p_shL);
    f3 d_fL     = f3_sub(p_wrL, p_elL);
    f3 d_uR     = f3_sub(p_elR, p_shR);
    f3 d_fR     = f3_sub(p_wrR, p_elR);

    // Angle diffs (radians)
    float a_sp_mid = 0.f, a_mid_ne=0.f, a_uL=0.f, a_fL=0.f, a_uR=0.f, a_fR=0.f;
    if(st->dir_sp_mid.has) a_sp_mid = angle_between(st->dir_sp_mid.dir, d_sp_mid);
    if(st->dir_mid_ne.has) a_mid_ne = angle_between(st->dir_mid_ne.dir, d_mid_ne);
    if(st->dir_uL.has)     a_uL     = angle_between(st->dir_uL.dir,     d_uL);
    if(st->dir_fL.has)     a_fL     = angle_between(st->dir_fL.dir,     d_fL);
    if(st->dir_uR.has)     a_uR     = angle_between(st->dir_uR.dir,     d_uR);
    if(st->dir_fR.has)     a_fR     = angle_between(st->dir_fR.dir,     d_fR);

    // Gaussian lambdas
    const auto& cfg = st->cfg;
    float lp_sp  = gaussian_val(d_sp,  cfg.mu_pos, cfg.delta_pos);
    float lp_mid = gaussian_val(d_mid, cfg.mu_pos, cfg.delta_pos);
    float lp_ne  = gaussian_val(d_ne,  cfg.mu_pos, cfg.delta_pos);
    float lp_shL = gaussian_val(d_shL, cfg.mu_pos, cfg.delta_pos);
    float lp_shR = gaussian_val(d_shR, cfg.mu_pos, cfg.delta_pos);
    float lp_elL = gaussian_val(d_elL, cfg.mu_pos, cfg.delta_pos);
    float lp_elR = gaussian_val(d_elR, cfg.mu_pos, cfg.delta_pos);
    float lp_wrL = gaussian_val(d_wrL, cfg.mu_pos, cfg.delta_pos);
    float lp_wrR = gaussian_val(d_wrR, cfg.mu_pos, cfg.delta_pos);

    float la_sp_mid = gaussian_val(a_sp_mid, cfg.mu_ang, cfg.delta_ang);
    float la_mid_ne = gaussian_val(a_mid_ne, cfg.mu_ang, cfg.delta_ang);
    float la_uL     = gaussian_val(a_uL,     cfg.mu_ang, cfg.delta_ang);
    float la_fL     = gaussian_val(a_fL,     cfg.mu_ang, cfg.delta_ang);
    float la_uR     = gaussian_val(a_uR,     cfg.mu_ang, cfg.delta_ang);
    float la_fR     = gaussian_val(a_fR,     cfg.mu_ang, cfg.delta_ang);

    // Map angle lambda to joints (take the more conservative/min of associated segments)
    float lam_sp  = combine_lambda(cfg, lp_sp,  la_sp_mid);
    float lam_mid = combine_lambda(cfg, lp_mid, fminf(la_sp_mid, la_mid_ne));
    float lam_ne  = combine_lambda(cfg, lp_ne,  la_mid_ne);

    float lam_shL = combine_lambda(cfg, lp_shL, la_uL);
    float lam_elL = combine_lambda(cfg, lp_elL, fminf(la_uL, la_fL));
    float lam_wrL = combine_lambda(cfg, lp_wrL, la_fL);

    float lam_shR = combine_lambda(cfg, lp_shR, la_uR);
    float lam_elR = combine_lambda(cfg, lp_elR, fminf(la_uR, la_fR));
    float lam_wrR = combine_lambda(cfg, lp_wrR, la_fR);

    // EMA updates
    f3 y_sp  = ema_update(st->ema_sp,  p_sp,  lam_sp);
    f3 y_mid = ema_update(st->ema_mid, p_mid, lam_mid);
    f3 y_ne  = ema_update(st->ema_ne,  p_ne,  lam_ne);

    f3 y_shL = ema_update(st->ema_shL, p_shL, lam_shL);
    f3 y_elL = ema_update(st->ema_elL, p_elL, lam_elL);
    f3 y_wrL = ema_update(st->ema_wrL, p_wrL, lam_wrL);

    f3 y_shR = ema_update(st->ema_shR, p_shR, lam_shR);
    f3 y_elR = ema_update(st->ema_elR, p_elR, lam_elR);
    f3 y_wrR = ema_update(st->ema_wrR, p_wrR, lam_wrR);

    // Write back smoothed for the 9 controlled joints
    store_xyz(out, is,  y_sp);
    store_xyz(out, im,  y_mid);
    store_xyz(out, inx, y_ne);
    store_xyz(out, iSL, y_shL);
    store_xyz(out, iEL, y_elL);
    store_xyz(out, iWL, y_wrL);
    store_xyz(out, iSR, y_shR);
    store_xyz(out, iER, y_elR);
    store_xyz(out, iWR, y_wrR);

    // Update prev raw + segment dirs
    st->prev_sp  = p_sp;  st->prev_mid = p_mid; st->prev_ne = p_ne;
    st->prev_shL = p_shL; st->prev_elL = p_elL; st->prev_wrL = p_wrL;
    st->prev_shR = p_shR; st->prev_elR = p_elR; st->prev_wrR = p_wrR;
    st->has_prev_raw = 1;

    update_dir(st->dir_sp_mid, d_sp_mid);
    update_dir(st->dir_mid_ne, d_mid_ne);
    update_dir(st->dir_uL,     d_uL);
    update_dir(st->dir_fL,     d_fL);
    update_dir(st->dir_uR,     d_uR);
    update_dir(st->dir_fR,     d_fR);
}

// ---- Host-callable wrappers ----
void init_state_gaussian_cuda(torch::Tensor state_buf, int N, const GaussianEMAConfigDevice& cfg){
    TORCH_CHECK(state_buf.is_cuda(), "state must be CUDA");
    TORCH_CHECK(state_buf.dtype() == torch::kUInt8, "state dtype must be uint8");
    auto* st = reinterpret_cast<PoseSmootherGaussianState*>(state_buf.data_ptr<uint8_t>());
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_init<<<1,1,0,stream>>>(st, N, cfg);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void set_indices_gaussian_cuda(torch::Tensor state_buf,
                               int idx_spine, int idx_mid, int idx_neck,
                               int idx_shL,  int idx_shR, int idx_elL, int idx_elR,
                               int idx_wrL,  int idx_wrR)
{
    TORCH_CHECK(state_buf.is_cuda(), "state must be CUDA");
    auto* st = reinterpret_cast<PoseSmootherGaussianState*>(state_buf.data_ptr<uint8_t>());
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_set_indices<<<1,1,0,stream>>>(st, idx_spine, idx_mid, idx_neck, idx_shL, idx_shR, idx_elL, idx_elR, idx_wrL, idx_wrR);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor filter_gaussian_cuda(torch::Tensor state_buf, torch::Tensor frame_flat){
    TORCH_CHECK(state_buf.is_cuda(), "state must be CUDA");
    TORCH_CHECK(frame_flat.is_cuda(), "input must be CUDA");
    TORCH_CHECK(frame_flat.dim()==1, "expected [N*3]");
    TORCH_CHECK(frame_flat.dtype()==torch::kFloat32, "expected float32");

    int64_t nn = frame_flat.size(0);
    auto out = torch::empty_like(frame_flat);
    auto* st = reinterpret_cast<PoseSmootherGaussianState*>(state_buf.data_ptr<uint8_t>());
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_filter<<<1,1,0,stream>>>(st,
        frame_flat.data_ptr<float>(),
        out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor filter_sequence_gaussian_cuda(torch::Tensor state_buf, torch::Tensor frames_Tx3N){
    TORCH_CHECK(state_buf.is_cuda(), "state must be CUDA");
    TORCH_CHECK(frames_Tx3N.is_cuda(), "input must be CUDA");
    TORCH_CHECK(frames_Tx3N.dim()==2, "expected [T, N*3]");
    TORCH_CHECK(frames_Tx3N.dtype()==torch::kFloat32, "expected float32");

    auto T = frames_Tx3N.size(0);
    auto out = torch::empty_like(frames_Tx3N);
    for (int64_t t = 0; t < T; ++t){
        auto in_row  = frames_Tx3N.slice(0, t, t+1).reshape({-1});
        auto out_row = filter_gaussian_cuda(state_buf, in_row);
        out.slice(0, t, t+1).copy_(out_row.reshape({1, -1}));
    }
    return out;
}

void reset_gaussian_cuda(torch::Tensor state_buf){
    TORCH_CHECK(state_buf.is_cuda(), "state must be CUDA");
    auto* st = reinterpret_cast<PoseSmootherGaussianState*>(state_buf.data_ptr<uint8_t>());
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_reset<<<1,1,0,stream>>>(st);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
