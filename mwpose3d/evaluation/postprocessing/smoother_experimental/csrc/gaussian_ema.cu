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
    // Copy default
    for(int i=0;i<st->N*3;i++) out[i] = in[i];

    // indices
    const int is = st->idx_spine, im = st->idx_mid, inx = st->idx_neck;
    const int iSL= st->idx_shL,   iSR= st->idx_shR;
    const int iEL= st->idx_elL,   iER= st->idx_elR;
    const int iWL= st->idx_wrL,   iWR= st->idx_wrR;

    // Raw world positions
    f3 p_sp = load_xyz(in, is);
    f3 p_mid= load_xyz(in, im);
    f3 p_ne = load_xyz(in, inx);
    f3 p_shL= load_xyz(in, iSL);
    f3 p_shR= load_xyz(in, iSR);
    f3 p_elL= load_xyz(in, iEL);
    f3 p_elR= load_xyz(in, iER);
    f3 p_wrL= load_xyz(in, iWL);
    f3 p_wrR= load_xyz(in, iWR);

    // ---- Compute relative offsets for nodes that will be possibly smoothed relatively ----
    // chain: spine -> mid -> neck -> shoulder -> elbow -> wrist
    f3 rel_mid = f3_sub(p_mid, p_sp);   // mid rel to spine
    f3 rel_ne  = f3_sub(p_ne,  p_mid);  // neck rel to mid
    f3 rel_shL = f3_sub(p_shL, p_ne);   // shoulder rel to neck
    f3 rel_shR = f3_sub(p_shR, p_ne);
    f3 rel_elL = f3_sub(p_elL, p_shL);  // elbow rel to shoulder
    f3 rel_elR = f3_sub(p_elR, p_shR);
    f3 rel_wrL = f3_sub(p_wrL, p_elL);  // wrist rel to elbow
    f3 rel_wrR = f3_sub(p_wrR, p_elR);

    // ---- Positional deltas (translation) ----
    // Use absolute deltas for spine. For other joints, if previous raw exists,
    // compute delta in relative coordinates (current_rel - prev_rel).
    float d_sp=0.f, d_mid=0.f, d_ne=0.f, d_shL=0.f, d_shR=0.f, d_elL=0.f, d_elR=0.f, d_wrL=0.f, d_wrR=0.f;
    if(st->has_prev_raw){
        // absolute delta for spine
        d_sp  = f3_norm(f3_sub(p_sp,  st->prev_sp));

        // relative deltas for the rest (parent-relative)
        d_mid = f3_norm(f3_sub(rel_mid, f3_sub(st->prev_mid, st->prev_sp)));
        d_ne  = f3_norm(f3_sub(rel_ne,  f3_sub(st->prev_ne,  st->prev_mid)));
        d_shL = f3_norm(f3_sub(rel_shL, f3_sub(st->prev_shL, st->prev_ne)));
        d_shR = f3_norm(f3_sub(rel_shR, f3_sub(st->prev_shR, st->prev_ne)));
        d_elL = f3_norm(f3_sub(rel_elL, f3_sub(st->prev_elL, st->prev_shL)));
        d_elR = f3_norm(f3_sub(rel_elR, f3_sub(st->prev_elR, st->prev_shR)));
        d_wrL = f3_norm(f3_sub(rel_wrL, f3_sub(st->prev_wrL, st->prev_elL)));
        d_wrR = f3_norm(f3_sub(rel_wrR, f3_sub(st->prev_wrR, st->prev_elR)));
    }

    // ---- Segment directions for rotation deltas (same as before) ----
    f3 d_sp_mid = f3_sub(p_mid, p_sp);
    f3 d_mid_ne = f3_sub(p_ne,  p_mid);
    f3 d_uL     = f3_sub(p_elL, p_shL);
    f3 d_fL     = f3_sub(p_wrL, p_elL);
    f3 d_uR     = f3_sub(p_elR, p_shR);
    f3 d_fR     = f3_sub(p_wrR, p_elR);

    float a_sp_mid = 0.f, a_mid_ne=0.f, a_uL=0.f, a_fL=0.f, a_uR=0.f, a_fR=0.f;
    if(st->dir_sp_mid.has) a_sp_mid = angle_between(st->dir_sp_mid.dir, d_sp_mid);
    if(st->dir_mid_ne.has) a_mid_ne = angle_between(st->dir_mid_ne.dir, d_mid_ne);
    if(st->dir_uL.has)     a_uL     = angle_between(st->dir_uL.dir,     d_uL);
    if(st->dir_fL.has)     a_fL     = angle_between(st->dir_fL.dir,     d_fL);
    if(st->dir_uR.has)     a_uR     = angle_between(st->dir_uR.dir,     d_uR);
    if(st->dir_fR.has)     a_fR     = angle_between(st->dir_fR.dir,     d_fR);

    // ---- Map joint-type -> gaussian parameters and compute lambdas ----
    // joint type indexes: 0=spine,1=mid,2=neck,3=shoulder,4=elbow,5=wrist
    const auto& cfg = st->cfg;

    // translation gaussians (use per-type mu/delta)
    auto g_trans = [&](int joint_type, float delta_val){
        return gaussian_val(delta_val, cfg.mu_trans[joint_type], cfg.delta_trans[joint_type]);
    };
    // rotation gaussians (per-type)
    auto g_rot = [&](int joint_type, float ang_val){
        return gaussian_val(ang_val, cfg.mu_rot[joint_type], cfg.delta_rot[joint_type]);
    };

    // compute g for pos and rot for each controlled joint
    float gt_sp  = g_trans(0, d_sp);
    float gt_mid = g_trans(1, d_mid);
    float gt_ne  = g_trans(2, d_ne);
    float gt_shL = g_trans(3, d_shL);
    float gt_shR = g_trans(3, d_shR);
    float gt_elL = g_trans(4, d_elL);
    float gt_elR = g_trans(4, d_elR);
    float gt_wrL = g_trans(5, d_wrL);
    float gt_wrR = g_trans(5, d_wrR);

    float gr_sp_mid = g_rot(0, a_sp_mid); // use spine type for segment spine->mid rotation (or choose mapping)
    float gr_mid_ne = g_rot(1, a_mid_ne);
    float gr_uL     = g_rot(3, a_uL); // upper arm uses shoulder-type rotation params
    float gr_fL     = g_rot(4, a_fL); // forearm uses elbow-type rotation params
    float gr_uR     = g_rot(3, a_uR);
    float gr_fR     = g_rot(4, a_fR);

    // combine translation and rotation Gaussians into final lambda per joint
    float lam_sp  = combine_lambda(cfg, gt_sp, gr_sp_mid);              // spine
    float lam_mid = combine_lambda(cfg, gt_mid, fminf(gr_sp_mid, gr_mid_ne));
    float lam_ne  = combine_lambda(cfg, gt_ne,  gr_mid_ne);

    float lam_shL = combine_lambda(cfg, gt_shL, gr_uL);
    float lam_elL = combine_lambda(cfg, gt_elL, fminf(gr_uL, gr_fL));
    float lam_wrL = combine_lambda(cfg, gt_wrL, gr_fL);

    float lam_shR = combine_lambda(cfg, gt_shR, gr_uR);
    float lam_elR = combine_lambda(cfg, gt_elR, fminf(gr_uR, gr_fR));
    float lam_wrR = combine_lambda(cfg, gt_wrR, gr_fR);

    // ---- EMA updates: apply on absolute spine, but for other joints either operate on relative offsets
    // or absolute positions depending on cfg.use_relative_pos_except_spine ----
    // spine absolute EMA
    f3 y_sp  = ema_update(st->ema_sp, p_sp, lam_sp);

    // Helper: decide whether to EMA on relative or absolute
    bool rel = (cfg.use_relative_pos_except_spine != 0);

    // mid: relative to spine or absolute
    f3 y_mid;
    if(rel){
        // EMA on rel_mid; reconstruct with y_sp
        f3 prev_rel_mid = f3_sub(st->prev_mid, st->prev_sp);
        EMA3Lite& e_mid = st->ema_mid;
        f3 cur_rel = rel_mid;
        // If no prev, ema_update will init
        f3 y_rel_mid = ema_update(e_mid, cur_rel, lam_mid);
        y_mid = f3_add(y_sp, y_rel_mid);
    } else {
        y_mid = ema_update(st->ema_mid, p_mid, lam_mid);
    }

    // neck: relative to mid or absolute
    f3 y_ne;
    if(rel){
        EMA3Lite& e_ne = st->ema_ne;
        f3 prev_rel_ne = f3_sub(st->prev_ne, st->prev_mid);
        f3 cur_rel = rel_ne;
        f3 y_rel_ne = ema_update(e_ne, cur_rel, lam_ne);
        y_ne = f3_add(y_mid, y_rel_ne);
    } else {
        y_ne = ema_update(st->ema_ne, p_ne, lam_ne);
    }

    // left arm chain: shoulder->elbow->wrist
    f3 y_shL;
    if(rel){
        EMA3Lite& e_shL = st->ema_shL;
        f3 y_rel_shL = ema_update(e_shL, rel_shL, lam_shL);
        y_shL = f3_add(y_ne, y_rel_shL); // shoulder relative to neck
    } else {
        y_shL = ema_update(st->ema_shL, p_shL, lam_shL);
    }

    f3 y_elL;
    if(rel){
        EMA3Lite& e_elL = st->ema_elL;
        f3 y_rel_elL = ema_update(e_elL, rel_elL, lam_elL);
        y_elL = f3_add(y_shL, y_rel_elL);
    } else {
        y_elL = ema_update(st->ema_elL, p_elL, lam_elL);
    }

    f3 y_wrL;
    if(rel){
        EMA3Lite& e_wrL = st->ema_wrL;
        f3 y_rel_wrL = ema_update(e_wrL, rel_wrL, lam_wrL);
        y_wrL = f3_add(y_elL, y_rel_wrL);
    } else {
        y_wrL = ema_update(st->ema_wrL, p_wrL, lam_wrL);
    }

    // right arm chain
    f3 y_shR;
    if(rel){
        EMA3Lite& e_shR = st->ema_shR;
        f3 y_rel_shR = ema_update(e_shR, rel_shR, lam_shR);
        y_shR = f3_add(y_ne, y_rel_shR);
    } else {
        y_shR = ema_update(st->ema_shR, p_shR, lam_shR);
    }

    f3 y_elR;
    if(rel){
        EMA3Lite& e_elR = st->ema_elR;
        f3 y_rel_elR = ema_update(e_elR, rel_elR, lam_elR);
        y_elR = f3_add(y_shR, y_rel_elR);
    } else {
        y_elR = ema_update(st->ema_elR, p_elR, lam_elR);
    }

    f3 y_wrR;
    if(rel){
        EMA3Lite& e_wrR = st->ema_wrR;
        f3 y_rel_wrR = ema_update(e_wrR, rel_wrR, lam_wrR);
        y_wrR = f3_add(y_elR, y_rel_wrR);
    } else {
        y_wrR = ema_update(st->ema_wrR, p_wrR, lam_wrR);
    }

    // ---- write outputs (same ordering) ----
    store_xyz(out, is,  y_sp);
    store_xyz(out, im,  y_mid);
    store_xyz(out, inx, y_ne);
    store_xyz(out, iSL, y_shL);
    store_xyz(out, iEL, y_elL);
    store_xyz(out, iWL, y_wrL);
    store_xyz(out, iSR, y_shR);
    store_xyz(out, iER, y_elR);
    store_xyz(out, iWR, y_wrR);

    // ---- update prev raw absolute positions (used to compute prev_rel as needed) ----
    st->prev_sp  = p_sp;  st->prev_mid = p_mid; st->prev_ne = p_ne;
    st->prev_shL = p_shL; st->prev_elL = p_elL; st->prev_wrL = p_wrL;
    st->prev_shR = p_shR; st->prev_elR = p_elR; st->prev_wrR = p_wrR;
    st->has_prev_raw = 1;

    // Update segment directions (unchanged)
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
