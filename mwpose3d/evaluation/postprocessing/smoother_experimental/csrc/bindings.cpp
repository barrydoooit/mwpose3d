#include <torch/extension.h>
#include "pose_smoother.h"
#include "gaussian_ema.h"
#include <pybind11/stl.h>

namespace py = pybind11;

// ---------------- ORIGINAL (unchanged) ----------------
struct PoseSmootherConfigHost {
    float dt;
    bool  use_spine_cv = true;
    bool  follow_raw_angle_when_unlocked = true;

    float shoulder_offset_lambda = 0.2f;
    float length_lambda = 0.05f;
    float spine_ema_lambda = 0.2f;
    float angle_ema_lambda = 0.2f;

    float spine_lock_radius_m = 0.03f;
    int   spine_lock_in_frames = 2;
    int   spine_lock_out_frames = 2;

    int   cv_restart_frames = 2;
    float cv_pos_violate_tol_m = 0.02f;

    float angle_tol_deg = 15.f;
    int   angle_unlock_out_frames = 2;
    int   angle_lock_in_frames = 1;

    ConfigDevice to_device() const {
        ConfigDevice c{};
        c.dt = dt;
        c.use_spine_cv = use_spine_cv ? 1 : 0;
        c.follow_raw_angle_when_unlocked = follow_raw_angle_when_unlocked ? 1 : 0;

        c.shoulder_offset_lambda = shoulder_offset_lambda;
        c.length_lambda = length_lambda;
        c.spine_ema_lambda = spine_ema_lambda;
        c.angle_ema_lambda = angle_ema_lambda;

        c.spine_lock_radius_m = spine_lock_radius_m;
        c.spine_lock_in_frames = spine_lock_in_frames;
        c.spine_lock_out_frames = spine_lock_out_frames;

        c.cv_restart_frames = cv_restart_frames;
        c.cv_pos_violate_tol_m = cv_pos_violate_tol_m;

        c.angle_tol_rad = angle_tol_deg * 3.14159265f / 180.f;
        c.angle_unlock_out_frames = angle_unlock_out_frames;
        c.angle_lock_in_frames = angle_lock_in_frames;
        return c;
    }
};

class PoseSmootherHost {
public:
    PoseSmootherHost(int N_joints, const PoseSmootherConfigHost& cfg)
    : N_(N_joints),
      idx_spine(0), idx_mid(1), idx_neck(2), idx_shL(3), idx_shR(4), idx_elL(5), idx_elR(6), idx_wrL(7), idx_wrR(8)
    {
        TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
        state_ = torch::empty({ (long long)sizeof(PoseSmootherState) },
                              torch::dtype(torch::kUInt8).device(torch::kCUDA));
        init_state_cuda(state_, N_, cfg.to_device());
        set_indices_all();
    }

    void set_indices(int a,int b,int c,int d,int e,int f,int g,int h,int i){
        check_idx(a); check_idx(b); check_idx(c); check_idx(d); check_idx(e); check_idx(f); check_idx(g); check_idx(h); check_idx(i);
        idx_spine=a; idx_mid=b; idx_neck=c; idx_shL=d; idx_shR=e; idx_elL=f; idx_elR=g; idx_wrL=h; idx_wrR=i;
        set_indices_all();
    }
    void set_indices_all(){
        set_indices_cuda(state_, idx_spine, idx_mid, idx_neck, idx_shL, idx_shR, idx_elL, idx_elR, idx_wrL, idx_wrR);
    }

    torch::Tensor filter(torch::Tensor frame_flat){
        TORCH_CHECK(frame_flat.is_cuda(), "input must be CUDA");
        TORCH_CHECK(frame_flat.dim()==1 && frame_flat.size(0)==N_*3,
                    "expected shape [N*3]=[%d], got %ld", N_*3, (long)frame_flat.size(0));

        torch::Tensor input32;
        bool was_half = (frame_flat.scalar_type() == torch::kFloat16);
        if (was_half) {
            input32 = frame_flat.to(torch::kFloat32);
        } else {
            TORCH_CHECK(frame_flat.dtype() == torch::kFloat32, "input must be float32 or float16");
            input32 = frame_flat;
        }
        torch::Tensor out32 = filter_cuda(state_, input32);
        if (was_half) return out32.to(torch::kFloat16);
        return out32;
    }

    torch::Tensor filter_sequence(torch::Tensor frames_Tx3N){
        TORCH_CHECK(frames_Tx3N.is_cuda(), "input must be CUDA");
        TORCH_CHECK(frames_Tx3N.dtype() == torch::kFloat32, "input must be float32");
        TORCH_CHECK(frames_Tx3N.dim()==2 && frames_Tx3N.size(1)==N_*3,
                    "expected shape [T, N*3] with N=%d; got %ld", N_, (long)frames_Tx3N.size(1));
        return filter_sequence_cuda(state_, frames_Tx3N);
    }

    void reset(){ reset_cuda(state_); }
    int N() const { return N_; }

private:
    void check_idx(int i) const {
        TORCH_CHECK(i>=0 && i<N_, "index %d out of range [0,%d)", i, N_);
    }

    int N_;
    torch::Tensor state_;

public:
    int idx_spine, idx_mid, idx_neck, idx_shL, idx_shR, idx_elL, idx_elR, idx_wrL, idx_wrR;
};

// ---------------- NEW Gaussian EMA smoother ----------------
struct GaussianEMAConfigHost {
    float dt;

    // translation params per type (meters)
    float mu_trans_spine = 0.0f;
    float delta_trans_spine = 0.02f;
    float mu_trans_mid = 0.0f;
    float delta_trans_mid = 0.02f;
    float mu_trans_neck = 0.0f;
    float delta_trans_neck = 0.02f;
    float mu_trans_shoulder = 0.0f;
    float delta_trans_shoulder = 0.02f;
    float mu_trans_elbow = 0.0f;
    float delta_trans_elbow = 0.02f;
    float mu_trans_wrist = 0.0f;
    float delta_trans_wrist = 0.02f;

    // rotation params per type (radians)
    float mu_rot_spine = 0.0f;
    float delta_rot_spine = 0.35f;
    float mu_rot_mid = 0.0f;
    float delta_rot_mid = 0.35f;
    float mu_rot_neck = 0.0f;
    float delta_rot_neck = 0.35f;
    float mu_rot_shoulder = 0.0f;
    float delta_rot_shoulder = 0.35f;
    float mu_rot_elbow = 0.0f;
    float delta_rot_elbow = 0.35f;
    float mu_rot_wrist = 0.0f;
    float delta_rot_wrist = 0.35f;

    // shared controls
    float lam_min = 0.05f;
    float lam_max = 0.95f;
    int   combine_rule = 0;
    float combine_alpha = 0.5f;

    bool  use_relative_pos_except_spine = true;

    GaussianEMAConfigDevice to_device() const {
        GaussianEMAConfigDevice c{};
        c.dt = dt;

        // copy translation params
        c.mu_trans[0] = mu_trans_spine;  c.delta_trans[0] = delta_trans_spine;
        c.mu_trans[1] = mu_trans_mid;    c.delta_trans[1] = delta_trans_mid;
        c.mu_trans[2] = mu_trans_neck;   c.delta_trans[2] = delta_trans_neck;
        c.mu_trans[3] = mu_trans_shoulder; c.delta_trans[3] = delta_trans_shoulder;
        c.mu_trans[4] = mu_trans_elbow;  c.delta_trans[4] = delta_trans_elbow;
        c.mu_trans[5] = mu_trans_wrist;  c.delta_trans[5] = delta_trans_wrist;

        // copy rotation params
        c.mu_rot[0] = mu_rot_spine;  c.delta_rot[0] = delta_rot_spine;
        c.mu_rot[1] = mu_rot_mid;    c.delta_rot[1] = delta_rot_mid;
        c.mu_rot[2] = mu_rot_neck;   c.delta_rot[2] = delta_rot_neck;
        c.mu_rot[3] = mu_rot_shoulder; c.delta_rot[3] = delta_rot_shoulder;
        c.mu_rot[4] = mu_rot_elbow;  c.delta_rot[4] = delta_rot_elbow;
        c.mu_rot[5] = mu_rot_wrist;  c.delta_rot[5] = delta_rot_wrist;

        c.lam_min = lam_min;
        c.lam_max = lam_max;
        c.combine_rule = combine_rule;
        c.combine_alpha = combine_alpha;
        c.use_relative_pos_except_spine = use_relative_pos_except_spine ? 1 : 0;
        return c;
    }
};

class PoseSmootherGaussianHost {
public:
    PoseSmootherGaussianHost(int N_joints, const GaussianEMAConfigHost& cfg)
    : N_(N_joints),
      idx_spine(0), idx_mid(1), idx_neck(2), idx_shL(3), idx_shR(4), idx_elL(5), idx_elR(6), idx_wrL(7), idx_wrR(8)
    {
        TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
        state_ = torch::empty({ (long long)sizeof(PoseSmootherGaussianState) },
                              torch::dtype(torch::kUInt8).device(torch::kCUDA));
        init_state_gaussian_cuda(state_, N_, cfg.to_device());
        set_indices_all();
    }

    void set_indices(int a,int b,int c,int d,int e,int f,int g,int h,int i){
        check_idx(a); check_idx(b); check_idx(c); check_idx(d); check_idx(e); check_idx(f); check_idx(g); check_idx(h); check_idx(i);
        idx_spine=a; idx_mid=b; idx_neck=c; idx_shL=d; idx_shR=e; idx_elL=f; idx_elR=g; idx_wrL=h; idx_wrR=i;
        set_indices_all();
    }
    void set_indices_all(){
        set_indices_gaussian_cuda(state_, idx_spine, idx_mid, idx_neck, idx_shL, idx_shR, idx_elL, idx_elR, idx_wrL, idx_wrR);
    }

    torch::Tensor filter(torch::Tensor frame_flat){
        TORCH_CHECK(frame_flat.is_cuda(), "input must be CUDA");
        TORCH_CHECK(frame_flat.dim()==1 && frame_flat.size(0)==N_*3,
                    "expected shape [N*3]=[%d], got %ld", N_*3, (long)frame_flat.size(0));

        torch::Tensor input32;
        bool was_half = (frame_flat.scalar_type() == torch::kFloat16);
        if (was_half) {
            input32 = frame_flat.to(torch::kFloat32);
        } else {
            TORCH_CHECK(frame_flat.dtype() == torch::kFloat32, "input must be float32 or float16");
            input32 = frame_flat;
        }
        torch::Tensor out32 = filter_gaussian_cuda(state_, input32);
        if (was_half) return out32.to(torch::kFloat16);
        return out32;
    }

    torch::Tensor filter_sequence(torch::Tensor frames_Tx3N){
        TORCH_CHECK(frames_Tx3N.is_cuda(), "input must be CUDA");
        TORCH_CHECK(frames_Tx3N.dtype() == torch::kFloat32, "input must be float32");
        TORCH_CHECK(frames_Tx3N.dim()==2 && frames_Tx3N.size(1)==N_*3,
                    "expected shape [T, N*3] with N=%d; got %ld", N_, (long)frames_Tx3N.size(1));
        return filter_sequence_gaussian_cuda(state_, frames_Tx3N);
    }

    void reset(){ reset_gaussian_cuda(state_); }
    int N() const { return N_; }

private:
    void check_idx(int i) const {
        TORCH_CHECK(i>=0 && i<N_, "index %d out of range [0,%d)", i, N_);
    }

    int N_;
    torch::Tensor state_;

public:
    int idx_spine, idx_mid, idx_neck, idx_shL, idx_shR, idx_elL, idx_elR, idx_wrL, idx_wrR;
};

PYBIND11_MODULE(cuda_pose_smoother, m) {
    // --------- Original "lock" smoother ---------
    py::class_<PoseSmootherConfigHost>(m, "PoseSmootherConfig")
        .def(py::init<float>(), py::arg("dt"))
        .def_readwrite("use_spine_cv", &PoseSmootherConfigHost::use_spine_cv)
        .def_readwrite("follow_raw_angle_when_unlocked", &PoseSmootherConfigHost::follow_raw_angle_when_unlocked)
        .def_readwrite("shoulder_offset_lambda", &PoseSmootherConfigHost::shoulder_offset_lambda)
        .def_readwrite("length_lambda", &PoseSmootherConfigHost::length_lambda)
        .def_readwrite("spine_ema_lambda", &PoseSmootherConfigHost::spine_ema_lambda)
        .def_readwrite("angle_ema_lambda", &PoseSmootherConfigHost::angle_ema_lambda)
        .def_readwrite("spine_lock_radius_m", &PoseSmootherConfigHost::spine_lock_radius_m)
        .def_readwrite("spine_lock_in_frames", &PoseSmootherConfigHost::spine_lock_in_frames)
        .def_readwrite("spine_lock_out_frames", &PoseSmootherConfigHost::spine_lock_out_frames)
        .def_readwrite("cv_restart_frames", &PoseSmootherConfigHost::cv_restart_frames)
        .def_readwrite("cv_pos_violate_tol_m", &PoseSmootherConfigHost::cv_pos_violate_tol_m)
        .def_readwrite("angle_tol_deg", &PoseSmootherConfigHost::angle_tol_deg)
        .def_readwrite("angle_unlock_out_frames", &PoseSmootherConfigHost::angle_unlock_out_frames)
        .def_readwrite("angle_lock_in_frames", &PoseSmootherConfigHost::angle_lock_in_frames)
    ;

    py::class_<PoseSmootherHost>(m, "PoseSmoother")
        .def(py::init<int, const PoseSmootherConfigHost&>(), py::arg("N_joints"), py::arg("cfg"))
        .def("set_indices", &PoseSmootherHost::set_indices,
             py::arg("idx_spine"), py::arg("idx_mid"), py::arg("idx_neck"),
             py::arg("idx_shL"), py::arg("idx_shR"),
             py::arg("idx_elL"), py::arg("idx_elR"),
             py::arg("idx_wrL"), py::arg("idx_wrR"))
        .def("filter", &PoseSmootherHost::filter, py::arg("frame_flat"))
        .def("filter_sequence", &PoseSmootherHost::filter_sequence, py::arg("frames_Tx3N"))
        .def("reset", &PoseSmootherHost::reset)
        .def_property_readonly("N", &PoseSmootherHost::N)
        // --- attribute-style index accessors via lambdas (no member getters/setters needed) ---
        .def_property("idx_spine",
            [](PoseSmootherHost& s){ return s.idx_spine; },
            [](PoseSmootherHost& s, int v){
                s.set_indices(v, s.idx_mid, s.idx_neck, s.idx_shL, s.idx_shR, s.idx_elL, s.idx_elR, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_mid",
            [](PoseSmootherHost& s){ return s.idx_mid; },
            [](PoseSmootherHost& s, int v){
                s.set_indices(s.idx_spine, v, s.idx_neck, s.idx_shL, s.idx_shR, s.idx_elL, s.idx_elR, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_neck",
            [](PoseSmootherHost& s){ return s.idx_neck; },
            [](PoseSmootherHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, v, s.idx_shL, s.idx_shR, s.idx_elL, s.idx_elR, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_shoulder_L",
            [](PoseSmootherHost& s){ return s.idx_shL; },
            [](PoseSmootherHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, s.idx_neck, v, s.idx_shR, s.idx_elL, s.idx_elR, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_shoulder_R",
            [](PoseSmootherHost& s){ return s.idx_shR; },
            [](PoseSmootherHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, s.idx_neck, s.idx_shL, v, s.idx_elL, s.idx_elR, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_elbow_L",
            [](PoseSmootherHost& s){ return s.idx_elL; },
            [](PoseSmootherHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, s.idx_neck, s.idx_shL, s.idx_shR, v, s.idx_elR, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_elbow_R",
            [](PoseSmootherHost& s){ return s.idx_elR; },
            [](PoseSmootherHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, s.idx_neck, s.idx_shL, s.idx_shR, s.idx_elL, v, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_wrist_L",
            [](PoseSmootherHost& s){ return s.idx_wrL; },
            [](PoseSmootherHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, s.idx_neck, s.idx_shL, s.idx_shR, s.idx_elL, s.idx_elR, v, s.idx_wrR);
            })
        .def_property("idx_wrist_R",
            [](PoseSmootherHost& s){ return s.idx_wrR; },
            [](PoseSmootherHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, s.idx_neck, s.idx_shL, s.idx_shR, s.idx_elL, s.idx_elR, s.idx_wrL, v);
            })
    ;

    // --------- Gaussian EMA smoother ---------
    py::class_<GaussianEMAConfigHost>(m, "GaussianEMASmootherConfig")
        .def(py::init<float>(), py::arg("dt"))
        // translation fields
        .def_readwrite("mu_trans_spine", &GaussianEMAConfigHost::mu_trans_spine)
        .def_readwrite("delta_trans_spine", &GaussianEMAConfigHost::delta_trans_spine)
        .def_readwrite("mu_trans_mid", &GaussianEMAConfigHost::mu_trans_mid)
        .def_readwrite("delta_trans_mid", &GaussianEMAConfigHost::delta_trans_mid)
        .def_readwrite("mu_trans_neck", &GaussianEMAConfigHost::mu_trans_neck)
        .def_readwrite("delta_trans_neck", &GaussianEMAConfigHost::delta_trans_neck)
        .def_readwrite("mu_trans_shoulder", &GaussianEMAConfigHost::mu_trans_shoulder)
        .def_readwrite("delta_trans_shoulder", &GaussianEMAConfigHost::delta_trans_shoulder)
        .def_readwrite("mu_trans_elbow", &GaussianEMAConfigHost::mu_trans_elbow)
        .def_readwrite("delta_trans_elbow", &GaussianEMAConfigHost::delta_trans_elbow)
        .def_readwrite("mu_trans_wrist", &GaussianEMAConfigHost::mu_trans_wrist)
        .def_readwrite("delta_trans_wrist", &GaussianEMAConfigHost::delta_trans_wrist)
        // rotation fields
        .def_readwrite("mu_rot_spine", &GaussianEMAConfigHost::mu_rot_spine)
        .def_readwrite("delta_rot_spine", &GaussianEMAConfigHost::delta_rot_spine)
        .def_readwrite("mu_rot_mid", &GaussianEMAConfigHost::mu_rot_mid)
        .def_readwrite("delta_rot_mid", &GaussianEMAConfigHost::delta_rot_mid)
        .def_readwrite("mu_rot_neck", &GaussianEMAConfigHost::mu_rot_neck)
        .def_readwrite("delta_rot_neck", &GaussianEMAConfigHost::delta_rot_neck)
        .def_readwrite("mu_rot_shoulder", &GaussianEMAConfigHost::mu_rot_shoulder)
        .def_readwrite("delta_rot_shoulder", &GaussianEMAConfigHost::delta_rot_shoulder)
        .def_readwrite("mu_rot_elbow", &GaussianEMAConfigHost::mu_rot_elbow)
        .def_readwrite("delta_rot_elbow", &GaussianEMAConfigHost::delta_rot_elbow)
        .def_readwrite("mu_rot_wrist", &GaussianEMAConfigHost::mu_rot_wrist)
        .def_readwrite("delta_rot_wrist", &GaussianEMAConfigHost::delta_rot_wrist)
        // shared
        .def_readwrite("lam_min", &GaussianEMAConfigHost::lam_min)
        .def_readwrite("lam_max", &GaussianEMAConfigHost::lam_max)
        .def_readwrite("combine_rule", &GaussianEMAConfigHost::combine_rule)
        .def_readwrite("combine_alpha", &GaussianEMAConfigHost::combine_alpha)
        .def_readwrite("use_relative_pos_except_spine", &GaussianEMAConfigHost::use_relative_pos_except_spine)
    ;


    py::class_<PoseSmootherGaussianHost>(m, "GaussianEMASmoother")
        .def(py::init<int, const GaussianEMAConfigHost&>(), py::arg("N_joints"), py::arg("cfg"))
        .def("set_indices", &PoseSmootherGaussianHost::set_indices,
             py::arg("idx_spine"), py::arg("idx_mid"), py::arg("idx_neck"),
             py::arg("idx_shL"), py::arg("idx_shR"),
             py::arg("idx_elL"), py::arg("idx_elR"),
             py::arg("idx_wrL"), py::arg("idx_wrR"))
        .def("filter", &PoseSmootherGaussianHost::filter, py::arg("frame_flat"))
        .def("filter_sequence", &PoseSmootherGaussianHost::filter_sequence, py::arg("frames_Tx3N"))
        .def("reset", &PoseSmootherGaussianHost::reset)
        .def_property_readonly("N", &PoseSmootherGaussianHost::N)
        // same lambda style for indices
        .def_property("idx_spine",
            [](PoseSmootherGaussianHost& s){ return s.idx_spine; },
            [](PoseSmootherGaussianHost& s, int v){
                s.set_indices(v, s.idx_mid, s.idx_neck, s.idx_shL, s.idx_shR, s.idx_elL, s.idx_elR, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_mid",
            [](PoseSmootherGaussianHost& s){ return s.idx_mid; },
            [](PoseSmootherGaussianHost& s, int v){
                s.set_indices(s.idx_spine, v, s.idx_neck, s.idx_shL, s.idx_shR, s.idx_elL, s.idx_elR, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_neck",
            [](PoseSmootherGaussianHost& s){ return s.idx_neck; },
            [](PoseSmootherGaussianHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, v, s.idx_shL, s.idx_shR, s.idx_elL, s.idx_elR, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_shoulder_L",
            [](PoseSmootherGaussianHost& s){ return s.idx_shL; },
            [](PoseSmootherGaussianHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, s.idx_neck, v, s.idx_shR, s.idx_elL, s.idx_elR, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_shoulder_R",
            [](PoseSmootherGaussianHost& s){ return s.idx_shR; },
            [](PoseSmootherGaussianHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, s.idx_neck, s.idx_shL, v, s.idx_elL, s.idx_elR, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_elbow_L",
            [](PoseSmootherGaussianHost& s){ return s.idx_elL; },
            [](PoseSmootherGaussianHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, s.idx_neck, s.idx_shL, s.idx_shR, v, s.idx_elR, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_elbow_R",
            [](PoseSmootherGaussianHost& s){ return s.idx_elR; },
            [](PoseSmootherGaussianHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, s.idx_neck, s.idx_shL, s.idx_shR, s.idx_elL, v, s.idx_wrL, s.idx_wrR);
            })
        .def_property("idx_wrist_L",
            [](PoseSmootherGaussianHost& s){ return s.idx_wrL; },
            [](PoseSmootherGaussianHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, s.idx_neck, s.idx_shL, s.idx_shR, s.idx_elL, s.idx_elR, v, s.idx_wrR);
            })
        .def_property("idx_wrist_R",
            [](PoseSmootherGaussianHost& s){ return s.idx_wrR; },
            [](PoseSmootherGaussianHost& s, int v){
                s.set_indices(s.idx_spine, s.idx_mid, s.idx_neck, s.idx_shL, s.idx_shR, s.idx_elL, s.idx_elR, s.idx_wrL, v);
            })
    ;
}