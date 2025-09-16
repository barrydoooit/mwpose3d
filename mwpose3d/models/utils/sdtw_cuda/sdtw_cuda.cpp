// mwpose3d/models/utils/sdtw_cuda/sdtw_cuda.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <limits>

extern "C" {
void softdtw_forward_cuda_float(
    const float* D, float* R,
    float gamma, int bandwidth,
    int B, int N, int M,
    cudaStream_t stream);

void softdtw_backward_cuda_float(
    const float* Dpad, float* R, float* E,
    float inv_gamma, int bandwidth,
    int B, int N, int M,
    cudaStream_t stream);
}

namespace {

void check_cuda_float(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32.");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous.");
}

} // anonymous

// Forward: returns (costs, R)
//   D: (B,N,M) float32 CUDA
//   gamma: float
//   bandwidth: int (0 disables band)
std::tuple<at::Tensor, at::Tensor> softdtw_forward(
    const at::Tensor& D,
    const double gamma,
    const int64_t bandwidth)
{
  check_cuda_float(D, "D");
  TORCH_CHECK(D.dim() == 3, "D must be (B,N,M).");
  const int B = D.size(0);
  const int N = D.size(1);
  const int M = D.size(2);

  auto opts = D.options();
  // R padded (B, N+2, M+2)
  at::Tensor R = at::empty({B, N+2, M+2}, opts);
  R.fill_(std::numeric_limits<float>::infinity());
  // R[:,0,0] = 0
  R.index_put_({at::indexing::Slice(), 0, 0}, 0.0f);

  // launch
  auto stream = at::cuda::getCurrentCUDAStream();
  softdtw_forward_cuda_float(
      D.data_ptr<float>(),
      R.data_ptr<float>(),
      static_cast<float>(gamma),
      static_cast<int>(bandwidth),
      B, N, M,
      stream.stream());

  // cost = R[:, -2, -2]
  at::Tensor cost = R.index({at::indexing::Slice(), N, M}).contiguous(); // -2 == N, M in padded

  return {cost, R};
}

// Backward: returns E interior (B,N,M)
// Inputs:
//   D    : (B,N,M)
//   R    : (B,N+2,M+2) from forward
//   inv_gamma: 1/gamma
//   bandwidth: int
at::Tensor softdtw_backward(
    const at::Tensor& D,
    at::Tensor R,
    const double inv_gamma,
    const int64_t bandwidth)
{
  check_cuda_float(D, "D");
  check_cuda_float(R, "R");
  TORCH_CHECK(D.dim()==3, "D must be (B,N,M).");
  TORCH_CHECK(R.dim()==3, "R must be (B,N+2,M+2).");

  const int B = D.size(0);
  const int N = D.size(1);
  const int M = D.size(2);

  auto opts = D.options();

  // D padded (B, N+2, M+2)
  at::Tensor Dpad = at::zeros({B, N+2, M+2}, opts);
  Dpad.index_put_({at::indexing::Slice(), at::indexing::Slice(1, N+1), at::indexing::Slice(1, M+1)}, D);

  // Set R boundaries:
  // R[:,:,-1] = -inf; R[:,-1,:] = -inf; R[:,-1,-1] = R[:,-2,-2]
  const float ninf = -std::numeric_limits<float>::infinity();
  R.index_put_({at::indexing::Slice(), at::indexing::Slice(), M+1}, ninf);
  R.index_put_({at::indexing::Slice(), N+1, at::indexing::Slice()}, ninf);
  auto tail = R.index({at::indexing::Slice(), N, M}).contiguous();
  R.index_put_({at::indexing::Slice(), N+1, M+1}, tail);

  // E (B, N+2, M+2) zeros; E[:,-1,-1] = 1
  at::Tensor E = at::zeros({B, N+2, M+2}, opts);
  E.index_put_({at::indexing::Slice(), N+1, M+1}, 1.0f);

  // launch
  auto stream = at::cuda::getCurrentCUDAStream();
  softdtw_backward_cuda_float(
      Dpad.data_ptr<float>(),
      R.data_ptr<float>(),
      E.data_ptr<float>(),
      static_cast<float>(inv_gamma),
      static_cast<int>(bandwidth),
      B, N, M,
      stream.stream());

  // interior E[:,1:N+1,1:M+1]
  at::Tensor E_interior = E.index({at::indexing::Slice(),
                                   at::indexing::Slice(1, N+1),
                                   at::indexing::Slice(1, M+1)}).contiguous();
  return E_interior;
}

TORCH_LIBRARY(sdtw_cuda, m) {
  m.def("forward(Tensor D, float gamma, int bandwidth) -> (Tensor, Tensor)");
  m.def("backward(Tensor D, Tensor R, float inv_gamma, int bandwidth) -> Tensor");
}

TORCH_LIBRARY_IMPL(sdtw_cuda, CUDA, m) {
  m.impl("forward", softdtw_forward);
  m.impl("backward", softdtw_backward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &softdtw_forward,
        "SoftDTW forward (returns (cost, R))",
        py::arg("D"), py::arg("gamma"), py::arg("bandwidth"));
  m.def("backward", &softdtw_backward,
        "SoftDTW backward (returns E interior)",
        py::arg("D"), py::arg("R"), py::arg("inv_gamma"), py::arg("bandwidth"));
}