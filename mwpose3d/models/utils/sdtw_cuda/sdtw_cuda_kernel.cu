#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <cmath>

template <typename scalar_t>
__device__ __forceinline__ scalar_t neg_inf() {
  return -std::numeric_limits<scalar_t>::infinity();
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t pos_inf() {
  return  std::numeric_limits<scalar_t>::infinity();
}

// R: (B, N+2, M+2), D: (B, N, M)
// Flattened index helpers
__device__ __forceinline__
int idx_R(int b, int i, int j, int N, int M) {
  // i in [0..N+1], j in [0..M+1]
  return ((b*(N+2) + i) * (M+2) + j);
}

__device__ __forceinline__
int idx_D(int b, int i, int j, int N, int M) {
  // i in [0..N-1], j in [0..M-1]
  return ((b*N + i) * M + j);
}

// Forward: compute R via soft-DTW recurrence along anti-diagonals
template <typename scalar_t>
__global__
void softdtw_forward_kernel(
    const scalar_t* __restrict__ D,   // (B, N, M)
    scalar_t* __restrict__ R,         // (B, N+2, M+2) prefilled with +inf and R[b,0,0]=0
    const scalar_t gamma,
    const int bandwidth,  // 0 disables band
    const int B,
    const int N,
    const int M)
{
  const int b = blockIdx.x;                // one block per batch
  const int tid = threadIdx.x;
  const int tcount = blockDim.x;

  const scalar_t inv_gamma = scalar_t(1) / gamma;

  const int n_passes = N + M - 1;
  for (int p = 0; p < n_passes; ++p) {
    for (int ii = tid; ii < N; ii += tcount) {
      const int jj = p - ii;
      if (jj < 0 || jj >= M) continue;
      // padded indices
      const int i = ii + 1;
      const int j = jj + 1;

      if (bandwidth > 0) {
        const int diff = (i - j); // already 1-based inside padded R
        if (abs(diff) > bandwidth) continue;
      }

      const int r_ij   = idx_R(b, i,   j,   N, M);
      const int r_im1j = idx_R(b, i-1, j,   N, M);
      const int r_ijm1 = idx_R(b, i,   j-1, N, M);
      const int r_im1jm1 = idx_R(b, i-1, j-1, N, M);

      // NOTE: numerical stability like your Python version
      scalar_t r0 = -R[r_im1jm1] * inv_gamma;
      scalar_t r1 = -R[r_im1j]   * inv_gamma;
      scalar_t r2 = -R[r_ijm1]   * inv_gamma;

      // rmax trick
      scalar_t rmax = fmaxf(fmaxf(r0, r1), r2);
      // if rmax is -inf, exp terms will be 0; but that only happens if all are -inf.
      scalar_t e0 = expf(r0 - rmax);
      scalar_t e1 = expf(r1 - rmax);
      scalar_t e2 = expf(r2 - rmax);
      scalar_t rsum = e0 + e1 + e2;
      scalar_t softmin = -gamma * (logf(rsum) + rmax);

      const int d_ij = idx_D(b, ii, jj, N, M);
      R[r_ij] = D[d_ij] + softmin;
    }
    __syncthreads();
  }
}

// Backward: compute E by reverse wavefront
template <typename scalar_t>
__global__
void softdtw_backward_kernel(
    const scalar_t* __restrict__ Dpad,  // (B, N+2, M+2), D with interior filled, borders 0
    scalar_t* __restrict__ R,           // (B, N+2, M+2), already has boundaries set to -inf except R[-1,-1]=R[-2,-2]
    scalar_t* __restrict__ E,           // (B, N+2, M+2), initialized to 0 except E[-1,-1]=1
    const scalar_t inv_gamma,
    const int bandwidth,
    const int B,
    const int N,
    const int M)
{
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int tcount = blockDim.x;

  const int n_passes = N + M - 1;

  for (int p = 0; p < n_passes; ++p) {
    const int rev_p = n_passes - p - 1;

    for (int ii = tid; ii < N; ii += tcount) {
      const int jj = rev_p - ii;
      if (jj < 0 || jj >= M) continue;

      const int i = ii + 1;
      const int j = jj + 1;

      if (bandwidth > 0) {
        const int diff = (i - j);
        if (abs(diff) > bandwidth) continue;
      }

      const int r_ij      = idx_R(b, i,   j,   N, M);
      const int r_ip1j    = idx_R(b, i+1, j,   N, M);
      const int r_ijp1    = idx_R(b, i,   j+1, N, M);
      const int r_ip1jp1  = idx_R(b, i+1, j+1, N, M);

      // If R[i,j] is +inf (unreachable), keep it -inf to avoid nan
      if (isinf(R[r_ij]) && R[r_ij] > 0) {
        R[r_ij] = neg_inf<scalar_t>();
      }

      const int d_ip1j    = ((b*(N+2) + (i+1))*(M+2) + j);
      const int d_ijp1    = ((b*(N+2) + i)*(M+2) + (j+1));
      const int d_ip1jp1  = ((b*(N+2) + (i+1))*(M+2) + (j+1));

      // a,b,c as in your Python kernel
      scalar_t a = expf((R[r_ip1j]   - R[r_ij] - Dpad[d_ip1j])   * inv_gamma);
      scalar_t b_ = expf((R[r_ijp1]  - R[r_ij] - Dpad[d_ijp1])  * inv_gamma);
      scalar_t c = expf((R[r_ip1jp1] - R[r_ij] - Dpad[d_ip1jp1]) * inv_gamma);

      const int e_ij     = ((b*(N+2)+ i)*(M+2)+ j);
      const int e_ip1j   = ((b*(N+2)+ i+1)*(M+2)+ j);
      const int e_ijp1   = ((b*(N+2)+ i)*(M+2)+ j+1);
      const int e_ip1jp1 = ((b*(N+2)+ i+1)*(M+2)+ j+1);

      E[e_ij] = E[e_ip1j] * a + E[e_ijp1] * b_ + E[e_ip1jp1] * c;
    }
    __syncthreads();
  }
}

// Launcher helpers (called from C++)
template <typename scalar_t>
void launch_forward(
    const scalar_t* D, scalar_t* R,
    scalar_t gamma, int bandwidth,
    int B, int N, int M,
    cudaStream_t stream)
{
  const int threads = 256; // good default; diagonal work is strided anyway
  const dim3 grid(B);
  softdtw_forward_kernel<scalar_t><<<grid, threads, 0, stream>>>(
      D, R, gamma, bandwidth, B, N, M);
}

template <typename scalar_t>
void launch_backward(
    const scalar_t* Dpad,
    scalar_t* R,
    scalar_t* E,
    scalar_t inv_gamma,
    int bandwidth,
    int B, int N, int M,
    cudaStream_t stream)
{
  const int threads = 256;
  const dim3 grid(B);
  softdtw_backward_kernel<scalar_t><<<grid, threads, 0, stream>>>(
      Dpad, R, E, inv_gamma, bandwidth, B, N, M);
}

// Extern "C" wrappers used by the C++ binding
extern "C" {

void softdtw_forward_cuda_float(
    const float* D, float* R,
    float gamma, int bandwidth,
    int B, int N, int M,
    cudaStream_t stream) {
  launch_forward<float>(D, R, gamma, bandwidth, B, N, M, stream);
}

void softdtw_backward_cuda_float(
    const float* Dpad, float* R, float* E,
    float inv_gamma, int bandwidth,
    int B, int N, int M,
    cudaStream_t stream) {
  launch_backward<float>(Dpad, R, E, inv_gamma, bandwidth, B, N, M, stream);
}

}
