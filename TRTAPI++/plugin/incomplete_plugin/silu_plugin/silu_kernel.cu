#include <cassert>
#include <cstring>
#include <vector>

#include "common.cuh"
#include "silu_kernel.h"

namespace nvinfer1 {
namespace plugin {

template <typename T>
__global__ void silu_kernel(const T* input, const int N, T* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    output[idx] = input[idx] * sigmoid(input[idx]);
  }
}

template <typename T>
int compute_silu_tpl(const T* input, const int N, T* output, cudaStream_t stream) {

  constexpr int block_size = 256;
  const int grid_size = (N + block_size - 1) / block_size;
  silu_kernel<<<grid_size, block_size, 0, stream>>>(input, N, output);

  CUDA_CHECK(cudaPeekAtLastError());

  return 0;
}

int compute_silu(const float* input, const int N, float* output, cudaStream_t stream) {
  return compute_silu_tpl<float>(input, N, output, stream);
}

int compute_silu(const half* input, const int N, half* output, cudaStream_t stream) {
  return compute_silu_tpl<half>(input, N, output, stream);
}

}  // namespace plugin
}  // namespace nvinfer1
