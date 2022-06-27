#include "left_padding_cache_plugin.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cfloat>
#include <cub/cub.cuh>

#include "common.cuh"
#include "common.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// input_len >= cache_len
template <typename T, unsigned TPB>
__global__ void LeftPaddingCacheSmallKernel(const int input_len, const int cache_len, const int dim, const T* input,
                                            const T* in_cache, T* output, T* out_cache) {
  if (threadIdx.x < dim) {
    const int input_batch_offset = blockIdx.y * input_len * dim;
    const int cache_batch_offset = blockIdx.y * cache_len * dim;

    // the offset of cache and input for output
    auto output0_ptr = output + input_batch_offset + cache_batch_offset;
    auto output1_ptr = output0_ptr + cache_len * dim;

    auto input_ptr = input + input_batch_offset;
    auto in_cache_ptr = in_cache + cache_batch_offset;

    auto offset = blockIdx.x * dim + threadIdx.x;

    // copy cache
    if (blockIdx.x < cache_len) output0_ptr[offset] = in_cache_ptr[offset];

    // copy input
    if (blockIdx.x < input_len) output1_ptr[offset] = input_ptr[offset];

    // copy part input to out_cache
    if (blockIdx.x < cache_len) {
      auto input_ptr_of_cache_ptr = input_ptr + (input_len - cache_len) * dim;
      auto out_cache_ptr = out_cache + cache_batch_offset;
      out_cache_ptr[offset] = input_ptr_of_cache_ptr[offset];
    }
  }
}

template <typename T, unsigned TPB>
__global__ void LeftPaddingCacheKernel(const int input_len, const int cache_len, const int dim, const T* input,
                                       const T* in_cache, T* output, T* out_cache) {
  if (threadIdx.x < dim) {
    const int input_batch_offset = blockIdx.y * input_len * dim;
    const int cache_batch_offset = blockIdx.y * cache_len * dim;
    // the offset of cache and input for output
    auto output0_ptr = output + input_batch_offset + cache_batch_offset;
    auto output1_ptr = output0_ptr + cache_len * dim;

    auto input_ptr = input + input_batch_offset;
    auto in_cache_ptr = in_cache + cache_batch_offset;

    auto offset = blockIdx.x * dim;

    // copy cache to output
    if (blockIdx.x < cache_len) {
      for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output0_ptr[offset + i] = in_cache_ptr[offset + i];
      }
    }

    // copy input to output
    if (blockIdx.x < input_len) {
      for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output1_ptr[offset + i] = input_ptr[offset + i];
      }
    }

    // copy part input to out_cache
    if (blockIdx.x < cache_len) {
      auto input_ptr_of_cache_ptr = input_ptr + (input_len - cache_len) * dim;
      auto out_cache_ptr = out_cache + cache_batch_offset;
      for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out_cache_ptr[offset + i] = input_ptr_of_cache_ptr[offset + i];
    }
  }
}

template <typename T>
int ComputeLeftPaddingCacheTpl(cudaStream_t stream, const int batch, const int input_len, const int cache_len,
                               const int dim, const T* input, const T* in_cache, T* output, T* out_cache) {
  auto N = max(cache_len, input_len);
  const dim3 grid(N, batch, 1);
  // for smaller problems, e.g. BERT base B=1, this is not optimal
  if (dim <= 32) {
    constexpr int block_size = 32;
    LeftPaddingCacheSmallKernel<T, block_size>
        <<<grid, block_size, 0, stream>>>(input_len, cache_len, dim, input, in_cache, output, out_cache);
  } else if (dim < 128) {
    constexpr int block_size = 128;
    LeftPaddingCacheSmallKernel<T, block_size>
        <<<grid, block_size, 0, stream>>>(input_len, cache_len, dim, input, in_cache, output, out_cache);
  } else if (dim < 256) {
    constexpr int block_size = 256;
    LeftPaddingCacheSmallKernel<T, block_size>
        <<<grid, block_size, 0, stream>>>(input_len, cache_len, dim, input, in_cache, output, out_cache);
  } else {
    constexpr int block_size = 256;
    LeftPaddingCacheKernel<T, block_size>
        <<<grid, block_size, 0, stream>>>(input_len, cache_len, dim, input, in_cache, output, out_cache);
  }

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

int ComputeLeftPaddingCache(cudaStream_t stream, const int batch, const int input_len, const int cache_len,
                            const int dim, const float* input, const float* in_cache, float* output, float* out_cache) {
  if (input_len > cache_len) {
    return ComputeLeftPaddingCacheTpl<float>(stream, batch, input_len, cache_len, dim, input, in_cache, output,
                                             out_cache);
  } else {
    assert(0);
  }
  return 0;
}

int ComputeLeftPaddingCache(cudaStream_t stream, const int batch, const int input_len, const int cache_len,
                            const int dim, const half* input, const half* in_cache, half* output, half* out_cache) {
  if (input_len > cache_len) {
    return ComputeLeftPaddingCacheTpl<half>(stream, batch, input_len, cache_len, dim, input, in_cache, output,
                                            out_cache);
  } else {
    assert(0);
  }
  return 0;
}

}  // namespace plugin
}  // namespace nvinfer1
