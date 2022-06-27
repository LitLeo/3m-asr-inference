#include "cat_split_cache_kernel.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cfloat>
#include <cub/cub.cuh>

#include "common.cuh"
#include "common.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// when input_dim > cache_dim
template <typename T>
__global__ void CatSplitCacheKernel(const int B, const int cache_dim, const int input_dim, const T* in_cache,
                                    const T* input, T* output, T* out_cache) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * input_dim) {
    auto batch_idx = idx / input_dim;
    auto input_ptr = input + batch_idx * input_dim;
    auto output_ptr = output + batch_idx * (input_dim + cache_dim);

    int offset = idx % input_dim;
    // copy input to output
    output_ptr[offset + cache_dim] = input_ptr[offset];
    if (offset < cache_dim) {
      // copy cache to output
      auto in_cache_ptr = in_cache + batch_idx * cache_dim;
      output_ptr[offset] = in_cache_ptr[offset];

      // copy part input to out_cache
      auto out_cache_ptr = out_cache + batch_idx * cache_dim;
      auto input_ptr_of_cache_ptr = input_ptr + input_dim - cache_dim;
      out_cache_ptr[offset] = input_ptr_of_cache_ptr[offset];
    }
  }
}

// cat tow matrix
// the larger_dim is the max(input0_dim, input1_dim)
template <typename T>
__global__ void Cat2Kernel(const int B, const int input0_dim, const int input1_dim, const int larger_dim,
                           const T* input0, const T* input1, T* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * larger_dim) {
    auto batch_idx = idx / larger_dim;
    auto input0_ptr = input0 + batch_idx * input0_dim;
    auto input1_ptr = input1 + batch_idx * input1_dim;
    auto output_ptr = output + batch_idx * (input0_dim + input1_dim);

    int offset = idx % larger_dim;

    if (offset < input0_dim) output_ptr[offset] = input0_ptr[offset];

    if (offset < input1_dim) output_ptr[input0_dim + offset] = input1_ptr[offset];
  }
}

// split cache from input, input_dim > cache_dim
template <typename T>
__global__ void SplitCacheKernel(const int B, const int input_dim, const int cache_dim, const T* input, T* cache) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * cache_dim) {
    auto batch_idx = idx / cache_dim;
    auto input_ptr = input + batch_idx * input_dim + input_dim - cache_dim;
    auto cache_ptr = cache + batch_idx * cache_dim;

    int offset = idx % cache_dim;

    if (offset < cache_dim) cache_ptr[offset] = input_ptr[offset];
  }
}

template <typename T>
int ComputeCatSplitCacheTpl(cudaStream_t stream, const int batch, const int cache_dim, const int input_dim,
                            const T* in_cache, const T* input, T* output, T* out_cache) {
  if (input_dim >= cache_dim) {
    int block_size = 64;
    int grid_size = (batch * input_dim + block_size - 1) / block_size;
    CatSplitCacheKernel<T>
        <<<grid_size, block_size, 0, stream>>>(batch, cache_dim, input_dim, in_cache, input, output, out_cache);
  } else {
    int block_size = 64;
    int grid_size = (batch * cache_dim + block_size - 1) / block_size;

    // cat cache, input => output
    Cat2Kernel<T>
        <<<grid_size, block_size, 0, stream>>>(batch, cache_dim, input_dim, cache_dim, in_cache, input, output);

    // split cache from output
    SplitCacheKernel<T>
        <<<grid_size, block_size, 0, stream>>>(batch, cache_dim + input_dim, cache_dim, output, out_cache);

    // TODO(leowgyang):
    // there two plans to merge above 2 kernels
    // (1) 1 block process 1 line and sync
    // (2) copy output_cache from in_cache and input
  }

  auto ret = cudaPeekAtLastError();
  if (ret != cudaSuccess) {
    gLogError << "ComputeCatSplitCacheTpl failed! status = " << cudaGetErrorString(ret) << endl;
    return -1;
  }
  return 0;
}

int ComputeCatSplitCache(cudaStream_t stream, const int batch, const int cache_dim, const int input_dim,
                         const float* in_cache, const float* input, float* output, float* out_cache) {
  return ComputeCatSplitCacheTpl<float>(stream, batch, cache_dim, input_dim, in_cache, input, output, out_cache);
}

int ComputeCatSplitCache(cudaStream_t stream, const int batch, const int cache_dim, const int input_dim,
                         const half* in_cache, const half* input, half* output, half* out_cache) {
  return ComputeCatSplitCacheTpl<half>(stream, batch, cache_dim, input_dim, in_cache, input, output, out_cache);
}

}  // namespace plugin
}  // namespace nvinfer1
