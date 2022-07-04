// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "att_stream_softmax_kernel.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cfloat>
#include <cub/cub.cuh>

#include "common.cuh"
#include "common.h"
/*#include "debug.h"*/

using namespace std;

namespace nvinfer1 {
namespace plugin {

template <typename T, unsigned TPB>
__device__ inline void AttStreamScaledSoftmaxSmall(const int ld, const int first_valid, const int last_valid,
                                                   const float rsqrt_head_size, const T* input, T* output) {
  if (threadIdx.x >= ld) {
    return;
  }
  using BlockReduce = cub::BlockReduce<float, TPB>;

  __shared__ typename BlockReduce::TempStorage tmpStorage;

  __shared__ float rZ;
  __shared__ float fMax;

  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;

  const float w(rsqrt_head_size);
  cub::Sum sum;
  float thread_data(-FLT_MAX);

  const int idx = offset + threadIdx.x;
  if (threadIdx.x >= first_valid && threadIdx.x < last_valid) {
    thread_data = input[idx];
  }

  /*if (blockIdx.x == 0 && threadIdx.x == 0) {*/
  /*printf("first_valid=%d, last_valid=%d input=%f, output=%f\n", first_valid, last_valid, __half2float(input[0]),
   * __half2float(output[0]));*/
  /*printf("in_ptr=%d, out_ptr = %d\n", input, output);*/
  /*}*/

  const float max_elem = BlockReduce(tmpStorage).Reduce(thread_data, cub::Max());
  if (threadIdx.x == 0) {
    fMax = max_elem;
  }
  __syncthreads();

  if (threadIdx.x >= first_valid && threadIdx.x < last_valid) {
    thread_data = exp((thread_data - fMax) * w);
  } else {
    thread_data = 0;
  }

  const auto Z = BlockReduce(tmpStorage).Reduce(thread_data, sum);

  if (threadIdx.x == 0) {
    rZ = (1.f) / Z;
  }
  __syncthreads();

  output[idx] = T(thread_data * rZ);
}

template <typename T, unsigned TPB>
__device__ inline void AttStreamScaledSoftmax(const int ld, const int first_valid, const int last_valid,
                                              const float rsqrt_head_size, const T* input, T* output) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  __shared__ float rZ;
  __shared__ float fMax;

  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;

  const float w(rsqrt_head_size);
  cub::Sum sum;
  float thread_data(-FLT_MAX);

  for (int i = threadIdx.x; i < last_valid; i += TPB) {
    if (i >= first_valid) {
      const int idx = offset + i;
      thread_data = max(static_cast<float>(input[idx]), thread_data);
    }
  }

  const float max_elem = BlockReduce(tmpStorage).Reduce(thread_data, cub::Max());
  if (threadIdx.x == 0) {
    fMax = max_elem;
  }
  __syncthreads();

  thread_data = 0;

  for (int i = threadIdx.x; i < last_valid; i += TPB) {
    if (i >= first_valid) {
      const int idx = offset + i;
      thread_data += exp((static_cast<float>(input[idx]) - fMax) * w);
    }
  }

  const auto Z = BlockReduce(tmpStorage).Reduce(thread_data, sum);

  if (threadIdx.x == 0) {
    rZ = 1.f / Z;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < last_valid; i += TPB) {
    const int idx = offset + i;
    const float val = (i >= first_valid) ? exp((static_cast<float>(input[idx]) - fMax) * w) * rZ : 0.f;
    output[idx] = T(val);
  }

  __syncthreads();
}

// mask_idx means decode frame num in right
template <typename T, unsigned TPB>
__global__ void AttStreamScaledSoftmaxKernelSmall(const int ld, const float rsqrt_head_size, const int cache_len,
                                                  const int* decode_frame_num, const int* mask_id, const T* input,
                                                  T* output) {
  __shared__ int first_valid, last_valid;
  /*if (blockIdx.x == 0 && threadIdx.x == 0) {*/
  /*printf("input=%f, output=%f\n", __half2float(input[0]),  __half2float(output[0]));*/
  /*printf("in_ptr=%d, out_ptr = %d\n", input, output);*/
  /*}*/

  if (threadIdx.x == 0) {
    first_valid = max(0, ld - decode_frame_num[blockIdx.y]);
    last_valid = min(ld, mask_id[blockIdx.y]) + cache_len;
  }
  __syncthreads();

  AttStreamScaledSoftmaxSmall<T, TPB>(ld, first_valid, last_valid, rsqrt_head_size, input, output);
}

template <typename T, unsigned TPB>
__global__ void AttStreamScaledSoftmaxKernel(const int ld, const float rsqrt_head_size, const int cache_len,
                                             const int* decode_frame_num, const int* mask_idx, const T* input,
                                             T* output) {
  __shared__ int first_valid, last_valid;

  if (threadIdx.x == 0) {
    first_valid = max(0, ld - decode_frame_num[blockIdx.y]);
    last_valid = min(ld, mask_idx[blockIdx.y]) + cache_len;
  }

  __syncthreads();
  AttStreamScaledSoftmax<T, TPB>(ld, first_valid, last_valid, rsqrt_head_size, input, output);
}

template <typename T>
int ComputeAttStreamScaledSoftmaxTpl(cudaStream_t stream, const int ld, const int B, const int N,
                                     const float rsqrt_head_size, const int cache_len, const int* decode_frame_num,
                                     const int* mask_idx, const T* input, T* output) {
  // Mask idx is of length B and assumes the valid region is contiguous starting
  // from the beginning of the sequence

  /*printf("ComputeAttStreamScaledSoftmaxTpl in_ptr=%d, out_ptr = %d\n", input, output);*/

  const dim3 grid(N, B, 1);
  // for smaller problems, e.g. BERT base B=1, this is not optimal
  if (ld <= 32) {
    constexpr int block_size = 32;
    AttStreamScaledSoftmaxKernelSmall<T, block_size>
        <<<grid, block_size, 0, stream>>>(ld, rsqrt_head_size, cache_len, decode_frame_num, mask_idx, input, output);
  } else if (ld <= 128) {
    constexpr int block_size = 128;
    AttStreamScaledSoftmaxKernelSmall<T, block_size><<<grid, block_size, 0, stream>>>(
        ld, rsqrt_head_size, cache_len, decode_frame_num, mask_idx, const_cast<T*>(input), output);
  } else {
    constexpr int block_size = 256;
    AttStreamScaledSoftmaxKernel<T, block_size>
        <<<grid, block_size, 0, stream>>>(ld, rsqrt_head_size, cache_len, decode_frame_num, mask_idx, input, output);
  }
  cudaDeviceSynchronize();

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

int ComputeAttStreamSoftmax(cudaStream_t stream, const int ld, const int B, const int N, const float rsqrt_head_size,
                            const int cache_len, const int* decode_frame_num, const int* mask_idx, const float* input,
                            float* output) {
  return ComputeAttStreamScaledSoftmaxTpl<float>(stream, ld, B, N, rsqrt_head_size, cache_len, decode_frame_num,
                                                 mask_idx, input, output);
}

int ComputeAttStreamSoftmax(cudaStream_t stream, const int ld, const int B, const int N, const float rsqrt_head_size,
                            int cache_len, const int* decode_frame_num, const int* mask_idx, const half* input,
                            half* output) {
  return ComputeAttStreamScaledSoftmaxTpl<half>(stream, ld, B, N, rsqrt_head_size, cache_len, decode_frame_num,
                                                mask_idx, input, output);
}

}  // namespace plugin
}  // namespace nvinfer1
