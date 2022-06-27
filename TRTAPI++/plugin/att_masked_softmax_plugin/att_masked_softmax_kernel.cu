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

#include "att_masked_softmax_kernel.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cfloat>
#include <cub/cub.cuh>

#include "common.cuh"
#include "common.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

template <typename T, int TPB, int VPT>
__global__ void masked_softmax(const float rsqrt_head_size, const T* input, T* output, const int* mask_idx) {
  using BlockReduce = cub::BlockReduce<float, TPB>;

  union SMem {
    T shm[VPT * TPB];
    typename BlockReduce::TempStorage reduce;
    SMem() {}
  };
  __shared__ SMem tmp;

  // grid: (NxS, B)
  const int b = blockIdx.y;
  const int block_offset = (b * gridDim.x + blockIdx.x) * TPB;
  __shared__ int last_valid;
  if (threadIdx.x == 0) {
    last_valid = min(TPB, mask_idx[b]);
  }
  __syncthreads();
  float local[VPT];

  __shared__ float rZ;
  __shared__ float fMax[VPT];

  const int idx = (block_offset + threadIdx.x) * VPT;
  T* myshm = &tmp.shm[threadIdx.x * VPT];
  copy<sizeof(T) * VPT>(&input[idx], myshm);

  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    local[it] = (threadIdx.x < last_valid) ? float(tmp.shm[it * TPB + threadIdx.x]) : -FLT_MAX;
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    float max_elem = BlockReduce(tmp.reduce).Reduce(local[it], cub::Max());
    if (threadIdx.x == 0) {
      fMax[it] = max_elem;
    }
    __syncthreads();
  }

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    local[it] = (threadIdx.x < last_valid) ? myExp<float>(rsqrt_head_size * (local[it] - fMax[it])) : 0.f;
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    const auto Z = BlockReduce(tmp.reduce).Reduce(local[it], cub::Sum());

    if (threadIdx.x == 0) {
      rZ = (1.f) / Z;
    }
    __syncthreads();
    local[it] *= rZ;
  }

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    tmp.shm[it * TPB + threadIdx.x] = local[it];
  }
  __syncthreads();
  copy<sizeof(T) * VPT>(myshm, &output[idx]);
}

template <typename T, int TPB, int VPT>
__global__ void softmax(const float rsqrt_head_size, const T* input, T* output) {
  float local[VPT];

  using BlockReduce = cub::BlockReduce<float, TPB>;

  union SMem {
    T shm[VPT * TPB];
    typename BlockReduce::TempStorage reduce;
    SMem() {}
  };
  __shared__ SMem tmp;

  __shared__ float rZ;
  __shared__ float fMax[VPT];

  const int idx = (TPB * blockIdx.x + threadIdx.x) * VPT;
  T* myshm = &tmp.shm[threadIdx.x * VPT];
  copy<sizeof(T) * VPT>(&input[idx], myshm);

  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    local[it] = float(tmp.shm[it * TPB + threadIdx.x]);
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    float max_elem = BlockReduce(tmp.reduce).Reduce(local[it], cub::Max());
    if (threadIdx.x == 0) {
      fMax[it] = max_elem;
    }
    __syncthreads();
  }

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    local[it] = myExp<float>(rsqrt_head_size * (local[it] - fMax[it]));
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    const auto Z = BlockReduce(tmp.reduce).Reduce(local[it], cub::Sum());

    if (threadIdx.x == 0) {
      rZ = 1.f / Z;
    }
    __syncthreads();
    local[it] *= rZ;
  }

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    tmp.shm[it * TPB + threadIdx.x] = local[it];
  }
  __syncthreads();
  copy<sizeof(T) * VPT>(myshm, &output[idx]);
}

template <typename T, unsigned TPB>
__global__ void scaled_softmax_kernel_small(const int ld, const float rsqrt_head_size, const T* input, T* output) {
  scaledSoftmaxSmall<T, TPB>(ld, ld, rsqrt_head_size, input, output);
}

template <typename T, unsigned TPB>
__global__ void scaled_softmax_kernel(const int ld, const float rsqrt_head_size, const T* input, T* output) {
  scaledSoftmax<T, TPB>(ld, ld, rsqrt_head_size, input, output);
}

template <typename T>
int compute_scaled_softmax(cudaStream_t stream, const int ld, const int B, const int N, const float rsqrt_head_size,
                           const T* input, T* output) {
  constexpr int VPT = 16 / sizeof(T);

  const dim3 grid(ld * N, B, 1);

  if (ld <= 32) {
    const int block_size = 32;
    scaled_softmax_kernel_small<T, block_size><<<grid, block_size, 0, stream>>>(ld, rsqrt_head_size, input, output);
  } else if (ld < 128) {
    const int block_size = 128;
    scaled_softmax_kernel_small<T, block_size><<<grid, block_size, 0, stream>>>(ld, rsqrt_head_size, input, output);
  } else if (ld == 128) {
    const int grid = B * N * ld / (VPT);
    softmax<T, 128, VPT><<<grid, 128, 0, stream>>>(rsqrt_head_size, input, output);
  } else if (ld == 384) {
    const int grid = B * N * ld / (VPT);
    softmax<T, 384, VPT><<<grid, 384, 0, stream>>>(rsqrt_head_size, input, output);
  } else {
    const int block_size = 256;

    scaled_softmax_kernel<T, block_size><<<grid, block_size, 0, stream>>>(ld, rsqrt_head_size, input, output);
  }

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

template <typename T, unsigned TPB>
__global__ void masked_scaled_softmax_kernel_small(const int ld, const float rsqrt_head_size, const int* mask_idx,
                                                   const T* input, T* output) {
  __shared__ int last_valid;

  if (threadIdx.x == 0) {
    last_valid = min(ld, mask_idx[blockIdx.y]);
  }
  __syncthreads();

  scaledSoftmaxSmall<T, TPB>(ld, last_valid, rsqrt_head_size, input, output);
}

template <typename T, unsigned TPB>
__global__ void masked_scaled_softmax_kernel(const int ld, const float rsqrt_head_size, const int* mask_idx,
                                             const T* input, T* output) {
  __shared__ int last_valid;

  if (threadIdx.x == 0) {
    last_valid = min(ld, mask_idx[blockIdx.y]);
  }
  __syncthreads();
  scaledSoftmax<T, TPB>(ld, last_valid, rsqrt_head_size, input, output);
}

template <typename T>
int compute_masked_scaled_softmax_tpl(cudaStream_t stream, const int ld, const int B, const int N,
                                      const float rsqrt_head_size, const int* mask_idx, const T* input, T* output) {
  // Mask idx is of length B and assumes the valid region is contiguous starting
  // from the beginning of the sequence

  const dim3 grid(ld * N, B, 1);
  // for smaller problems, e.g. BERT base B=1, this is not optimal
  if (ld <= 32) {
    constexpr int block_size = 32;
    masked_scaled_softmax_kernel_small<T, block_size>
        <<<grid, block_size, 0, stream>>>(ld, rsqrt_head_size, mask_idx, input, output);
  } else if (ld < 128) {
    constexpr int block_size = 128;
    masked_scaled_softmax_kernel_small<T, block_size>
        <<<grid, block_size, 0, stream>>>(ld, rsqrt_head_size, mask_idx, input, output);
  } else if (ld == 128) {
    if (B == 1) {
      constexpr int VPT = 4 / sizeof(T);
      constexpr int block_size = 128;
      const dim3 grid(ld * N / VPT, B, 1);
      masked_softmax<T, block_size, VPT><<<grid, block_size, 0, stream>>>(rsqrt_head_size, input, output, mask_idx);
    } else {
      constexpr int VPT = 16 / sizeof(T);
      constexpr int block_size = 128;
      const dim3 grid(ld * N / VPT, B, 1);
      masked_softmax<T, block_size, VPT><<<grid, block_size, 0, stream>>>(rsqrt_head_size, input, output, mask_idx);
    }
  } else if (ld == 384) {
    if (B == 1) {
      constexpr int VPT = 4 / sizeof(T);
      constexpr int block_size = 384;
      const dim3 grid(ld * N / VPT, B, 1);
      masked_softmax<T, block_size, VPT><<<grid, block_size, 0, stream>>>(rsqrt_head_size, input, output, mask_idx);
    } else {
      constexpr int VPT = 16 / sizeof(T);
      constexpr int block_size = 384;
      const dim3 grid(ld * N / VPT, B, 1);
      masked_softmax<T, block_size, VPT><<<grid, block_size, 0, stream>>>(rsqrt_head_size, input, output, mask_idx);
    }
  } else {
    constexpr int block_size = 256;
    masked_scaled_softmax_kernel<T, block_size>
        <<<grid, block_size, 0, stream>>>(ld, rsqrt_head_size, mask_idx, input, output);
  }

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

int compute_att_masked_softmax(cudaStream_t stream, const int ld, const int B, const int N, const float rsqrt_head_size,
                               const int* mask_idx, const float* input, float* output) {
  return compute_masked_scaled_softmax_tpl<float>(stream, ld, B, N, rsqrt_head_size, mask_idx, input, output);
}

int compute_att_masked_softmax(cudaStream_t stream, const int ld, const int B, const int N, const float rsqrt_head_size,
                               const int* mask_idx, const half* input, half* output) {
  return compute_masked_scaled_softmax_tpl<half>(stream, ld, B, N, rsqrt_head_size, mask_idx, input, output);
}

}  // namespace plugin
}  // namespace nvinfer1
