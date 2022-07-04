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

#include <cassert>
#include <cstring>
#include <vector>

#include "common.cuh"
#include "layer_norm_kernel.h"

namespace nvinfer1 {
namespace plugin {

template <typename T>
using kvp = cub::KeyValuePair<T, T>;

template <typename T>
struct mySum {
  __host__ __device__ __forceinline__ kvp<T> operator()(const kvp<T>& a, const kvp<T>& b) const {
    return kvp<T>(a.key + b.key, a.value + b.value);
  }
};

template <typename T, typename OP_T, int TPB>
__global__ void layer_norm_kernel_small(const int nHiddenDimension, const T* input, const T* gamma, const T* beta,
                                        T* output) {
  const int index = blockIdx.x * nHiddenDimension + threadIdx.x;
  const T denominator = T(1) / T(nHiddenDimension);
  OP_T val = 0;
  kvp<OP_T> threadData(0, 0);

  if (threadIdx.x < nHiddenDimension) {
    val = input[index] * denominator;
    OP_T tmp0 = val * (OP_T)denominator, tmp1 = val * tmp0;
    threadData = mySum<OP_T>()(threadData, kvp<OP_T>(tmp0, tmp1));
  }

  using WarpReduce = cub::WarpReduce<kvp<OP_T>, TPB>;
  __shared__ typename WarpReduce::TempStorage temp;
  __shared__ OP_T mu, rsigma;

  const auto sumKV = WarpReduce(temp).Reduce(threadData, mySum<OP_T>());

  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu);
  }
  __syncthreads();

  if (threadIdx.x < nHiddenDimension) {
    const OP_T g = gamma[threadIdx.x], b = beta[threadIdx.x];
    output[index] = (val - mu) * rsigma * g + b;
  }
}

template <typename T, typename OP_T, int TPB, int VPT>
__global__ void layer_norm_kernel_medium(const int nHiddenDimension, const T* input, const T* gamma, const T* beta,
                                         T* output) {
  // 考虑一个 block 上的寄存器使用量，当 nHiddenDimension 最大为 1024，元素为 float 时，
  // localX:      256 thread/block * 4 element/thread（即VPT） * 4 Byte/element = 4 KiB
  // localBeta:   1024 element / block * 4 Byte / element = 4 KiB
  // localGamma:  1024 element / block * 4 Byte / element = 4 KiB
  // localBias:   1024 element / block * 4 Byte / element = 4 KiB（这里没有）

  const int index = blockIdx.x * nHiddenDimension + threadIdx.x * VPT;
  T localX[VPT], localGamma[VPT], localBeta[VPT];
  const OP_T denominator = OP_T(1) / OP_T(nHiddenDimension);
  kvp<OP_T> threadData(0, 0);

  copy<sizeof(T) * VPT>(&input[index], localX);
#pragma unroll
  for (int it = 0; it < VPT; it++) {
    const OP_T tmp = (OP_T)localX[it] * denominator;
    threadData = mySum<OP_T>()(threadData, kvp<OP_T>(tmp, tmp * (OP_T)localX[it]));
  }

  copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], localBeta);
  copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], localGamma);

  using BlockReduce = cub::BlockReduce<kvp<OP_T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ OP_T mu, rsigma;

  const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, mySum<OP_T>());
  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu);
  }
  __syncthreads();

#pragma unroll
  for (int it = 0; it < VPT; it++) {
    localX[it] = (OP_T)localGamma[it] * ((OP_T)localX[it] - mu) * rsigma + (OP_T)localBeta[it];
  }

  copy<sizeof(T) * VPT>(localX, &output[index]);
}

template <typename T, typename OP_T, int TPB>
__global__ void layer_norm_kernel_large(const int nHiddenDimension, const T* input, const T* gamma, const T* beta,
                                        T* output) {
  const int offset = blockIdx.x * nHiddenDimension;
  const OP_T denominator = OP_T(1) / OP_T(nHiddenDimension);
  kvp<OP_T> threadData(0, 0);

  for (int i = threadIdx.x; i < nHiddenDimension; i += TPB) {
    const int index = offset + i;
    OP_T val = input[index];
    const OP_T tmp = val * denominator;
    threadData = mySum<OP_T>()(threadData, kvp<OP_T>(tmp, tmp * val));
    output[index] = val;
  }

  using BlockReduce = cub::BlockReduce<kvp<OP_T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ OP_T mu, rsigma;

  const auto sumKV = BlockReduce(temp).Reduce(threadData, mySum<OP_T>());

  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < nHiddenDimension; i += TPB) {
    const int index = offset + i;
    output[index] = ((OP_T)output[index] - mu) * rsigma * (OP_T)gamma[i] + (OP_T)beta[i];
  }
}

template <typename T>
int compute_layer_norm_tpl(cudaStream_t stream, const int gridSize, const int ld, const T* input, const T* gamma,
                           const T* beta, T* output) {
  constexpr int VPT = 16 / sizeof(T);

  if (ld <= 32) {
    constexpr int TPB = 32;
    (layer_norm_kernel_small<T, float, TPB>)<<<gridSize, TPB, 0, stream>>>(ld, input, gamma, beta, output);
  } else if (ld == 256) {
    constexpr int TPB = 256 / VPT;
    (layer_norm_kernel_medium<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(ld, input, gamma, beta, output);
  } else if (ld == 1024) {
    constexpr int TPB = 1024 / VPT;
    (layer_norm_kernel_medium<T, float, TPB, VPT>)<<<gridSize, TPB, 0, stream>>>(ld, input, gamma, beta, output);
  } else {
    constexpr int TPB = 256;
    (layer_norm_kernel_large<T, float, TPB>)<<<gridSize, TPB, 0, stream>>>(ld, input, gamma, beta, output);
  }

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

int compute_layer_norm(cudaStream_t stream, const int gridSize, const int ld, const float* input, const float* gamma,
                       const float* beta, float* output) {
  return compute_layer_norm_tpl<float>(stream, gridSize, ld, input, gamma, beta, output);
}

int compute_layer_norm(cudaStream_t stream, const int gridSize, const int ld, const half* input, const half* gamma,
                       const half* beta, half* output) {
  return compute_layer_norm_tpl<half>(stream, gridSize, ld, input, gamma, beta, output);
}

}  // namespace plugin
}  // namespace nvinfer1
