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
#include "common.h"

#include "fmoe_expert_kernel.h"

namespace nvinfer1 {
namespace plugin {

__global__ void ScatterMappingKernel(const int* gate_idx, const int num_expert, const int idx_num, int* mapping,
                                     int* acc_histogram) {
  int idx = threadIdx.x;
  extern __shared__ int his[];
  if (idx < num_expert + 1) his[idx] = 0;

  __syncthreads();

  for (int i = threadIdx.x; i < idx_num; i += blockDim.x) {
    // calc his
    auto old = atomicAdd(&his[gate_idx[i] + 1], 1);
    mapping[i] = old;
  }

  __syncthreads();

  /*// print*/
  /*if (threadIdx.x == 0) {*/
  /*for (int i = 0; i < idx_num; i++) */
  /*printf("%d ", mapping[i]);*/
  /*printf("\n");*/

  /*for (int i = 0; i < num_expert; i++) */
  /*printf("%d ", his[i]);*/
  /*printf("\n");*/
  /*}*/
  /*__syncthreads();*/

  // acc his
  if (threadIdx.x == 0) {
    for (int i = 0; i < num_expert; i++) his[i + 1] += his[i];
  }
  __syncthreads();

  /*// print*/
  /*if (threadIdx.x == 0) {*/
  /*for (int i = 0; i < num_expert; i++) */
  /*printf("%d ", his[i]);*/
  /*printf("\n");*/
  /*}*/
  /*__syncthreads();*/

  for (int i = threadIdx.x; i < idx_num; i += blockDim.x) {
    // calc his
    mapping[i] += his[gate_idx[i]];
  }

  if (idx < num_expert + 1) acc_histogram[idx] = his[idx];
}

int ComputeScatterMapping(const int* gate_idx, const int num_expert, const int idx_num, int* mapping,
                          int* acc_histogram, cudaStream_t stream) {
  int block_size = 0;
  if (idx_num < 1024)
    block_size = 256;
  else if (idx_num < 4096)
    block_size = 512;
  else
    block_size = 1024;

  ScatterMappingKernel<<<1, block_size, (num_expert + 1) * sizeof(int), stream>>>(gate_idx, num_expert, idx_num,
                                                                                  mapping, acc_histogram);

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

template <class T>
__global__ void ScatterMappingCopyKernel(const T* input, const int* mapping, const int dim, const int numel,
                                         T* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;

  int s = idx / dim;
  int i = idx % dim;

  int mapping_idx = mapping[s];

  output[mapping_idx * dim + i] = input[idx];
}

template <class T>
int ComputeScatterMappingCopyTpl(const T* input, const int* mapping, const int S, const int dim, T* output,
                                 cudaStream_t stream) {
  auto numel = S * dim;

  int block_size = 256;
  int grid_size = (numel + block_size - 1) / block_size;

  ScatterMappingCopyKernel<T><<<grid_size, block_size, 0, stream>>>(input, mapping, dim, numel, output);

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

int ComputeScatterMappingCopy(const float* input, const int* mapping, const int S, const int dim, float* output,
                              cudaStream_t stream) {
  return ComputeScatterMappingCopyTpl(input, mapping, S, dim, output, stream);
}

int ComputeScatterMappingCopy(const half* input, const int* mapping, const int S, const int dim, half* output,
                              cudaStream_t stream) {
  return ComputeScatterMappingCopyTpl(input, mapping, S, dim, output, stream);
}

template <typename T>
__global__ void BiasSiluKernel(const T* input, const T* bias, const int N, const int dim, T* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    int bias_idx = idx % dim;
    auto tmp = input[idx] + bias[bias_idx];
    output[idx] = tmp * sigmoid(tmp);
  }
}

template <typename T>
int ComputeBiasSiluTpl(const T* input, const T* bias, const int N, const int dim, T* output, cudaStream_t stream) {
  constexpr int block_size = 512;
  const int grid_size = (N + block_size - 1) / block_size;
  BiasSiluKernel<T><<<grid_size, block_size, 0, stream>>>(input, bias, N, dim, output);

  CUDA_CHECK(cudaPeekAtLastError());

  return 0;
}

int ComputeBiasSilu(const float* input, const float* bias, const int N, const int dim, float* output,
                    cudaStream_t stream) {
  return ComputeBiasSiluTpl<float>(input, bias, N, dim, output, stream);
}

int ComputeBiasSilu(const half* input, const half* bias, const int N, const int dim, half* output,
                    cudaStream_t stream) {
  return ComputeBiasSiluTpl<half>(input, bias, N, dim, output, stream);
}

template <typename T>
__global__ void BiasKernel(const T* input, const T* bias, const int N, const int dim, T* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    int bias_idx = idx % dim;
    output[idx] = input[idx] + bias[bias_idx];
  }
}

template <typename T>
int ComputeBiasTpl(const T* input, const T* bias, const int N, const int dim, T* output, cudaStream_t stream) {
  constexpr int block_size = 512;
  const int grid_size = (N + block_size - 1) / block_size;
  BiasKernel<T><<<grid_size, block_size, 0, stream>>>(input, bias, N, dim, output);

  CUDA_CHECK(cudaPeekAtLastError());

  return 0;
}

int ComputeBias(const float* input, const float* bias, const int N, const int dim, float* output, cudaStream_t stream) {
  return ComputeBiasTpl<float>(input, bias, N, dim, output, stream);
}

int ComputeBias(const half* input, const half* bias, const int N, const int dim, half* output, cudaStream_t stream) {
  return ComputeBiasTpl<half>(input, bias, N, dim, output, stream);
}

template <class T>
__global__ void GatherrMappingCopyKernel(const T* input, const int* mapping, const int dim, const int numel,
                                         T* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;

  int s = idx / dim;
  int i = idx % dim;

  int mapping_idx = mapping[s];

  output[idx] = input[mapping_idx * dim + i];
}

template <class T>
int ComputeGatherMappingCopyTpl(const T* input, const int* mapping, const int S, const int dim, T* output,
                                cudaStream_t stream) {
  auto numel = S * dim;

  int block_size = 256;
  int grid_size = (numel + block_size - 1) / block_size;

  GatherrMappingCopyKernel<T><<<grid_size, block_size, 0, stream>>>(input, mapping, dim, numel, output);

  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

int ComputeGatherrMappingCopy(const float* input, const int* mapping, const int S, const int dim, float* output,
                              cudaStream_t stream) {
  return ComputeGatherMappingCopyTpl(input, mapping, S, dim, output, stream);
}

int ComputeGatherrMappingCopy(const half* input, const int* mapping, const int S, const int dim, half* output,
                              cudaStream_t stream) {
  return ComputeGatherMappingCopyTpl(input, mapping, S, dim, output, stream);
}

}  // namespace plugin
}  // namespace nvinfer1
