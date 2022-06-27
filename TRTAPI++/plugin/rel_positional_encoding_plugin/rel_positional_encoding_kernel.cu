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

#include "rel_positional_encoding_kernel.h"

#include <cuda_fp16.h>
#include <algorithm>

#include "common.cuh"
#include "common.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

template <typename T>
__global__ void scale_kernel(const T* input, T scale, int n, T* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) output[idx] = input[idx] * scale;
}

int compute_scale(const float* input, float scale, int n, float* output, cudaStream_t stream) {
  int block_size = 512;
  int grid_size = (n + block_size - 1) / block_size;

  scale_kernel<float><<<grid_size, block_size, 0, stream>>>(input, scale, n, output);
  auto ret = cudaPeekAtLastError();
  if (ret != cudaSuccess) {
    gLogError << "scale_kernel ailed! status = " << cudaGetErrorString(ret) << endl;
    return -1;
  }
  return 0;
}

int compute_scale(const half* input, half scale, int n, half* output, cudaStream_t stream) {
  int block_size = 512;
  int grid_size = (n + block_size - 1) / block_size;

  scale_kernel<half><<<grid_size, block_size, 0, stream>>>(input, scale, n, output);
  auto ret = cudaPeekAtLastError();
  if (ret != cudaSuccess) {
    gLogError << "scale_kernel ailed! status = " << cudaGetErrorString(ret) << endl;
    return -1;
  }
  return 0;
}

// x = x * self.xscale [batch, seq_len, dim]
// pos_emb = self.pe[:, 0:seq_len] [seq_len, dim]
template <typename T>
__global__ void rel_positional_encoding_kernel(const T* input, const T* pe, const T scale, const int x_size,
                                               const int pos_size, T* output, T* pos_emb) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < x_size) output[idx] = input[idx] * scale;

  if (idx < pos_size) pos_emb[idx] = pe[idx];
}

int compute_rel_positional_encoding(const float* input, const float* pe, const float scale, const int batch,
                                    const int seq_len, const int dim, float* output, float* pos_emb,
                                    cudaStream_t stream) {
  int pos_size = seq_len * dim;
  int x_size = batch * pos_size;

  int block_size = 512;
  int grid_size = (x_size + block_size - 1) / block_size;

  rel_positional_encoding_kernel<float>
      <<<grid_size, block_size, 0, stream>>>(input, pe, scale, x_size, pos_size, output, pos_emb);
  auto ret = cudaPeekAtLastError();
  if (ret != cudaSuccess) {
    gLogError << "rel_positional_encoding_kernel ailed! status = " << cudaGetErrorString(ret) << endl;
    return -1;
  }
  return 0;
}

int compute_rel_positional_encoding(const half* input, const half* pe, const half scale, const int batch,
                                    const int seq_len, const int dim, half* output, half* pos_emb,
                                    cudaStream_t stream) {
  int pos_size = seq_len * dim;
  int x_size = batch * pos_size;

  int block_size = 512;
  int grid_size = (x_size + block_size - 1) / block_size;

  rel_positional_encoding_kernel<half>
      <<<grid_size, block_size, 0, stream>>>(input, pe, scale, x_size, pos_size, output, pos_emb);
  auto ret = cudaPeekAtLastError();
  if (ret != cudaSuccess) {
    gLogError << "rel_positional_encoding_kernel ailed! status = " << cudaGetErrorString(ret) << endl;
    return -1;
  }
  return 0;
}

// x = x * self.xscale [batch, seq_len, dim]
// pos_emb = self.pe[:, offset:offect+seq_len] [seq_len, dim]
// frame_num_input = frame_num_input + seq_len [B]
template <typename T>
__global__ void rel_positional_encoding_streaming_kernel(const T* input, const T* pe, const int* frame_num_input,
                                                         const T scale, const int batch, const int seq_len,
                                                         const int dim, T* output, T* pos_emb) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int pos_size = seq_len * dim;
  int x_size = pos_size * batch;

  if (idx < x_size) output[idx] = input[idx] * scale;

  if (idx < pos_size) pos_emb[idx] = pe[idx];
}

int compute_rel_positional_encoding_streaming(const float* input, const float* pe, const int* frame_num_input,
                                              const float scale, const int batch, const int seq_len, const int dim,
                                              float* output, float* pos_emb, cudaStream_t stream) {
  int x_size = batch * seq_len * dim;

  int block_size = 512;
  int grid_size = (x_size + block_size - 1) / block_size;

  rel_positional_encoding_streaming_kernel<float>
      <<<grid_size, block_size, 0, stream>>>(input, pe, frame_num_input, scale, batch, seq_len, dim, output, pos_emb);
  auto ret = cudaPeekAtLastError();
  if (ret != cudaSuccess) {
    gLogError << "rel_positional_encoding_streaming_kernelf ailed! status = " << cudaGetErrorString(ret) << endl;
    return -1;
  }
  return 0;
}

int compute_rel_positional_encoding_streaming(const half* input, const half* pe, const int* frame_num_input,
                                              const half scale, const int batch, const int seq_len, const int dim,
                                              half* output, half* pos_emb, cudaStream_t stream) {
  int x_size = batch * seq_len * dim;

  int block_size = 512;
  int grid_size = (x_size + block_size - 1) / block_size;

  rel_positional_encoding_streaming_kernel<half>
      <<<grid_size, block_size, 0, stream>>>(input, pe, frame_num_input, scale, batch, seq_len, dim, output, pos_emb);
  auto ret = cudaPeekAtLastError();
  if (ret != cudaSuccess) {
    gLogError << "rel_positional_encoding_streaming_kernelf ailed! status = " << cudaGetErrorString(ret) << endl;
    return -1;
  }
  return 0;
}

}  // namespace plugin
}  // namespace nvinfer1
