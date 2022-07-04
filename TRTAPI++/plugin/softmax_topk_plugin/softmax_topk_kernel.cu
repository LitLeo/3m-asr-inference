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

#include "softmax_topk_kernel.h"

#include <cuda_fp16.h>
#include <algorithm>

#include "common.cuh"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// 先按topk=1实现
template <typename T, unsigned TPB>
__global__ void SoftmaxAndTop1KernelSmall(int seq_len, int width, int ld, const int* mask, const T* input,
                                          int* topk_idx, T* topk_value) {
  const int t_idx = threadIdx.x;
  const int seq_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;

  __shared__ int s_last_valid_seq;
  __shared__ T s_data[TPB];
  __shared__ short s_max_idx[TPB];
  if (threadIdx.x == 0) s_last_valid_seq = mask[batch_idx];
  __syncthreads();

  // get sum, topk_idx and topk_value
  if (seq_idx < s_last_valid_seq) {
    int seq_len_offset = batch_idx * seq_len + seq_idx;
    const T* input_ptr = input + seq_len_offset * ld;

    // step1 assign input to s_data
    if (threadIdx.x < width) {
      s_data[t_idx] = input_ptr[t_idx];
      s_max_idx[t_idx] = t_idx;
    } else {
      s_data[t_idx] = 0;
      s_max_idx[t_idx] = -1;
    }
    __syncthreads();

    // step2 get max idx
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
      if (t_idx < stride) {
        int next_idx = t_idx + stride;
        if (s_data[t_idx] < s_data[next_idx]) {
          s_data[t_idx] = s_data[next_idx];
          s_max_idx[t_idx] = s_max_idx[next_idx];
        }
      }
      __syncthreads();
    }

    // step3 assign input to s_data to get sum
    if (threadIdx.x < width)
      s_data[t_idx] = exp(input_ptr[t_idx] - input_ptr[s_max_idx[0]]);
    else
      s_data[t_idx] = 0;
    __syncthreads();

    // step4 get sum
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
      if (t_idx < stride) {
        int next_idx = t_idx + stride;
        s_data[t_idx] += s_data[next_idx];
      }
      __syncthreads();
    }

    // step5 output
    if (t_idx == 0) {
      topk_idx[seq_len_offset] = s_max_idx[0];
      /*topk_value[seq_len_offset] = exp(input_ptr[s_max_idx[0]]) / s_data[0];*/
      topk_value[seq_len_offset] = T(1) / s_data[0];
    }
  }
}

template <typename T>
int ComputeSoftmaxAndTop1Tpl(const T* input, const int* mask, int B, int seq_len, int dim, int ld, int* topk_idx,
                             T* topk_value, cudaStream_t stream) {
  const dim3 grid(seq_len, B, 1);

  if (dim > 1024) return -1;

  if (ld <= 32) {
    const int block_size = 32;
    SoftmaxAndTop1KernelSmall<T, block_size>
        <<<grid, block_size, 0, stream>>>(seq_len, dim, ld, mask, input, topk_idx, topk_value);
  } else if (dim <= 64) {
    const int block_size = 64;
    SoftmaxAndTop1KernelSmall<T, block_size>
        <<<grid, block_size, 0, stream>>>(seq_len, dim, ld, mask, input, topk_idx, topk_value);
  } else if (dim <= 128) {
    const int block_size = 128;
    SoftmaxAndTop1KernelSmall<T, block_size>
        <<<grid, block_size, 0, stream>>>(seq_len, dim, ld, mask, input, topk_idx, topk_value);
  } else {
    return -1;
  }

  auto ret = cudaPeekAtLastError();
  if (ret != cudaSuccess) {
    LOG(ERROR) << "SoftmaxAndTop1KernelSmall failed! status = " << cudaGetErrorString(ret) << endl;
    return -1;
  }
  return 0;
}

int ComputeSoftmaxAndTop1(const float* input, const int* mask, int B, int seq_len, int dim, int ld, int* topk_idx,
                          float* topk_value, cudaStream_t stream) {
  return ComputeSoftmaxAndTop1Tpl(input, mask, B, seq_len, dim, ld, topk_idx, topk_value, stream);
}

int ComputeSoftmaxAndTop1(const half* input, const int* mask, int B, int seq_len, int dim, int ld, int* topk_idx,
                          half* topk_value, cudaStream_t stream) {
  return ComputeSoftmaxAndTop1Tpl(input, mask, B, seq_len, dim, ld, topk_idx, topk_value, stream);
}

}  // namespace plugin
}  // namespace nvinfer1
