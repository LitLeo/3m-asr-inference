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

#include "masked_fill_kernel.h"

#include <cuda_fp16.h>
#include <algorithm>

#include "common.cuh"
#include "common.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

template <class T>
__global__ void masked_fill_kernel(const T* input, const int* masked, T fill, int dim, int seq_len, T* output) {
  int batch_id = blockIdx.y;
  int bidx = blockIdx.x;
  int tidx = threadIdx.x;

  int offset = (batch_id * dim + bidx) * seq_len;
  auto input_ptr = input + offset;
  auto output_ptr = output + offset;
  auto last_valid = masked[batch_id];

  for (int i = tidx; i < seq_len; i += blockDim.x) {
    if (i >= last_valid)
      output_ptr[i] = fill;
    else
      output_ptr[i] = input_ptr[i];
  }

  /*if (batch_id == 0 && bidx == 0 && tidx == 0) {*/
  /*printf("last_valid = %d\n", last_valid);*/
  /*printf("last_valid = %d\n", masked[1]);*/
  /*}*/
}

int compute_masked_fill(const float* input, const int* masked, float fill, int batch, int dim, int seq_len,
                        float* output, cudaStream_t stream) {
  dim3 grid_size(dim, batch, 1);
  int block_size = 64;
  masked_fill_kernel<float><<<grid_size, block_size, 0, stream>>>(input, masked, fill, dim, seq_len, output);
  auto ret = cudaPeekAtLastError();
  if (ret != cudaSuccess) {
    gLogError << "masked_fill_kernel ailed! status = " << cudaGetErrorString(ret) << endl;
    return -1;
  }
  return 0;
}

int compute_masked_fill(const half* input, const int* masked, half fill, int batch, int dim, int seq_len, half* output,
                        cudaStream_t stream) {
  dim3 grid_size(dim, batch, 1);
  int block_size = 64;
  masked_fill_kernel<half><<<grid_size, block_size, 0, stream>>>(input, masked, fill, dim, seq_len, output);
  auto ret = cudaPeekAtLastError();
  if (ret != cudaSuccess) {
    gLogError << "masked_fill_kernel ailed! status = " << cudaGetErrorString(ret) << endl;
    return -1;
  }
  return 0;
}

}  // namespace plugin
}  // namespace nvinfer1
