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

#include "mask_conv2d_sample_kernel.h"

#include <algorithm>

#include "common.cuh"
#include "common.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

/*
  input dim[1, batch]
  output = ceil((input - left_padding) * 1.0f / stride)
 */
__global__ void mask_conv2d_sample_kernel(const int* input, const int batch, const int left_padding, const int stride,
                                          int* output) {
  int tidx = threadIdx.x;

  if (tidx < batch) {
    auto tmp = input[tidx] - left_padding - 1;
    output[tidx] = tmp / stride + 1;
  }
}

int compute_mask_conv2d_sample(const int* input, const int batch, const int left_padding, const int stride, int* output,
                               cudaStream_t stream) {
  int grid_size = 1;
  int block_size = (batch + 32) / 32 * 32;
  mask_conv2d_sample_kernel<<<grid_size, block_size, 0, stream>>>(input, batch, left_padding, stride, output);
  auto ret = cudaPeekAtLastError();
  if (ret != cudaSuccess) {
    gLogError << "mask_conv2d_sample_kernel failed! status = " << cudaGetErrorString(ret) << endl;
    return -1;
  }
  return 0;
}

}  // namespace plugin
}  // namespace nvinfer1
