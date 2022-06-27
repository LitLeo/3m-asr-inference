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

#include "glu_kernel.h"

#include <cuda_fp16.h>
#include <algorithm>

#include "common.cuh"

using namespace std;

namespace nvinfer1 {
namespace plugin {

template <class T>
__global__ void glu_kernel(const int M, const int split_dim_size, const int N, const T* Xdata, T* Ydata) {
  const int xOffset = 2 * split_dim_size * N;
  const int yOffset = split_dim_size * N;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < M * split_dim_size * N) {
    const int i = index / split_dim_size / N;
    const int j = index / N % split_dim_size;
    const int k = index % N;
    const T x1 = Xdata[i * xOffset + j * N + k];
    const T x2 = Xdata[i * xOffset + (j + split_dim_size) * N + k];
    Ydata[i * yOffset + j * N + k] = x1 * sigmoid(x2);
  }
}

template <class T>
int ComputeGluTpl(const int M, const int split_dim_size, const int N, const T* x_data, T* y_data, cudaStream_t stream) {
  int block_size = 64;
  int grid_size = ((M * split_dim_size * N) + block_size - 1) / block_size;
  glu_kernel<<<grid_size, block_size, 0, stream>>>(M, split_dim_size, N, x_data, y_data);

  auto ret = cudaPeekAtLastError();
  if (ret != cudaSuccess) {
    gLogError << "gluDim0Kernel failed! status = " << cudaGetErrorString(ret) << endl;
    return -1;
  }
  return 0;
}

int ComputeGlu(const int M, const int split_dim_size, const int N, const float* x_data, float* y_data,
               cudaStream_t stream) {
  return ComputeGluTpl(M, split_dim_size, N, x_data, y_data, stream);
}

int ComputeGlu(const int M, const int split_dim_size, const int N, const half* x_data, half* y_data,
               cudaStream_t stream) {
  return ComputeGluTpl(M, split_dim_size, N, x_data, y_data, stream);
}

}  // namespace plugin
}  // namespace nvinfer1
