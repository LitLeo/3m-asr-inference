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

#ifndef PLUGIN_LAYER_NORM_PLUGIN_LAYER_NORM_KERNEL_H_
#define PLUGIN_LAYER_NORM_PLUGIN_LAYER_NORM_KERNEL_H_

#include "NvInfer.h"

// #include "plugin_math.h"
#include "common.h"

namespace nvinfer1 {
namespace plugin {

// int compute_layer_norm(cudaStream_t stream, int ld, int S, const float* input,
// const float* gamma, const float* beta, float* output, float eps);

int compute_layer_norm(cudaStream_t stream, const int ld, const int n, const float* input, const float* gamma,
                       const float* beta, float* output);

int compute_layer_norm(cudaStream_t stream, const int ld, const int n, const half* input, const half* gamma,
                       const half* beta, half* output);

int compute_layer_norm_torch(const float* X_data, const float* gamma, const float* beta, int64_t M, int64_t N,
                             float eps, float* Y_data, float* mean_data, float* rstd_data, cudaStream_t stream);

int compute_layer_norm_torch(const half* X_data, const half* gamma, const half* beta, int64_t M, int64_t N, half eps,
                             half* Y_data, half* mean_data, half* rstd_data, cudaStream_t stream);

}  // namespace plugin
}  // namespace nvinfer1

#endif  // PLUGIN_LAYER_NORM_PLUGIN_LAYER_NORM_KERNEL_H_
