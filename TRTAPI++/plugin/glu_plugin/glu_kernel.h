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

#ifndef PLUGIN_GLU_PLUGIN_GLU_KERNEL_H_
#define PLUGIN_GLU_PLUGIN_GLU_KERNEL_H_

#include "common.h"

namespace nvinfer1 {
namespace plugin {

int ComputeGlu(const int M, const int split_dim_size, const int N, const float* x_data, float* y_data,
               cudaStream_t stream);
int ComputeGlu(const int M, const int split_dim_size, const int N, const half* x_data, half* y_data,
               cudaStream_t stream);

}  // namespace plugin
}  // namespace nvinfer1
#endif  // PLUGIN_GLU_PLUGIN_GLU_KERNEL_H_
