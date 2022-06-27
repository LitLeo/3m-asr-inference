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

#ifndef PLUGIN_MASKED_FILL_PLUGIN_MASKED_FILL_KERNEL_H_
#define PLUGIN_MASKED_FILL_PLUGIN_MASKED_FILL_KERNEL_H_

#include "common.h"

namespace nvinfer1 {
namespace plugin {

int compute_masked_fill(const float* input, const int* masked, float fill, int batch, int dim, int seq_len,
                        float* output, cudaStream_t stream = 0);

int compute_masked_fill(const half* input, const int* masked, half fill, int batch, int dim, int seq_len, half* output,
                        cudaStream_t stream = 0);

}  // namespace plugin
}  // namespace nvinfer1
#endif  // PLUGIN_MASKED_FILL_PLUGIN_MASKED_FILL_KERNEL_H_
