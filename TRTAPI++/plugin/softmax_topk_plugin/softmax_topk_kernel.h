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

#ifndef PLUGIN_SOFTMAX_TOPK_PLUGIN_SOFTMAX_TOPK_KERNEL_H_
#define PLUGIN_SOFTMAX_TOPK_PLUGIN_SOFTMAX_TOPK_KERNEL_H_

#include "common.h"

namespace nvinfer1 {
namespace plugin {

int ComputeSoftmaxAndTop1(const float* input, const int* mask, int B, int seq_len, int dim, int ld, int* topk_idx,
                          float* topk_value, cudaStream_t stream);

int ComputeSoftmaxAndTop1(const half* input, const int* mask, int B, int seq_len, int dim, int ld, int* topk_idx,
                          half* topk_value, cudaStream_t stream);

}  // namespace plugin
}  // namespace nvinfer1
#endif  // PLUGIN_SOFTMAX_TOPK_PLUGIN_SOFTMAX_TOPK_KERNEL_H_
