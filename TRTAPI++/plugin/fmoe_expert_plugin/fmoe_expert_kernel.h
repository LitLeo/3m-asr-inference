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

#ifndef PLUGIN_FMOE_KERNEL_H_
#define PLUGIN_FMOE_KERNEL_H_

#include <string>
#include <vector>

#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include "cuda_fp16.h"

namespace nvinfer1 {
namespace plugin {

int ComputeScatterMapping(const int* gate_idx, const int num_expert,
    const int idx_num, int* mapping, int* acc_histogram, cudaStream_t stream);

int ComputeScatterMappingCopy(const float* input, const int* mapping,
    const int S, const int dim, float* output, cudaStream_t stream);

int ComputeScatterMappingCopy(const half* input, const int* mapping,
    const int S, const int dim, half* output, cudaStream_t stream);

int ComputeBiasSilu(const float* input, const float* bias, const int N, const int dim,
                 float* output, cudaStream_t stream);

int ComputeBiasSilu(const half* input, const half* bias, const int N, const int dim,
                 half* output, cudaStream_t stream);

int ComputeBias(const float* input, const float* bias, const int N, const int dim,
                 float* output, cudaStream_t stream);

int ComputeBias(const half* input, const half* bias, const int N, const int dim,
                 half* output, cudaStream_t stream);

int ComputeGatherrMappingCopy(const float* input, const int* mapping,
    const int S, const int dim, float* output, cudaStream_t stream);

int ComputeGatherrMappingCopy(const half* input, const int* mapping,
    const int S, const int dim, half* output, cudaStream_t stream);

}  // namespace plugin
}  // namespace nvinfer1

#endif  // PLUGIN_FMOE_KERNEL_H_
