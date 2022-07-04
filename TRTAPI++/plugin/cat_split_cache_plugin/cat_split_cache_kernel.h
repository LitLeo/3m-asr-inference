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

#ifndef PLUGIN_CAT_SPLIT_CACHE_KERNEL_H_
#define PLUGIN_CAT_SPLIT_CACHE_KERNEL_H_

#include "common.h"

namespace nvinfer1 {
namespace plugin {

int ComputeCatSplitCache(cudaStream_t stream, const int batch, const int cache_dim, const int input_dim,
                         const float* in_cache, const float* input, float* output, float* out_cache);

int ComputeCatSplitCache(cudaStream_t stream, const int batch, const int cache_dim, const int input_dim,
                         const half* in_cache, const half* input, half* output, half* out_cache);

}  // namespace plugin
}  // namespace nvinfer1
#endif  // PLUGIN_CAT_SPLIT_CACHE_KERNEL_H_
