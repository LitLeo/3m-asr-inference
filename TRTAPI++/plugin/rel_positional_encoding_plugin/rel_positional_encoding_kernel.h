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

#ifndef PLUGIN_REL_POSITIONAL_ENCODING_PLUGIN_REL_POSITIONAL_ENCODING_KERNEL_H_
#define PLUGIN_REL_POSITIONAL_ENCODING_PLUGIN_REL_POSITIONAL_ENCODING_KERNEL_H_

#include "common.h"

namespace nvinfer1 {
namespace plugin {

int compute_scale(const float* input, float scale, int n, float* output, cudaStream_t stream);
int compute_scale(const half* input, half scale, int n, half* output, cudaStream_t stream);

int compute_rel_positional_encoding(const float* input, const float* pe, const float scale, const int batch,
                                    const int seq_len, const int dim, float* output, float* pos_emb,
                                    cudaStream_t stream);

int compute_rel_positional_encoding(const half* input, const half* pe, const half scale, const int batch,
                                    const int seq_len, const int dim, half* output, half* pos_emb, cudaStream_t stream);

int compute_rel_positional_encoding_streaming(const float* input, const float* pe, const int* frame_num_input,
                                              const float scale, const int batch, const int seq_len, const int dim,
                                              float* output, float* pos_emb, cudaStream_t stream);

int compute_rel_positional_encoding_streaming(const half* input, const half* pe, const int* frame_num_input,
                                              const half scale, const int batch, const int seq_len, const int dim,
                                              half* output, half* pos_emb, cudaStream_t stream);

}  // namespace plugin
}  // namespace nvinfer1
#endif  // PLUGIN_REL_POSITIONAL_ENCODING_PLUGIN_REL_POSITIONAL_ENCODING_KERNEL_H_
