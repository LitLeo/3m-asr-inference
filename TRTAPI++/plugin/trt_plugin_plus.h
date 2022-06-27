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

#ifndef PLUGIN_TRT_PLUGIN_CREATOR_H_
#define PLUGIN_TRT_PLUGIN_CREATOR_H_
#include <iostream>
#include <vector>

#include "NvInfer.h"

namespace nvinfer1 {
namespace plugin {

using ITensorVector = std::vector<nvinfer1::ITensor*>;

bool init_trt_plugin_plus(void* logger, const char* libNamespace);

ITensorVector add_celu_plugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                              const nvinfer1::DataType type, const float alpha);

ITensorVector add_batch_norm_plugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                    const nvinfer1::DataType type, const nvinfer1::Weights& weight,
                                    const nvinfer1::Weights& bias, const nvinfer1::Weights& mean,
                                    const nvinfer1::Weights& var, const float eps = 1e-5);

ITensorVector AddGluPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                           const nvinfer1::DataType type, const int axis_dim = -1);

ITensorVector add_group_norm_plugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                    const nvinfer1::DataType type, const int num_groups,
                                    const nvinfer1::Weights& weight, const nvinfer1::Weights& bias,
                                    const float eps = 1e-5);

ITensorVector AddLayerNormPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                 const nvinfer1::DataType type, const size_t dim, const float eps = 1e-5);

ITensorVector AddSiluPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                            const nvinfer1::DataType type);

/******************your self plugin***********************/
ITensorVector AddAttMaskedSoftmaxPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                        const nvinfer1::DataType type, const float scale);

ITensorVector AddAttStreamSoftmaxPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                        const nvinfer1::DataType type, const float scale, const int cache_len);

ITensorVector AddCatSplitCachePlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                     const nvinfer1::DataType type, const int axis_dim);

ITensorVector AddDumpTensorPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                  const nvinfer1::DataType type);

ITensorVector AddFMoEExpertPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                  const nvinfer1::DataType type, const int num_expert, const int idim,
                                  const int hidden_units, const int act_type);

// ITensorVector AddLeftPaddingCachePlugin(INetworkDefinition* network,
// const ITensorVector& input_tensors, const nvinfer1::DataType type, const int axis_dim);

ITensorVector AddMaskedFillPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                  const nvinfer1::DataType type, const float fill);

ITensorVector AddMaskConv2dSamplePlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                        const int left_padding, const int stride);

ITensorVector AddRelPositionalEncodingPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                             const nvinfer1::DataType type, const float scale, int max_len, int dim,
                                             const int streaming = 0);

}  // namespace plugin
}  // namespace nvinfer1

#endif  // PLUGIN_TRT_PLUGIN_CREATOR_H_
