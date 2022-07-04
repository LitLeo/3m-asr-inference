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

#ifndef PLUGIN_SOFTMAX_TOPK_PLUGIN_SOFTMAX_TOPK_PLUGIN_H_
#define PLUGIN_SOFTMAX_TOPK_PLUGIN_SOFTMAX_TOPK_PLUGIN_H_

#include <thrust/device_vector.h>
#include <iostream>
#include <string>

#include <NvInfer.h>

#include "common.h"

namespace nvinfer1 {
namespace plugin {

constexpr const char* SOFTMAX_TOPK_PLUGIN_VERSION{"1"};
constexpr const char* SOFTMAX_TOPK_PLUGIN_NAME{"SoftmaxTopKPluginDynamic"};

class SoftmaxTopKPlugin final : public nvinfer1::IPluginV2DynamicExt {
  const std::string layer_name_;
  nvinfer1::DataType data_type_;
  int axis_dim_;
  int k_;

 public:
  SoftmaxTopKPlugin(const std::string name, const nvinfer1::DataType type, const int axis_dim, const int k);
  SoftmaxTopKPlugin(void const* serial_data, size_t serial_length);

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
                                 int nb_outputs) TRTNOEXCEPT override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* input_types,
                                       int nb_inputs) const TRTNOEXCEPT override;
  int initialize() TRTNOEXCEPT override;
  void terminate() TRTNOEXCEPT override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nb_inputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nb_outputs) TRTNOEXCEPT override;
  int enqueue(const PluginTensorDesc* input_desc, const PluginTensorDesc* output_desc, const void* const* inputs,
              void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT override;
  nvinfer1::DimsExprs getOutputDimensions(int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
                                          nvinfer1::IExprBuilder& expr_builder) TRTNOEXCEPT override;
  size_t getSerializationSize() const TRTNOEXCEPT override;
  void serialize(void* buffer) const TRTNOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt* clone() const TRTNOEXCEPT override;
  void destroy() TRTNOEXCEPT override;
  const char* getPluginVersion() const TRTNOEXCEPT override;
  const char* getPluginType() const TRTNOEXCEPT override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nb_inputs*/,
                          const nvinfer1::PluginTensorDesc* /*outputs*/, int /*nb_outputs*/) const TRTNOEXCEPT override;
  void setPluginNamespace(const char* /*plugin_namespace*/) TRTNOEXCEPT override {}
  const char* getPluginNamespace() const TRTNOEXCEPT override;
  int getNbOutputs() const TRTNOEXCEPT override;

  void attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                       nvinfer1::IGpuAllocator* allocator) TRTNOEXCEPT override;
  void detachFromContext() TRTNOEXCEPT override {}
};

class SoftmaxTopKCreator : public nvinfer1::IPluginCreator {
 public:
  SoftmaxTopKCreator() {}
  ~SoftmaxTopKCreator() {}

  const char* getPluginName() const TRTNOEXCEPT;
  const char* getPluginVersion() const TRTNOEXCEPT;
  const nvinfer1::PluginFieldCollection* getFieldNames() TRTNOEXCEPT;
  nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name,
                                              const nvinfer1::PluginFieldCollection* field_collection) TRTNOEXCEPT;
  nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serial_data,
                                                   size_t serial_length) TRTNOEXCEPT;

  void setPluginNamespace(const char* lib_namespace) TRTNOEXCEPT { m_namespace_ = lib_namespace; }

  const char* getPluginNamespace() const TRTNOEXCEPT { return m_namespace_.c_str(); }

 private:
  // static nvinfer1::PluginFieldCollection mFC_;
  // static std::vector<nvinfer1::PluginField> mPluginAttributes_;
  std::string m_namespace_;
};

}  // namespace plugin
}  // namespace nvinfer1
#endif  // PLUGIN_SOFTMAX_TOPK_PLUGIN_SOFTMAX_TOPK_PLUGIN_H_
