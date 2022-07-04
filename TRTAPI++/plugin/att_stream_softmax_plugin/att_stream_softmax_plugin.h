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

#ifndef PLUGIN_ATT_STREAM_SOFTMAX_PLUGIN_H_
#define PLUGIN_ATT_STREAM_SOFTMAX_PLUGIN_H_
#include <thrust/device_vector.h>
#include <iostream>
#include <string>

#include <NvInfer.h>

#include "common.h"

namespace nvinfer1 {
namespace plugin {

constexpr const char* ATT_STREAM_SOFTMAX_PLUGIN_VERSION{"1"};
constexpr const char* ATT_STREAM_SOFTMAX_PLUGIN_NAME{"AttStreamSoftmaxPluginDynamic"};

class AttStreamSoftmaxPlugin final : public nvinfer1::IPluginV2DynamicExt {
  const std::string layer_name_;
  nvinfer1::DataType data_type_;
  float scale_;
  int cache_len_;

  // protected:
  //// Supress warnings about hiding function names due to overloads and overrides of virtuals.
  // using IPluginV2DynamicExt::configurePlugin;
  // using IPluginV2DynamicExt::enqueue;
  // using IPluginV2DynamicExt::getOutputDimensions;
  // using IPluginV2DynamicExt::getWorkspaceSize;

 public:
  AttStreamSoftmaxPlugin(const std::string name, const nvinfer1::DataType type, const float scale, const int cache_len);
  AttStreamSoftmaxPlugin(void const* serialData, size_t serialLength);

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                 int nbOutputs) TRTNOEXCEPT override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const TRTNOEXCEPT override;
  int initialize() TRTNOEXCEPT override;
  void terminate() TRTNOEXCEPT override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) TRTNOEXCEPT override;
  int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
              void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) TRTNOEXCEPT override;
  size_t getSerializationSize() const TRTNOEXCEPT override;
  void serialize(void* buffer) const TRTNOEXCEPT override;
  nvinfer1::IPluginV2DynamicExt* clone() const TRTNOEXCEPT override;
  void destroy() TRTNOEXCEPT override;
  const char* getPluginVersion() const TRTNOEXCEPT override;
  const char* getPluginType() const TRTNOEXCEPT override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
                          const nvinfer1::PluginTensorDesc* /*outputs*/, int /*nbOutputs*/) const TRTNOEXCEPT override;
  void setPluginNamespace(const char* /*pluginNamespace*/) TRTNOEXCEPT override {}
  const char* getPluginNamespace() const TRTNOEXCEPT override;
  int getNbOutputs() const TRTNOEXCEPT override;

  void attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                       nvinfer1::IGpuAllocator* allocator) TRTNOEXCEPT override;
  void detachFromContext() TRTNOEXCEPT override {}
};

class AttStreamSoftmaxPluginCreator : public nvinfer1::IPluginCreator {
 public:
  AttStreamSoftmaxPluginCreator() {}
  ~AttStreamSoftmaxPluginCreator() {}

  const char* getPluginName() const TRTNOEXCEPT;
  const char* getPluginVersion() const TRTNOEXCEPT;
  const nvinfer1::PluginFieldCollection* getFieldNames() TRTNOEXCEPT;
  nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) TRTNOEXCEPT;
  nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData,
                                                   size_t serialLength) TRTNOEXCEPT;

  void setPluginNamespace(const char* libNamespace) TRTNOEXCEPT { m_namespace_ = libNamespace; }

  const char* getPluginNamespace() const TRTNOEXCEPT { return m_namespace_.c_str(); }

 private:
  // static nvinfer1::PluginFieldCollection mFC;
  // static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string m_namespace_;
};

}  // namespace plugin
}  // namespace nvinfer1
#endif  // PLUGIN_ATT_STREAM_SOFTMAX_PLUGIN_H_
