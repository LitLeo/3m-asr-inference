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

#ifndef PLUGIN_DUMP_TENSOR_PLUGIN_H_
#define PLUGIN_DUMP_TENSOR_PLUGIN_H_

#include <string>
#include <vector>

#include "NvInfer.h"

#include "common.h"

namespace nvinfer1 {
namespace plugin {

constexpr const char* DUMP_TENSOR_PLUGIN_VERSION{"1"};
constexpr const char* DUMP_TENSOR_PLUGIN_NAME{"DumpTensorPluginDynamic"};

void p(const float* data, std::vector<int>& dims);
void p(const int* data, std::vector<int>& dims);

void p_sum(const float* data, std::vector<int>& dims, std::string message);
void p_sum(const int* data, std::vector<int>& dims, std::string message);

class DumpTensorPlugin : public nvinfer1::IPluginV2DynamicExt {
 public:
  explicit DumpTensorPlugin(const std::string name);

  DumpTensorPlugin(const std::string name, const void* data, size_t length);

  // It doesn't make sense to make DumpTensorPlugin without arguments, so we delete
  // default constructor.
  DumpTensorPlugin() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt* clone() const TRTNOEXCEPT override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) TRTNOEXCEPT override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                 int nbOutputs) TRTNOEXCEPT override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) TRTNOEXCEPT override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const TRTNOEXCEPT override;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRTNOEXCEPT override;

  // IPluginV2 Methods
  const char* getPluginType() const TRTNOEXCEPT override;
  const char* getPluginVersion() const TRTNOEXCEPT override;
  int getNbOutputs() const TRTNOEXCEPT override;
  int initialize() TRTNOEXCEPT override;
  void terminate() TRTNOEXCEPT override;
  size_t getSerializationSize() const TRTNOEXCEPT override;
  void serialize(void* buffer) const TRTNOEXCEPT override;
  void destroy() TRTNOEXCEPT override;
  void setPluginNamespace(const char* pluginNamespace) TRTNOEXCEPT override;
  const char* getPluginNamespace() const TRTNOEXCEPT override;

 private:
  std::string layer_name_;
  std::string namespace_;

  nvinfer1::DataType data_type_;
  // int dump_tensor_num_;

 //protected:
  //// To prevent compiler warnings.
  //using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
  //using nvinfer1::IPluginV2DynamicExt::configurePlugin;
  //using nvinfer1::IPluginV2DynamicExt::enqueue;
  //using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
  //using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
  //using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
  //using nvinfer1::IPluginV2DynamicExt::supportsFormat;
};

class DumpTensorPluginCreator : public nvinfer1::IPluginCreator {
 public:
  DumpTensorPluginCreator();
  const char* getPluginName() const TRTNOEXCEPT override;
  const char* getPluginVersion() const TRTNOEXCEPT override;
  const nvinfer1::PluginFieldCollection* getFieldNames() TRTNOEXCEPT override;
  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) TRTNOEXCEPT override;
  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRTNOEXCEPT override;
  void setPluginNamespace(const char* pluginNamespace) TRTNOEXCEPT override;
  const char* getPluginNamespace() const TRTNOEXCEPT override;

 private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string namespace_;
};

}  // namespace plugin
}  // namespace nvinfer1

#endif  // PLUGIN_DUMP_TENSOR_PLUGIN_H_
