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

#include "att_masked_softmax_plugin.h"

#include <cuda_fp16.h>
#include <algorithm>

#include "common.h"
#include "att_masked_softmax_kernel.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
// PluginFieldCollection AttMaskedSoftmaxPluginCreator::mFC{};
// std::vector<PluginField> AttMaskedSoftmaxPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AttMaskedSoftmaxPluginCreator);

AttMaskedSoftmaxPlugin::AttMaskedSoftmaxPlugin(const std::string name, const nvinfer1::DataType type, const float scale)
    : layer_name_(name), data_type_(type), scale_(scale) {}

AttMaskedSoftmaxPlugin::AttMaskedSoftmaxPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &scale_);

  // scale_fp16_ = __float2half(scale_);
}

size_t AttMaskedSoftmaxPlugin::getSerializationSize() const TRTNOEXCEPT { return sizeof(data_type_) + sizeof(scale_); }

void AttMaskedSoftmaxPlugin::serialize(void* buffer) const TRTNOEXCEPT {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, scale_);
}

bool AttMaskedSoftmaxPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                                       int nbOutputs) TRTNOEXCEPT {
  assert(nbInputs == 2);
  assert(nbOutputs == 1);

  const PluginTensorDesc& input = inOut[0];
  if (pos == 0) return (input.type == data_type_) && (input.format == TensorFormat::kLINEAR);

  if (pos == 1) {
    const PluginTensorDesc& idx = inOut[pos];
    return (idx.type == DataType::kINT32) && (idx.format == TensorFormat::kLINEAR);
  }

  if (pos == 2) {
    const PluginTensorDesc& output = inOut[pos];
    return (input.type == output.type) && (output.format == TensorFormat::kLINEAR);
  }
  return false;
}

nvinfer1::DataType AttMaskedSoftmaxPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                             int nbInputs) const TRTNOEXCEPT {
  ASSERT(inputTypes && nbInputs > 0);
  return inputTypes[0];
}

int AttMaskedSoftmaxPlugin::initialize() TRTNOEXCEPT { return 0; }

void AttMaskedSoftmaxPlugin::terminate() TRTNOEXCEPT {}

void AttMaskedSoftmaxPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                             const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) TRTNOEXCEPT {}

// x = x * self.xscale
// pos_emb = self.pe[:, offset:offset + seq_len]
int AttMaskedSoftmaxPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                                    void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT {
  auto batch = inputDesc[0].dims.d[0];
  auto seq_len = inputDesc[0].dims.d[1];
  auto dim = inputDesc[0].dims.d[2];

  if (data_type_ == DataType::kFLOAT) {
    const auto input = static_cast<const float*>(inputs[0]);
    const auto masked = static_cast<const int*>(inputs[1]);
    auto output = static_cast<float*>(outputs[0]);

    auto ret = compute_att_masked_softmax(stream, dim, batch, seq_len, scale_, masked, input, output);
    return ret;
  } else if (data_type_ == DataType::kHALF) {
    const auto input = static_cast<const half*>(inputs[0]);
    const auto masked = static_cast<const int*>(inputs[1]);
    auto output = static_cast<half*>(outputs[0]);

    auto ret = compute_att_masked_softmax(stream, dim, batch, seq_len, scale_, masked, input, output);
    return ret;
  }

  return 0;
}

nvinfer1::DimsExprs AttMaskedSoftmaxPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                                int nbInputs, nvinfer1::IExprBuilder& exprBuilder) TRTNOEXCEPT {
  return inputs[0];
}

nvinfer1::IPluginV2DynamicExt* AttMaskedSoftmaxPlugin::clone() const TRTNOEXCEPT {
  return new AttMaskedSoftmaxPlugin(layer_name_, data_type_, scale_);
}

void AttMaskedSoftmaxPlugin::destroy() TRTNOEXCEPT {
  delete this;
}

const char* AttMaskedSoftmaxPlugin::getPluginVersion() const TRTNOEXCEPT { return ATT_MASKED_SOFTMAX_PLUGIN_VERSION; }

const char* AttMaskedSoftmaxPlugin::getPluginType() const TRTNOEXCEPT { return ATT_MASKED_SOFTMAX_PLUGIN_NAME; }

size_t AttMaskedSoftmaxPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
                                                const nvinfer1::PluginTensorDesc* /*outputs*/,
                                                int /*nbOutputs*/) const TRTNOEXCEPT {
  return 0;
}

const char* AttMaskedSoftmaxPlugin::getPluginNamespace() const TRTNOEXCEPT { return ""; }

int AttMaskedSoftmaxPlugin::getNbOutputs() const TRTNOEXCEPT { return 1; }

void AttMaskedSoftmaxPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                                             nvinfer1::IGpuAllocator* allocator) TRTNOEXCEPT {}

const char* AttMaskedSoftmaxPluginCreator::getPluginName() const TRTNOEXCEPT { return ATT_MASKED_SOFTMAX_PLUGIN_NAME; }

const char* AttMaskedSoftmaxPluginCreator::getPluginVersion() const TRTNOEXCEPT { return ATT_MASKED_SOFTMAX_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* AttMaskedSoftmaxPluginCreator::getFieldNames() TRTNOEXCEPT {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

nvinfer1::IPluginV2DynamicExt* AttMaskedSoftmaxPluginCreator::createPlugin(const char* name,
                                                                           const nvinfer1::PluginFieldCollection* fc) TRTNOEXCEPT {
  assert(fc->nbFields == 2);

  gLogVerbose << "Creating AttMaskedSoftmaxPlugin...\n";

  int data_type_id;
  float scale;

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("data_type") == 0) {
      data_type_id = static_cast<const int*>(fc->fields[i].data)[0];
      gLogVerbose << "Building data_type_id : " << data_type_id << std::endl;
    }
    if (field_name.compare("scale") == 0) {
      scale = static_cast<const float*>(fc->fields[i].data)[0];
      gLogVerbose << "Building scale : " << scale << std::endl;
    }
  }

  if (data_type_id < 0 || data_type_id > 3) {
    gLogError << "Invalid type id" << data_type_id << std::endl;
    assert(0);
  }

  DataType type = static_cast<DataType>(data_type_id);
  return new AttMaskedSoftmaxPlugin(string(name), type, scale);
}

nvinfer1::IPluginV2DynamicExt* AttMaskedSoftmaxPluginCreator::deserializePlugin(const char* name,
                                                                                const void* serialData,
                                                                                size_t serialLength) TRTNOEXCEPT {
  return new AttMaskedSoftmaxPlugin(serialData, serialLength);
}

}  // namespace plugin
}  // namespace nvinfer1
