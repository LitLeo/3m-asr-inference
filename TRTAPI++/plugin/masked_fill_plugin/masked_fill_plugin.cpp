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

#include "masked_fill_plugin.h"

#include <cuda_fp16.h>
#include <algorithm>

#include "common.h"
#include "debug.h"
#include "masked_fill_kernel.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
// PluginFieldCollection MaskedFillPluginCreator::mFC{};
// std::vector<PluginField> MaskedFillPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(MaskedFillPluginCreator);

MaskedFillPlugin::MaskedFillPlugin(const std::string name, const nvinfer1::DataType type, const float fill)
    : layer_name_(name), data_type_(type), fill_(fill) {
  if (data_type_ == DataType::kHALF) half_fill_ = __float2half(fill_);
}

MaskedFillPlugin::MaskedFillPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &fill_);

  if (data_type_ == DataType::kHALF) half_fill_ = __float2half(fill_);
}

size_t MaskedFillPlugin::getSerializationSize() const TRTNOEXCEPT { return sizeof(data_type_) + sizeof(fill_); }

void MaskedFillPlugin::serialize(void* buffer) const TRTNOEXCEPT {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, fill_);
}

bool MaskedFillPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
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

nvinfer1::DataType MaskedFillPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                       int nbInputs) const TRTNOEXCEPT {
  ASSERT(inputTypes && nbInputs > 0);
  return inputTypes[0];
}

int MaskedFillPlugin::initialize() TRTNOEXCEPT { return 0; }

void MaskedFillPlugin::terminate() TRTNOEXCEPT {}

void MaskedFillPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) TRTNOEXCEPT {}

// x = x * self.xscale
// pos_emb = self.pe[:, offset:offset + seq_len]
int MaskedFillPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                              const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT {
  auto batch = inputDesc[0].dims.d[0];
  auto dim = inputDesc[0].dims.d[1];
  auto seq_len = inputDesc[0].dims.d[2];

  if (data_type_ == DataType::kFLOAT) {
    const auto input = static_cast<const float*>(inputs[0]);
    const auto masked = static_cast<const int*>(inputs[1]);
    auto output = static_cast<float*>(outputs[0]);

    auto ret = compute_masked_fill(input, masked, fill_, batch, dim, seq_len, output, stream);
    return ret;
  } else if (data_type_ == DataType::kHALF) {
    const auto input = static_cast<const half*>(inputs[0]);
    const auto masked = static_cast<const int*>(inputs[1]);
    auto output = static_cast<half*>(outputs[0]);

    auto ret = compute_masked_fill(input, masked, half_fill_, batch, dim, seq_len, output, stream);
    return ret;
  }

  // std::vector<int64_t> input_dims(inputDesc->dims.nbDims);
  // for (int i = 0; i < inputDesc->dims.nbDims; i++)
  // input_dims[i] = inputDesc->dims.d[i];

  // at::Tensor input, output;
  // if (inputDesc->type == DataType::kHALF) {
  // auto type = at::device(at::kCUDA).dtype(torch::kFloat16);
  // input = at::from_blob((void*)inputs[0], input_dims, type);
  // output = at::from_blob((void*)outputs[0], input_dims, type);
  //} else {
  // auto type = at::device(at::kCUDA).dtype(torch::kFloat);
  // input = at::from_blob((void*)inputs[0], input_dims, type);
  // output = at::from_blob((void*)outputs[0], input_dims, type);
  //}
  // print_tensor(input, "input", true);
  // print_tensor(output, "output", true);

  return -1;
}

nvinfer1::DimsExprs MaskedFillPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                          int nbInputs, nvinfer1::IExprBuilder& exprBuilder) TRTNOEXCEPT {
  return inputs[0];
}

nvinfer1::IPluginV2DynamicExt* MaskedFillPlugin::clone() const TRTNOEXCEPT {
  return new MaskedFillPlugin(layer_name_, data_type_, fill_);
}

void MaskedFillPlugin::destroy() TRTNOEXCEPT {
  delete this;
}

const char* MaskedFillPlugin::getPluginVersion() const TRTNOEXCEPT { return MASKED_FILL_PLUGIN_VERSION; }

const char* MaskedFillPlugin::getPluginType() const TRTNOEXCEPT { return MASKED_FILL_PLUGIN_NAME; }

size_t MaskedFillPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
                                          const nvinfer1::PluginTensorDesc* /*outputs*/,
                                          int /*nbOutputs*/) const TRTNOEXCEPT {
  return 0;
}

const char* MaskedFillPlugin::getPluginNamespace() const TRTNOEXCEPT { return ""; }

int MaskedFillPlugin::getNbOutputs() const TRTNOEXCEPT { return 1; }

void MaskedFillPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) TRTNOEXCEPT {}

const char* MaskedFillPluginCreator::getPluginName() const TRTNOEXCEPT { return MASKED_FILL_PLUGIN_NAME; }

const char* MaskedFillPluginCreator::getPluginVersion() const TRTNOEXCEPT { return MASKED_FILL_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* MaskedFillPluginCreator::getFieldNames() TRTNOEXCEPT {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

nvinfer1::IPluginV2DynamicExt* MaskedFillPluginCreator::createPlugin(const char* name,
                                                                     const nvinfer1::PluginFieldCollection* fc) TRTNOEXCEPT {
  assert(fc->nbFields == 2);

  gLogVerbose << "Creating MaskedFillPlugin...\n";

  int data_type_id;
  float fill;

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("data_type") == 0) {
      data_type_id = static_cast<const int*>(fc->fields[i].data)[0];
      gLogVerbose << "Building data_type_id : " << data_type_id << std::endl;
    }

    if (field_name.compare("fill") == 0) {
      fill = static_cast<const float*>(fc->fields[i].data)[0];
      gLogVerbose << "Building fill: " << fill << std::endl;
    }
  }

  if (data_type_id < 0 || data_type_id > 3) {
    gLogError << "Invalid type id" << data_type_id << std::endl;
    assert(0);
  }

  DataType type = static_cast<DataType>(data_type_id);
  return new MaskedFillPlugin(string(name), type, fill);
}

nvinfer1::IPluginV2DynamicExt* MaskedFillPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                                          size_t serialLength) TRTNOEXCEPT {
  return new MaskedFillPlugin(serialData, serialLength);
}

}  // namespace plugin
}  // namespace nvinfer1
