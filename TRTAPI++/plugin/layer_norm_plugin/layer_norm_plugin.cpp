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

#include "layer_norm_plugin.h"

#include <cassert>
#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "common.h"
#include "layer_norm_kernel.h"
#include "serialize.hpp"

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
PluginFieldCollection LayerNormPluginCreator::FC_{};
std::vector<PluginField> LayerNormPluginCreator::plugin_attributes_;

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

LayerNormPlugin::LayerNormPlugin(const std::string& name, const nvinfer1::DataType type, const size_t dim,
                                 const float eps)
    : layer_name_(name), dim_(dim), data_type_(type), eps_(eps) {}

LayerNormPlugin::LayerNormPlugin(const std::string& name, const void* data, size_t length) : layer_name_(name) {
  // Deserialize in the same order as serialization
  deserialize_value(&data, &length, &data_type_);
  deserialize_value(&data, &length, &dim_);
  deserialize_value(&data, &length, &eps_);
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* LayerNormPlugin::clone() const TRTNOEXCEPT {
  auto ret = new LayerNormPlugin(layer_name_, data_type_, dim_, eps_);
  return ret;
}

DimsExprs LayerNormPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs,
                                               IExprBuilder& exprBuilder) TRTNOEXCEPT {
  assert(nbInputs == 3);
  return inputs[0];
}

bool LayerNormPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut,
                                                int nbInputs, int nbOutputs) TRTNOEXCEPT {
  assert(nbInputs == 3);
  assert(nbOutputs == 1);

  const PluginTensorDesc& in_out = inOut[pos];
  return (in_out.type == data_type_) && (in_out.format == TensorFormat::kLINEAR);
}

void LayerNormPlugin::configurePlugin(const DynamicPluginTensorDesc* inputs, int nbInputs,
                                      const DynamicPluginTensorDesc* outputs, int nbOutputs) TRTNOEXCEPT {
  // Validate input arguments
  assert(nbInputs == 3);
  assert(nbOutputs == 1);
  assert(data_type_ == inputs[0].desc.type);
}

size_t LayerNormPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
                                         int nbOutputs) const TRTNOEXCEPT {
  return 0;
}

int LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                             const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT {
  const int input_volume = volume(inputDesc[0].dims);
  const int S = input_volume / dim_;

  int status = -1;

  const size_t word_size = getElementSize(data_type_);

  if (data_type_ == DataType::kFLOAT) {
    // Our plugin outputs only one tensor
    const float* input = static_cast<const float*>(inputs[0]);
    const float* gamma_ptr = static_cast<const float*>(inputs[1]);
    const float* beta_ptr = static_cast<const float*>(inputs[2]);
    float* output = static_cast<float*>(outputs[0]);

    // status = compute_layer_norm(stream, dim_, input_volume, input, gamma_ptr, beta_ptr, output);
    status = compute_layer_norm(stream, S, dim_, input, gamma_ptr, beta_ptr, output);

  } else if (data_type_ == DataType::kHALF) {
    // Our plugin outputs only one tensor
    const half* input = static_cast<const half*>(inputs[0]);
    const half* gamma_ptr = static_cast<const half*>(inputs[1]);
    const half* beta_ptr = static_cast<const half*>(inputs[2]);
    half* output = static_cast<half*>(outputs[0]);

    // status = compute_layer_norm(stream, dim_, input_volume, input, gamma_ptr, beta_ptr, output);
    status = compute_layer_norm(stream, S, dim_, input, gamma_ptr, beta_ptr, output);

  } else {
    assert(false);
  }

  return status;
}

// IPluginV2Ext Methods
DataType LayerNormPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const TRTNOEXCEPT {
  assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
  return inputTypes[0];
}

// IPluginV2 Methods
const char* LayerNormPlugin::getPluginType() const TRTNOEXCEPT { return LAYER_NORM_NAME; }

const char* LayerNormPlugin::getPluginVersion() const TRTNOEXCEPT { return LAYER_NORM_VERSION; }

int LayerNormPlugin::getNbOutputs() const TRTNOEXCEPT { return 1; }

int LayerNormPlugin::initialize() TRTNOEXCEPT { return 0; }

void LayerNormPlugin::terminate() TRTNOEXCEPT {}

size_t LayerNormPlugin::getSerializationSize() const TRTNOEXCEPT {
  const size_t word_size = getElementSize(data_type_);
  return sizeof(data_type_) + sizeof(dim_) + sizeof(eps_);
}

void LayerNormPlugin::serialize(void* buffer) const TRTNOEXCEPT {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, dim_);
  serialize_value(&buffer, eps_);
}

void LayerNormPlugin::destroy() TRTNOEXCEPT {
  delete this;
}

void LayerNormPlugin::setPluginNamespace(const char* libNamespace) TRTNOEXCEPT { namespace_ = libNamespace; }

const char* LayerNormPlugin::getPluginNamespace() const TRTNOEXCEPT { return namespace_.c_str(); }

///////////////////////

LayerNormPluginCreator::LayerNormPluginCreator() {
  FC_.nbFields = plugin_attributes_.size();
  FC_.fields = plugin_attributes_.data();
}

const char* LayerNormPluginCreator::getPluginName() const TRTNOEXCEPT { return LAYER_NORM_NAME; }

const char* LayerNormPluginCreator::getPluginVersion() const TRTNOEXCEPT { return LAYER_NORM_VERSION; }

const PluginFieldCollection* LayerNormPluginCreator::getFieldNames() TRTNOEXCEPT { return &FC_; }

IPluginV2* LayerNormPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRTNOEXCEPT {
  gLogVerbose << "Creating LayerNormPlugin...\n";

  int typeId = -1;
  int dim = 0;
  float eps = 1e-05;
  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("data_type") == 0) {
      typeId = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building typeId: " << typeId << std::endl;
    }
    if (field_name.compare("eps") == 0) {
      eps = *static_cast<const float*>(fc->fields[i].data);
      gLogVerbose << "Building eps: " << eps << std::endl;
    }
    if (field_name.compare("dim") == 0) {
      dim = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building dim " << dim << std::endl;
    }
  }

  if (typeId < 0 || typeId > 1) {
    gLogError << "LayerNorm: invalid typeId " << typeId << std::endl;
    return nullptr;
  }

  DataType type = static_cast<DataType>(typeId);

  gLogVerbose << "Building the Plugin...\n";
  LayerNormPlugin* p = new LayerNormPlugin(name, type, dim, eps);
  return p;
}

IPluginV2* LayerNormPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRTNOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call LayerNormPlugin::destroy()
  return new LayerNormPlugin(name, serialData, serialLength);
}

void LayerNormPluginCreator::setPluginNamespace(const char* libNamespace) TRTNOEXCEPT { namespace_ = libNamespace; }

const char* LayerNormPluginCreator::getPluginNamespace() const TRTNOEXCEPT { return namespace_.c_str(); }

}  // namespace plugin
}  // namespace nvinfer1
