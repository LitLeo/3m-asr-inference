/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "silu_plugin.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <vector>

#include "silu_kernel.h"
#include "debug.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
// PluginFieldCollection SiluPluginCreator::mFC{};
// std::vector<PluginField> SiluPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(SiluPluginCreator);

SiluPlugin::SiluPlugin(const std::string name, const nvinfer1::DataType type) : layer_name_(name), data_type_(type) {}

SiluPlugin::SiluPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &data_type_);
}

size_t SiluPlugin::getSerializationSize() const { return sizeof(data_type_); }

void SiluPlugin::serialize(void* buffer) const { serialize_value(&buffer, data_type_); }

bool SiluPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                           int nbOutputs) {
  assert(nbInputs == 1);
  assert(nbOutputs == 1);

  const PluginTensorDesc& input = inOut[0];
  if (pos == 0) return (input.type == data_type_) && (input.format == TensorFormat::kLINEAR);

  if (pos == 1) {
    const PluginTensorDesc& output = inOut[1];
    return (input.type == output.type) && (output.format == TensorFormat::kLINEAR);
  }
  return false;
}

nvinfer1::DataType SiluPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  ASSERT(inputTypes && nbInputs > 0);
  return inputTypes[0];
}

int SiluPlugin::initialize() { return 0; }

void SiluPlugin::terminate() {}

void SiluPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                 const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {}

// silu(x) = x * sigmoid(x)
int SiluPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) {

  auto N = volume(inputDesc->dims);

  int status = -1;

  if (data_type_ == DataType::kFLOAT) {
    auto input_ptr = static_cast<const float*>(inputs[0]);
    auto output_ptr = static_cast<float*>(outputs[0]);

    status = compute_silu(input_ptr, N, output_ptr, stream);

  } else {
    auto input_ptr = static_cast<const half*>(inputs[0]);
    auto output_ptr = static_cast<half*>(outputs[0]);

    status = compute_silu(input_ptr, N, output_ptr, stream);
  }

  return status;
}

nvinfer1::DimsExprs SiluPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                                    nvinfer1::IExprBuilder& exprBuilder) {
  return inputs[0];
}

nvinfer1::IPluginV2DynamicExt* SiluPlugin::clone() const { return new SiluPlugin(layer_name_, data_type_); }

void SiluPlugin::destroy() {
  // gLogVerbose << "SiluPlugin destroy\n";
  // This gets called when the network containing plugin is destroyed
}

const char* SiluPlugin::getPluginVersion() const { return SILU_PLUGIN_VERSION; }

const char* SiluPlugin::getPluginType() const { return SILU_PLUGIN_NAME; }

size_t SiluPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
                                    const nvinfer1::PluginTensorDesc* /*outputs*/,
                                    int /*nbOutputs*/) const TRTNOEXCEPT {
  return 0;
}

const char* SiluPlugin::getPluginNamespace() const { return ""; }

int SiluPlugin::getNbOutputs() const { return 1; }

void SiluPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) {}

const char* SiluPluginCreator::getPluginName() const { return SILU_PLUGIN_NAME; }

const char* SiluPluginCreator::getPluginVersion() const { return SILU_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* SiluPluginCreator::getFieldNames() {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

nvinfer1::IPluginV2DynamicExt* SiluPluginCreator::createPlugin(const char* name,
                                                               const nvinfer1::PluginFieldCollection* fc) {
  assert(fc->nbFields == 1);

  gLogVerbose << "Creating SiluPlugin...\n";

  int data_type_id = 0;

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("data_type") == 0) {
      data_type_id = static_cast<const int*>(fc->fields[i].data)[0];
      gLogVerbose << "Building data_type_id : " << data_type_id << std::endl;

      if (data_type_id < 0 || data_type_id > 1) {
        gLogError << "Invalid type id" << data_type_id << std::endl;
        assert(0);
      }
    }
  }

  DataType type = static_cast<DataType>(data_type_id);
  return new SiluPlugin(string(name), type);
}

nvinfer1::IPluginV2DynamicExt* SiluPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                                    size_t serialLength) {
  return new SiluPlugin(serialData, serialLength);
}

}  // namespace plugin
}  // namespace nvinfer1
