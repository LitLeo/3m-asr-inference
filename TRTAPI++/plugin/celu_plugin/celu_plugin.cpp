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
#include "celu_plugin.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "torch/script.h"
#include "torch/torch.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
// PluginFieldCollection CeluPluginCreator::mFC{};
// std::vector<PluginField> CeluPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(CeluPluginCreator);

CeluPlugin::CeluPlugin(const std::string name, const nvinfer1::DataType type, float alpha)
    : layer_name_(name), data_type_(type), alpha_(alpha) {}

CeluPlugin::CeluPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &alpha_);
}

size_t CeluPlugin::getSerializationSize() const { return sizeof(data_type_) + sizeof(alpha_); }

void CeluPlugin::serialize(void* buffer) const {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, alpha_);
}

bool CeluPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
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

nvinfer1::DataType CeluPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  ASSERT(inputTypes && nbInputs > 0);
  return inputTypes[0];
}

int CeluPlugin::initialize() { return 0; }

void CeluPlugin::terminate() {}

void CeluPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                 const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {}

int CeluPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) {
  std::vector<int64_t> tensor_dims;
  for (int i = 0; i < inputDesc->dims.nbDims; i++) {
    tensor_dims.push_back(inputDesc->dims.d[i]);
  }

  at::Tensor input, output;
  if (inputDesc->type == DataType::kHALF) {
    input = at::from_blob((void*)inputs[0], tensor_dims, at::device(at::kCUDA).dtype(torch::kFloat16));
    output = at::from_blob(outputs[0], tensor_dims, at::device(at::kCUDA).dtype(torch::kFloat16));
  } else {
    input = at::from_blob((void*)inputs[0], tensor_dims, at::device(at::kCUDA).dtype(torch::kFloat));
    output = at::from_blob(outputs[0], tensor_dims, at::device(at::kCUDA).dtype(torch::kFloat));
  }

  c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard torch_guard(torch_stream);

  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, stream);

  cudaStreamWaitEvent(torch_stream.stream(), event, 0);

  at::Tensor result = at::celu(input, alpha_);
  output.copy_(result);

  cudaEvent_t torch_event;
  cudaEventCreate(&torch_event);
  cudaEventRecord(torch_event, torch_stream.stream());

  cudaStreamWaitEvent(stream, torch_event, 0);

  cudaEventDestroy(event);
  cudaEventDestroy(torch_event);
  return 0;
}

nvinfer1::DimsExprs CeluPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                                    nvinfer1::IExprBuilder& exprBuilder) {
  return inputs[0];
}

nvinfer1::IPluginV2DynamicExt* CeluPlugin::clone() const { return new CeluPlugin(layer_name_, data_type_, alpha_); }

void CeluPlugin::destroy() {
  // gLogVerbose << "CeluPlugin destroy\n";
  // This gets called when the network containing plugin is destroyed
}

const char* CeluPlugin::getPluginVersion() const { return CELU_PLUGIN_VERSION; }

const char* CeluPlugin::getPluginType() const { return CELU_PLUGIN_NAME; }

size_t CeluPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
                                    const nvinfer1::PluginTensorDesc* /*outputs*/,
                                    int /*nbOutputs*/) const TRTNOEXCEPT {
  return 0;
}

const char* CeluPlugin::getPluginNamespace() const { return ""; }

int CeluPlugin::getNbOutputs() const { return 1; }

void CeluPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) {}

const char* CeluPluginCreator::getPluginName() const { return CELU_PLUGIN_NAME; }

const char* CeluPluginCreator::getPluginVersion() const { return CELU_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* CeluPluginCreator::getFieldNames() {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

nvinfer1::IPluginV2DynamicExt* CeluPluginCreator::createPlugin(const char* name,
                                                               const nvinfer1::PluginFieldCollection* fc) {
  assert(fc->nbFields == 2);

  gLogVerbose << "Creating CeluPlugin...\n";

  int data_type_id = 0;
  float alpha = 1.0;

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

    if (field_name.compare("alpha") == 0) {
      alpha = static_cast<const float*>(fc->fields[i].data)[0];
      gLogVerbose << "Building alpha: " << alpha << std::endl;

      if (alpha == 0.0) {
        gLogError << "Alpha cannot be 0" << std::endl;
        assert(0);
      }
    }
  }

  DataType type = static_cast<DataType>(data_type_id);
  return new CeluPlugin(string(name), type, alpha);
}

nvinfer1::IPluginV2DynamicExt* CeluPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                                    size_t serialLength) {
  return new CeluPlugin(serialData, serialLength);
}

}  // namespace plugin
}  // namespace nvinfer1
