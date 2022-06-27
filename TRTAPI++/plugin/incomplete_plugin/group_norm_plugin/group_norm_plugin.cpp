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
#include "group_norm_plugin.h"

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
// PluginFieldCollection GroupNormPluginCreator::mFC{};
// std::vector<PluginField> GroupNormPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GroupNormPluginCreator);

GroupNormPlugin::GroupNormPlugin(const std::string name, const nvinfer1::DataType type, int num_groups,
                                 const nvinfer1::Weights& weight, const nvinfer1::Weights& bias, float eps)
    : layer_name_(name), data_type_(type), num_groups_(num_groups), eps_(eps) {
  weight_.convertAndCopy(weight, data_type_);
  copyToDevice(weight_, getWeightsSize(weight_, data_type_), weight_dev_ptr_);

  bias_.convertAndCopy(bias, data_type_);
  copyToDevice(bias_, getWeightsSize(bias_, data_type_), bias_dev_ptr_);

  weight_count_ = weight_.count;
}

GroupNormPlugin::GroupNormPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &num_groups_);
  deserialize_value(&serialData, &serialLength, &weight_count_);
  deserialize_value(&serialData, &serialLength, &eps_);

  const char* d = static_cast<const char*>(serialData);

  weight_.convertAndCopy(d, weight_count_, data_type_);
  copyToDevice(weight_, getWeightsSize(weight_, data_type_), weight_dev_ptr_);

  bias_.convertAndCopy(d, weight_count_, data_type_);
  copyToDevice(bias_, getWeightsSize(bias_, data_type_), bias_dev_ptr_);
}

size_t GroupNormPlugin::getSerializationSize() const {
  size_t word_size = getElementSize(data_type_);
  int element_num = weight_count_ * 2;
  return sizeof(data_type_) + sizeof(num_groups_) + sizeof(weight_count_) + element_num * word_size + sizeof(eps_);
}

void GroupNormPlugin::serialize(void* buffer) const {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, num_groups_);
  serialize_value(&buffer, weight_count_);
  serialize_value(&buffer, eps_);

  size_t word_size = getElementSize(data_type_);
  char* d = static_cast<char*>(buffer);
  serFromDev(d, static_cast<char*>(weight_dev_ptr_.get()), weight_count_ * word_size);
  serFromDev(d, static_cast<char*>(bias_dev_ptr_.get()), weight_count_ * word_size);
}

bool GroupNormPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
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

nvinfer1::DataType GroupNormPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                      int nbInputs) const {
  ASSERT(inputTypes && nbInputs > 0);
  return inputTypes[0];
}

int GroupNormPlugin::initialize() { return 0; }

void GroupNormPlugin::terminate() {}

void GroupNormPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                      const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {}

int GroupNormPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                             const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) {
  std::vector<int64_t> tensor_dims;
  for (int i = 0; i < inputDesc->dims.nbDims; i++) {
    tensor_dims.push_back(inputDesc->dims.d[i]);
  }

  torch::Tensor input, output, weight, bias;
  if (data_type_ == DataType::kHALF) {
    auto type = at::device(at::kCUDA).dtype(torch::kFloat16);
    input = at::from_blob((void*)inputs[0], tensor_dims, type);
    output = at::from_blob(outputs[0], tensor_dims, type);
    weight = at::from_blob((void*)weight_dev_ptr_.get(), {weight_count_}, type);
    bias = at::from_blob((void*)bias_dev_ptr_.get(), {weight_count_}, type);
  } else {
    auto type = at::device(at::kCUDA).dtype(torch::kFloat);
    input = at::from_blob((void*)inputs[0], tensor_dims, type);
    output = at::from_blob(outputs[0], tensor_dims, type);
    weight = at::from_blob((void*)weight_dev_ptr_.get(), {weight_count_}, type);
    bias = at::from_blob((void*)bias_dev_ptr_.get(), {weight_count_}, type);
  }

  c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard torch_guard(torch_stream);

  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, stream);

  cudaStreamWaitEvent(torch_stream.stream(), event, 0);

  auto result = at::group_norm(input, num_groups_, weight, bias, eps_);
  output.copy_(result);

  cudaEvent_t torch_event;
  cudaEventCreate(&torch_event);
  cudaEventRecord(torch_event, torch_stream.stream());

  cudaStreamWaitEvent(stream, torch_event, 0);

  cudaEventDestroy(event);
  cudaEventDestroy(torch_event);
  return 0;
}

nvinfer1::DimsExprs GroupNormPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                         int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
  return inputs[0];
}

nvinfer1::IPluginV2DynamicExt* GroupNormPlugin::clone() const {
  return new GroupNormPlugin(layer_name_, data_type_, num_groups_, weight_, bias_, eps_);
}

void GroupNormPlugin::destroy() {
  // gLogVerbose << "GroupNormPlugin destroy\n";
  // This gets called when the network containing plugin is destroyed
  weight_dev_ptr_.release();
  bias_dev_ptr_.release();
}

const char* GroupNormPlugin::getPluginVersion() const { return GROUP_NORM_PLUGIN_VERSION; }

const char* GroupNormPlugin::getPluginType() const { return GROUP_NORM_PLUGIN_NAME; }

size_t GroupNormPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
                                         const nvinfer1::PluginTensorDesc* /*outputs*/,
                                         int /*nbOutputs*/) const TRTNOEXCEPT {
  return 0;
}

const char* GroupNormPlugin::getPluginNamespace() const { return ""; }

int GroupNormPlugin::getNbOutputs() const { return 1; }

void GroupNormPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) {}

const char* GroupNormPluginCreator::getPluginName() const { return GROUP_NORM_PLUGIN_NAME; }

const char* GroupNormPluginCreator::getPluginVersion() const { return GROUP_NORM_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* GroupNormPluginCreator::getFieldNames() {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

nvinfer1::IPluginV2DynamicExt* GroupNormPluginCreator::createPlugin(const char* name,
                                                                    const nvinfer1::PluginFieldCollection* fc) {
  assert(fc->nbFields == 5 || fc->nbFields == 4);

  gLogVerbose << "Creating GroupNormPlugin...\n";

  int data_type_id = 0;
  int num_groups;
  Weights weight;
  Weights bias;
  float eps = 1e-05;

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

    if (field_name.compare("num_groups") == 0) {
      num_groups = static_cast<const int*>(fc->fields[i].data)[0];
      gLogVerbose << "Building num_groups: " << num_groups << std::endl;
    }

    if (field_name.compare("weight") == 0) {
      gLogVerbose << "Building weight...\n";
      weight.values = fc->fields[i].data;
      weight.count = fc->fields[i].length;
      weight.type = fieldTypeToDataType(fc->fields[i].type);
      gLogVerbose << "Is weight float32: " << (weight.type == DataType::kFLOAT) << std::endl;
    }

    if (field_name.compare("bias") == 0) {
      gLogVerbose << "Building bias...\n";
      bias.values = fc->fields[i].data;
      bias.count = fc->fields[i].length;
      bias.type = fieldTypeToDataType(fc->fields[i].type);
      gLogVerbose << "Is bias float32: " << (bias.type == DataType::kFLOAT) << std::endl;
    }

    if (field_name.compare("eps") == 0) {
      eps = static_cast<const float*>(fc->fields[i].data)[0];
      gLogVerbose << "Building eps: " << eps << std::endl;
    }
  }

  DataType type = static_cast<DataType>(data_type_id);
  return new GroupNormPlugin(string(name), type, num_groups, weight, bias, eps);
}

nvinfer1::IPluginV2DynamicExt* GroupNormPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                                         size_t serialLength) {
  return new GroupNormPlugin(serialData, serialLength);
}

}  // namespace plugin
}  // namespace nvinfer1
