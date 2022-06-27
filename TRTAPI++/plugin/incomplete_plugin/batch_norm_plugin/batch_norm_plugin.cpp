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
#include "batch_norm_plugin.h"

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
// PluginFieldCollection BatchNormPluginCreator::mFC{};
// std::vector<PluginField> BatchNormPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(BatchNormPluginCreator);

BatchNormPlugin::BatchNormPlugin(const std::string name, const nvinfer1::DataType type, const nvinfer1::Weights& weight,
                                 const nvinfer1::Weights& bias, const nvinfer1::Weights& mean,
                                 const nvinfer1::Weights& var, float eps)
    : layer_name_(name), data_type_(type), eps_(eps) {
  weight_.convertAndCopy(weight, data_type_);
  copyToDevice(weight_, getWeightsSize(weight_, data_type_), weight_dev_ptr_);

  bias_.convertAndCopy(bias, data_type_);
  copyToDevice(bias_, getWeightsSize(bias_, data_type_), bias_dev_ptr_);

  mean_.convertAndCopy(mean, data_type_);
  copyToDevice(mean_, getWeightsSize(mean_, data_type_), mean_dev_ptr_);

  var_.convertAndCopy(var, data_type_);
  copyToDevice(var_, getWeightsSize(var_, data_type_), var_dev_ptr_);

  channel_num_ = weight_.count;
}

BatchNormPlugin::BatchNormPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &channel_num_);
  deserialize_value(&serialData, &serialLength, &eps_);

  const char* d = static_cast<const char*>(serialData);

  weight_.convertAndCopy(d, channel_num_, data_type_);
  copyToDevice(weight_, getWeightsSize(weight_, data_type_), weight_dev_ptr_);

  bias_.convertAndCopy(d, channel_num_, data_type_);
  copyToDevice(bias_, getWeightsSize(bias_, data_type_), bias_dev_ptr_);

  mean_.convertAndCopy(d, channel_num_, data_type_);
  copyToDevice(mean_, getWeightsSize(mean_, data_type_), mean_dev_ptr_);

  var_.convertAndCopy(d, channel_num_, data_type_);
  copyToDevice(var_, getWeightsSize(var_, data_type_), var_dev_ptr_);
}

size_t BatchNormPlugin::getSerializationSize() const {
  size_t word_size = getElementSize(data_type_);
  int element_num = channel_num_ * 4;
  return sizeof(data_type_) + element_num * word_size + sizeof(channel_num_) + sizeof(eps_);
}

void BatchNormPlugin::serialize(void* buffer) const {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, channel_num_);
  serialize_value(&buffer, eps_);

  size_t word_size = getElementSize(data_type_);
  char* d = static_cast<char*>(buffer);
  serFromDev(d, static_cast<char*>(weight_dev_ptr_.get()), channel_num_ * word_size);
  serFromDev(d, static_cast<char*>(bias_dev_ptr_.get()), channel_num_ * word_size);
  serFromDev(d, static_cast<char*>(mean_dev_ptr_.get()), channel_num_ * word_size);
  serFromDev(d, static_cast<char*>(var_dev_ptr_.get()), channel_num_ * word_size);
}

bool BatchNormPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
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

nvinfer1::DataType BatchNormPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                      int nbInputs) const {
  ASSERT(inputTypes && nbInputs > 0);
  return inputTypes[0];
}

int BatchNormPlugin::initialize() { return 0; }

void BatchNormPlugin::terminate() {}

void BatchNormPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                      const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {}

int BatchNormPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                             const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) {
  std::vector<int64_t> tensor_dims;
  for (int i = 0; i < inputDesc->dims.nbDims; i++) {
    tensor_dims.push_back(inputDesc->dims.d[i]);
  }

  torch::Tensor input, output, weight, bias, mean, var;
  if (data_type_ == DataType::kHALF) {
    auto type = at::device(at::kCUDA).dtype(torch::kFloat16);
    input = at::from_blob((void*)inputs[0], tensor_dims, type);
    output = at::from_blob(outputs[0], tensor_dims, type);
    weight = at::from_blob((void*)weight_dev_ptr_.get(), {channel_num_}, type);
    bias = at::from_blob((void*)bias_dev_ptr_.get(), {channel_num_}, type);
    mean = at::from_blob((void*)mean_dev_ptr_.get(), {channel_num_}, type);
    var = at::from_blob((void*)var_dev_ptr_.get(), {channel_num_}, type);
  } else {
    auto type = at::device(at::kCUDA).dtype(torch::kFloat);
    input = at::from_blob((void*)inputs[0], tensor_dims, type);
    output = at::from_blob(outputs[0], tensor_dims, type);
    weight = at::from_blob((void*)weight_dev_ptr_.get(), {channel_num_}, type);
    bias = at::from_blob((void*)bias_dev_ptr_.get(), {channel_num_}, type);
    mean = at::from_blob((void*)mean_dev_ptr_.get(), {channel_num_}, type);
    var = at::from_blob((void*)var_dev_ptr_.get(), {channel_num_}, type);
  }

  c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard torch_guard(torch_stream);

  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, stream);

  cudaStreamWaitEvent(torch_stream.stream(), event, 0);

  auto result =
      at::batch_norm(input, weight, bias, mean, var, false, 0.1, eps_, at::globalContext().userEnabledCuDNN());
  output.copy_(result);

  cudaEvent_t torch_event;
  cudaEventCreate(&torch_event);
  cudaEventRecord(torch_event, torch_stream.stream());

  cudaStreamWaitEvent(stream, torch_event, 0);

  cudaEventDestroy(event);
  cudaEventDestroy(torch_event);
  return 0;
}

nvinfer1::DimsExprs BatchNormPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                         int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
  return inputs[0];
}

nvinfer1::IPluginV2DynamicExt* BatchNormPlugin::clone() const {
  return new BatchNormPlugin(layer_name_, data_type_, weight_, bias_, mean_, var_, eps_);
}

void BatchNormPlugin::destroy() {
  // gLogVerbose << "BatchNormPlugin destroy\n";
  // This gets called when the network containing plugin is destroyed
  weight_dev_ptr_.release();
  bias_dev_ptr_.release();
  mean_dev_ptr_.release();
  var_dev_ptr_.release();
}

const char* BatchNormPlugin::getPluginVersion() const { return BATCH_NORM_PLUGIN_VERSION; }

const char* BatchNormPlugin::getPluginType() const { return BATCH_NORM_PLUGIN_NAME; }

size_t BatchNormPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
                                         const nvinfer1::PluginTensorDesc* /*outputs*/,
                                         int /*nbOutputs*/) const TRTNOEXCEPT {
  return 0;
}

const char* BatchNormPlugin::getPluginNamespace() const { return ""; }

int BatchNormPlugin::getNbOutputs() const { return 1; }

void BatchNormPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) {}

const char* BatchNormPluginCreator::getPluginName() const { return BATCH_NORM_PLUGIN_NAME; }

const char* BatchNormPluginCreator::getPluginVersion() const { return BATCH_NORM_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* BatchNormPluginCreator::getFieldNames() {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

nvinfer1::IPluginV2DynamicExt* BatchNormPluginCreator::createPlugin(const char* name,
                                                                    const nvinfer1::PluginFieldCollection* fc) {
  assert(fc->nbFields == 6 || fc->nbFields == 5);

  gLogVerbose << "Creating BatchNormPlugin...\n";

  int data_type_id = 0;
  Weights weight;
  Weights bias;
  Weights mean;
  Weights var;
  float eps = 1e-5;

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

    if (field_name.compare("mean") == 0) {
      gLogVerbose << "Building mean...\n";
      mean.values = fc->fields[i].data;
      mean.count = fc->fields[i].length;
      mean.type = fieldTypeToDataType(fc->fields[i].type);
      gLogVerbose << "Is mean float32: " << (mean.type == DataType::kFLOAT) << std::endl;
    }

    if (field_name.compare("var") == 0) {
      gLogVerbose << "Building var...\n";
      var.values = fc->fields[i].data;
      var.count = fc->fields[i].length;
      var.type = fieldTypeToDataType(fc->fields[i].type);
      gLogVerbose << "Is var float32: " << (var.type == DataType::kFLOAT) << std::endl;
    }

    if (field_name.compare("eps") == 0) {
      eps = static_cast<const float*>(fc->fields[i].data)[0];
      gLogVerbose << "Building eps: " << eps << std::endl;
    }
  }

  DataType type = static_cast<DataType>(data_type_id);
  return new BatchNormPlugin(string(name), type, weight, bias, mean, var, eps);
}

nvinfer1::IPluginV2DynamicExt* BatchNormPluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                                         size_t serialLength) {
  return new BatchNormPlugin(serialData, serialLength);
}

}  // namespace plugin
}  // namespace nvinfer1
