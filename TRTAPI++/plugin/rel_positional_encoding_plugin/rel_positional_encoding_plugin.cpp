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

#include "rel_positional_encoding_plugin.h"

#include <cuda_fp16.h>
#include <algorithm>

#include "common.h"
#include "debug.h"
#include "rel_positional_encoding_kernel.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
// PluginFieldCollection RelPositionalEncodingPluginCreator::mFC{};
// std::vector<PluginField> RelPositionalEncodingPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(RelPositionalEncodingPluginCreator);

RelPositionalEncodingPlugin::RelPositionalEncodingPlugin(const std::string name, const nvinfer1::DataType type,
                                                         const float scale, const int max_len, const int dim,
                                                         const int streaming)
    : layer_name_(name), data_type_(type), scale_(scale), max_len_(max_len), dim_(dim), streaming_(streaming) {
  scale_fp16_ = __float2half(scale_);

  // pe_.convertAndCopy(pe, data_type_);
  // if (!is_build_)
  // copyToDevice(pe_, getWeightsSize(pe_, data_type_), pe_dev_ptr_);
  // printf("RelPositionalEncodingPlugin::RelPositionalEncodingPlugin\n");
}

RelPositionalEncodingPlugin::RelPositionalEncodingPlugin(void const* serialData, size_t serialLength) {
  // printf("RelPositionalEncodingPlugin::deserialize_value\n");
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &scale_);
  deserialize_value(&serialData, &serialLength, &max_len_);
  deserialize_value(&serialData, &serialLength, &dim_);
  deserialize_value(&serialData, &serialLength, &streaming_);

  // extra 8 bit(2 int) for compatibility
  int tmp = 0;
  deserialize_value(&serialData, &serialLength, &tmp);
  deserialize_value(&serialData, &serialLength, &tmp);

  scale_fp16_ = __float2half(scale_);

  // const char* d = static_cast<const char*>(serialData);

  // pe_.convertAndCopy(d, max_len_ * dim_, data_type_);
  // copyToDevice(pe_, getWeightsSize(pe_, data_type_), pe_dev_ptr_);

  // is_build_ = false;
}

size_t RelPositionalEncodingPlugin::getSerializationSize() const TRTNOEXCEPT {
  // size_t word_size = getElementSize(data_type_);
  // return word_size * max_len_ * dim_ +
  return sizeof(data_type_) + sizeof(scale_) + sizeof(max_len_) + sizeof(dim_) + sizeof(streaming_) + sizeof(int) * 2;
}

void RelPositionalEncodingPlugin::serialize(void* buffer) const TRTNOEXCEPT {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, scale_);
  serialize_value(&buffer, max_len_);
  serialize_value(&buffer, dim_);
  serialize_value(&buffer, streaming_);

  // extra 8 bit(2 int) for compatibility
  int tmp = 0;
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);

  // size_t word_size = getElementSize(data_type_);
  // char* d = static_cast<char*>(buffer);
  // serFromHost(d, static_cast<const char*>(pe_.values), max_len_ * dim_ * word_size);
}

bool RelPositionalEncodingPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
                                                            int nbInputs, int nbOutputs) TRTNOEXCEPT {
  if (streaming_) {
    assert(nbInputs == 3);
    assert(nbOutputs == 2);

    // input, w_pe, frame_num_input, output and pos_emb
    const PluginTensorDesc& in_out = inOut[pos];
    if (pos == 2) {
      return (in_out.type == DataType::kINT32) && (in_out.format == TensorFormat::kLINEAR);
    } else {
      return (in_out.type == data_type_) && (in_out.format == TensorFormat::kLINEAR);
    }

  } else {
    // non streaming
    // input, w_pe, output and pos_emb
    assert(nbInputs == 2);
    assert(nbOutputs == 2);

    const PluginTensorDesc& in_out = inOut[pos];
    return (in_out.type == data_type_) && (in_out.format == TensorFormat::kLINEAR);
  }

  return false;
}

nvinfer1::DataType RelPositionalEncodingPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                                  int nbInputs) const TRTNOEXCEPT {
  ASSERT(inputTypes && nbInputs > 0);
  // output && pos_emb
  if (index == 0 || index == 1) return inputTypes[0];
}

int RelPositionalEncodingPlugin::initialize() TRTNOEXCEPT { return 0; }

void RelPositionalEncodingPlugin::terminate() TRTNOEXCEPT {}

void RelPositionalEncodingPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                                  const nvinfer1::DynamicPluginTensorDesc* out,
                                                  int nbOutputs) TRTNOEXCEPT {}

int RelPositionalEncodingPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                         const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                                         void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT {
  // printf("RelPositionalEncodingPlugin::enqueue\n");
  auto batch = inputDesc[0].dims.d[0];
  auto seq_len = inputDesc[0].dims.d[1];
  auto dim = inputDesc[0].dims.d[2];

  if (seq_len >= max_len_) {
    LOG(ERROR) << "seq_len:" << seq_len << " >= max_len:" << max_len_ << endl;
    assert(0);
  }

  int ret = -1;

  if (data_type_ == DataType::kFLOAT) {
    const auto input = static_cast<const float*>(inputs[0]);
    const auto pe = static_cast<const float*>(inputs[1]);
    auto output = static_cast<float*>(outputs[0]);
    auto pos_output = static_cast<float*>(outputs[1]);

    if (!streaming_) {
      // non stream
      ret = compute_rel_positional_encoding(input, pe, scale_, batch, seq_len, dim, output, pos_output, stream);
    } else {
      const auto frame_num_input = static_cast<const int*>(inputs[2]);
      ret = compute_rel_positional_encoding_streaming(input, pe, frame_num_input, scale_, batch, seq_len, dim, output,
                                                      pos_output, stream);
    }
  } else if (data_type_ == DataType::kHALF) {
    const auto input = static_cast<const half*>(inputs[0]);
    const auto pe = static_cast<const half*>(inputs[1]);
    auto output = static_cast<half*>(outputs[0]);
    auto pos_output = static_cast<half*>(outputs[1]);

    if (!streaming_) {
      // non stream
      ret = compute_rel_positional_encoding(input, pe, scale_, batch, seq_len, dim, output, pos_output, stream);
    } else {
      const auto frame_num_input = static_cast<const int*>(inputs[2]);
      ret = compute_rel_positional_encoding_streaming(input, pe, frame_num_input, scale_fp16_, batch, seq_len, dim,
                                                      output, pos_output, stream);
    }
  } else {
    assert(0);
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

  return ret;
}

nvinfer1::DimsExprs RelPositionalEncodingPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                                     int nbInputs,
                                                                     nvinfer1::IExprBuilder& exprBuilder) TRTNOEXCEPT {
  nvinfer1::DimsExprs output;
  if (outputIndex == 0) {
    output = inputs[0];
  } else if (outputIndex == 1) {
    output.nbDims = 3;
    output.d[0] = exprBuilder.constant(1);
    output.d[1] = inputs[0].d[1];
    output.d[2] = exprBuilder.constant(dim_);
  } else if (outputIndex == 2) {
    // for streaming, frame_output
    assert(nbInputs == 3);
    output.nbDims = 2;
    output.d[0] = exprBuilder.constant(1);
    output.d[1] = inputs[0].d[0];
  } else {
    LOG(ERROR) << "outputIndex >1" << endl;
    assert(0);
  }
  return output;
}

nvinfer1::IPluginV2DynamicExt* RelPositionalEncodingPlugin::clone() const TRTNOEXCEPT {
  // printf("RelPositionalEncodingPlugin::clone\n");
  return new RelPositionalEncodingPlugin(layer_name_, data_type_, scale_, max_len_, dim_, streaming_);
}

void RelPositionalEncodingPlugin::destroy() TRTNOEXCEPT { delete this; }

const char* RelPositionalEncodingPlugin::getPluginVersion() const TRTNOEXCEPT {
  return RELPOSITIONAL_ENCODING_PLUGIN_VERSION;
}

const char* RelPositionalEncodingPlugin::getPluginType() const TRTNOEXCEPT {
  return RELPOSITIONAL_ENCODING_PLUGIN_NAME;
}

size_t RelPositionalEncodingPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
                                                     const nvinfer1::PluginTensorDesc* /*outputs*/,
                                                     int /*nbOutputs*/) const TRTNOEXCEPT {
  return 0;
}

const char* RelPositionalEncodingPlugin::getPluginNamespace() const TRTNOEXCEPT { return ""; }

int RelPositionalEncodingPlugin::getNbOutputs() const TRTNOEXCEPT { return 2; }

void RelPositionalEncodingPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                                                  nvinfer1::IGpuAllocator* allocator) TRTNOEXCEPT {
  cublas_handle_ = cublas;
}

const char* RelPositionalEncodingPluginCreator::getPluginName() const TRTNOEXCEPT {
  return RELPOSITIONAL_ENCODING_PLUGIN_NAME;
}

const char* RelPositionalEncodingPluginCreator::getPluginVersion() const TRTNOEXCEPT {
  return RELPOSITIONAL_ENCODING_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* RelPositionalEncodingPluginCreator::getFieldNames() TRTNOEXCEPT {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

nvinfer1::IPluginV2DynamicExt* RelPositionalEncodingPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) TRTNOEXCEPT {
  assert(fc->nbFields == 4 || fc->nbFields == 5);

  LOG(INFO) << "Creating RelPositionalEncodingPlugiRelPositionalEncodingPluginn...\n";

  int data_type_id;
  float scale;
  int max_len;
  int dim;
  int streaming = 0;  // default is no streaming

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("data_type") == 0) {
      data_type_id = static_cast<const int*>(fc->fields[i].data)[0];
      LOG(INFO) << "Building data_type_id : " << data_type_id << std::endl;
    }

    if (field_name.compare("scale") == 0) {
      scale = static_cast<const float*>(fc->fields[i].data)[0];
      LOG(INFO) << "Building scale: " << scale << std::endl;
    }

    if (field_name.compare("max_len") == 0) {
      max_len = static_cast<const int*>(fc->fields[i].data)[0];
      LOG(INFO) << "Building max_len: " << max_len << std::endl;
    }

    if (field_name.compare("dim") == 0) {
      dim = static_cast<const int*>(fc->fields[i].data)[0];
      LOG(INFO) << "Building dim: " << dim << std::endl;
    }

    if (field_name.compare("streaming") == 0) {
      streaming = static_cast<const int*>(fc->fields[i].data)[0];
      LOG(INFO) << "Building streaming: " << streaming << std::endl;
    }
  }

  if (dim <= 0) {
    LOG(ERROR) << "Invalid output dimension" << std::endl;
    assert(0);
  }
  if (data_type_id != 0 && data_type_id != 1) {
    LOG(ERROR) << "Invalid type id" << data_type_id << std::endl;
    assert(0);
  }

  DataType type = static_cast<DataType>(data_type_id);
  return new RelPositionalEncodingPlugin(string(name), type, scale, max_len, dim, streaming);
}

nvinfer1::IPluginV2DynamicExt* RelPositionalEncodingPluginCreator::deserializePlugin(const char* name,
                                                                                     const void* serialData,
                                                                                     size_t serialLength) TRTNOEXCEPT {
  return new RelPositionalEncodingPlugin(serialData, serialLength);
}

}  // namespace plugin
}  // namespace nvinfer1
