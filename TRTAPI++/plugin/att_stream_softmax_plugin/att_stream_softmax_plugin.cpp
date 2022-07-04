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

#include "att_stream_softmax_plugin.h"

#include <cuda_fp16.h>
#include <algorithm>

#include "att_stream_softmax_kernel.h"
#include "debug.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
// PluginFieldCollection AttStreamSoftmaxPluginCreator::mFC{};
// std::vector<PluginField> AttStreamSoftmaxPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AttStreamSoftmaxPluginCreator);

AttStreamSoftmaxPlugin::AttStreamSoftmaxPlugin(const std::string name, const nvinfer1::DataType type, const float scale,
                                               const int cache_len)
    : layer_name_(name), data_type_(type), scale_(scale), cache_len_(cache_len) {}

AttStreamSoftmaxPlugin::AttStreamSoftmaxPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &scale_);
  deserialize_value(&serialData, &serialLength, &cache_len_);

  int tmp = 0;
  deserialize_value(&serialData, &serialLength, &tmp);
  deserialize_value(&serialData, &serialLength, &tmp);
  deserialize_value(&serialData, &serialLength, &tmp);

  // scale_fp16_ = __float2half(scale_);
}

size_t AttStreamSoftmaxPlugin::getSerializationSize() const TRTNOEXCEPT {
  return sizeof(data_type_) + sizeof(scale_) + sizeof(cache_len_) + sizeof(int) * 3;
}

void AttStreamSoftmaxPlugin::serialize(void* buffer) const TRTNOEXCEPT {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, scale_);
  serialize_value(&buffer, cache_len_);

  int tmp = 0;
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);
}

bool AttStreamSoftmaxPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                                       int nbOutputs) TRTNOEXCEPT {
  assert(nbInputs == 3);
  assert(nbOutputs == 1);

  const PluginTensorDesc& input = inOut[0];
  // input, float or half
  if (pos == 0) return (input.type == data_type_) && (input.format == TensorFormat::kLINEAR);

  // decode_frame_num and input_len, int32
  if (pos == 1 || pos == 2) {
    const PluginTensorDesc& idx = inOut[pos];
    return (idx.type == DataType::kINT32) && (idx.format == TensorFormat::kLINEAR);
  }

  // output, fp32 or half
  if (pos == 3) {
    const PluginTensorDesc& output = inOut[pos];
    return (input.type == output.type) && (output.format == TensorFormat::kLINEAR);
  }
  return false;
}

nvinfer1::DataType AttStreamSoftmaxPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                             int nbInputs) const TRTNOEXCEPT {
  ASSERT(inputTypes && nbInputs > 0);
  return inputTypes[0];
}

int AttStreamSoftmaxPlugin::initialize() TRTNOEXCEPT { return 0; }

void AttStreamSoftmaxPlugin::terminate() TRTNOEXCEPT {}

void AttStreamSoftmaxPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                             const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) TRTNOEXCEPT {}

int AttStreamSoftmaxPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                                    void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT {
  auto batch = inputDesc[0].dims.d[0];
  auto N = inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2];
  auto ld = inputDesc[0].dims.d[3];

  if (data_type_ == DataType::kFLOAT) {
    const auto input = static_cast<const float*>(inputs[0]);
    const auto decode_frame_num = static_cast<const int*>(inputs[1]);
    const auto mask_idx = static_cast<const int*>(inputs[2]);
    auto output = static_cast<float*>(outputs[0]);

    auto ret =
        ComputeAttStreamSoftmax(stream, ld, batch, N, scale_, cache_len_, decode_frame_num, mask_idx, input, output);

    // print_data(input, ld, "input");
    // print_data(output, ld, "output");

    return ret;
  } else if (data_type_ == DataType::kHALF) {
    // TODO(leowgyang): tes not pass
    const auto input = static_cast<const half*>(inputs[0]);
    const auto decode_frame_num = static_cast<const int*>(inputs[1]);
    const auto mask_idx = static_cast<const int*>(inputs[2]);
    auto output = static_cast<half*>(outputs[0]);

    // printf("in_ptr=%d, out_ptr = %d\n", input, output);
    // print_data(input, 10, "input");
    // print_data(output, 10, "output");

    auto ret =
        ComputeAttStreamSoftmax(stream, ld, batch, N, scale_, cache_len_, decode_frame_num, mask_idx, input, output);
    return ret;
  }

  return -1;
}

nvinfer1::DimsExprs AttStreamSoftmaxPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                                int nbInputs,
                                                                nvinfer1::IExprBuilder& exprBuilder) TRTNOEXCEPT {
  return inputs[0];
}

nvinfer1::IPluginV2DynamicExt* AttStreamSoftmaxPlugin::clone() const TRTNOEXCEPT {
  return new AttStreamSoftmaxPlugin(layer_name_, data_type_, scale_, cache_len_);
}

void AttStreamSoftmaxPlugin::destroy() TRTNOEXCEPT { delete this; }

const char* AttStreamSoftmaxPlugin::getPluginVersion() const TRTNOEXCEPT { return ATT_STREAM_SOFTMAX_PLUGIN_VERSION; }

const char* AttStreamSoftmaxPlugin::getPluginType() const TRTNOEXCEPT { return ATT_STREAM_SOFTMAX_PLUGIN_NAME; }

size_t AttStreamSoftmaxPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
                                                const nvinfer1::PluginTensorDesc* /*outputs*/,
                                                int /*nbOutputs*/) const TRTNOEXCEPT {
  return 0;
}

const char* AttStreamSoftmaxPlugin::getPluginNamespace() const TRTNOEXCEPT { return ""; }

int AttStreamSoftmaxPlugin::getNbOutputs() const TRTNOEXCEPT { return 1; }

void AttStreamSoftmaxPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                                             nvinfer1::IGpuAllocator* allocator) TRTNOEXCEPT {}

const char* AttStreamSoftmaxPluginCreator::getPluginName() const TRTNOEXCEPT { return ATT_STREAM_SOFTMAX_PLUGIN_NAME; }

const char* AttStreamSoftmaxPluginCreator::getPluginVersion() const TRTNOEXCEPT {
  return ATT_STREAM_SOFTMAX_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* AttStreamSoftmaxPluginCreator::getFieldNames() TRTNOEXCEPT {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

nvinfer1::IPluginV2DynamicExt* AttStreamSoftmaxPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) TRTNOEXCEPT {
  assert(fc->nbFields == 3);

  LOG(INFO) << "Creating AttStreamSoftmaxPlugin...\n";

  int data_type_id;
  float scale;
  int cache_len;

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("data_type") == 0) {
      data_type_id = static_cast<const int*>(fc->fields[i].data)[0];
      LOG(INFO) << "Building data_type_id : " << data_type_id << std::endl;
    }
    if (field_name.compare("scale") == 0) {
      scale = static_cast<const float*>(fc->fields[i].data)[0];
      LOG(INFO) << "Building scale : " << scale << std::endl;
    }
    if (field_name.compare("cache_len") == 0) {
      cache_len = static_cast<const int*>(fc->fields[i].data)[0];
      LOG(INFO) << "Building cache_len : " << cache_len << std::endl;
    }
  }

  if (data_type_id < 0 || data_type_id > 3) {
    LOG(ERROR) << "Invalid type id" << data_type_id << std::endl;
    assert(0);
  }

  DataType type = static_cast<DataType>(data_type_id);
  return new AttStreamSoftmaxPlugin(string(name), type, scale, cache_len);
}

nvinfer1::IPluginV2DynamicExt* AttStreamSoftmaxPluginCreator::deserializePlugin(const char* name,
                                                                                const void* serialData,
                                                                                size_t serialLength) TRTNOEXCEPT {
  return new AttStreamSoftmaxPlugin(serialData, serialLength);
}

}  // namespace plugin
}  // namespace nvinfer1
