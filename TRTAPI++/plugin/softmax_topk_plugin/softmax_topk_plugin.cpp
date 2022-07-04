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

#include "softmax_topk_plugin.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <numeric>

#include "common.h"
#include "softmax_topk_kernel.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

REGISTER_TENSORRT_PLUGIN(SoftmaxTopKCreator);

SoftmaxTopKPlugin::SoftmaxTopKPlugin(const std::string name, const nvinfer1::DataType type, const int axis_dim,
                                     const int k)
    : layer_name_(name), data_type_(type), axis_dim_(axis_dim), k_(k) {}

SoftmaxTopKPlugin::SoftmaxTopKPlugin(void const* serial_data, size_t serial_length) {
  deserialize_value(&serial_data, &serial_length, &data_type_);
  deserialize_value(&serial_data, &serial_length, &axis_dim_);
  deserialize_value(&serial_data, &serial_length, &k_);

  int tmp = 0;
  deserialize_value(&serial_data, &serial_length, &tmp);
  deserialize_value(&serial_data, &serial_length, &tmp);
  deserialize_value(&serial_data, &serial_length, &tmp);
}

size_t SoftmaxTopKPlugin::getSerializationSize() const TRTNOEXCEPT {
  return sizeof(data_type_) + sizeof(axis_dim_) + sizeof(k_) + sizeof(int) * 3;
}

void SoftmaxTopKPlugin::serialize(void* buffer) const TRTNOEXCEPT {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, axis_dim_);
  serialize_value(&buffer, k_);

  int tmp = 0;
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);
}

bool SoftmaxTopKPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
                                                  int nb_outputs) TRTNOEXCEPT {
  assert(nb_inputs == 2);
  assert(nb_outputs == 2);

  // input, mask, value, idx
  const PluginTensorDesc& inout = in_out[pos];
  if (pos == 0 || pos == 2) return (inout.type == data_type_) && (inout.format == TensorFormat::kLINEAR);

  if (pos == 1 || pos == 3)
    return (inout.type == nvinfer1::DataType::kINT32) && (inout.format == TensorFormat::kLINEAR);

  return false;
}

nvinfer1::DataType SoftmaxTopKPlugin::getOutputDataType(int index, const nvinfer1::DataType* input_types,
                                                        int nb_inputs) const TRTNOEXCEPT {
  ASSERT(input_types && nb_inputs > 0);
  // input, mask, value, idx
  return input_types[index];
}

int SoftmaxTopKPlugin::initialize() TRTNOEXCEPT { return 0; }

void SoftmaxTopKPlugin::terminate() TRTNOEXCEPT {}

void SoftmaxTopKPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nb_inputs,
                                        const nvinfer1::DynamicPluginTensorDesc* out, int nb_outputs) TRTNOEXCEPT {}

int SoftmaxTopKPlugin::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                               const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
                               void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT {
  int nb_dims = input_desc[0].dims.nbDims;

  if (nb_dims != 3) {
    LOG(ERROR) << "nbDims != 3 not support! " << endl;
    assert(0);
  }

  auto d = input_desc[0].dims.d;

  // int batch = accumulate(d, d + axis_dim_, 1, std::multiplies<int>());
  // int seq_len = d[axis_dim_];
  // int dim = accumulate(d + axis_dim_ + 1, d + nb_dims, 1, std::multiplies<int>());

  int batch = d[0];
  int seq_len = d[1];
  int dim = d[2];

  int ret = -1;

  if (data_type_ == DataType::kFLOAT) {
    const auto input = static_cast<const float*>(inputs[0]);
    const auto mask = static_cast<const int*>(inputs[1]);
    auto values = static_cast<float*>(outputs[0]);
    auto idxs = static_cast<int*>(outputs[1]);

    ret = ComputeSoftmaxAndTop1(input, mask, batch, seq_len, dim, dim, idxs, values, stream);
  } else if (data_type_ == DataType::kHALF) {
    const auto input = static_cast<const half*>(inputs[0]);
    const auto mask = static_cast<const int*>(inputs[1]);
    auto values = static_cast<half*>(outputs[0]);
    auto idxs = static_cast<int*>(outputs[1]);

    ret = ComputeSoftmaxAndTop1(input, mask, batch, seq_len, dim, dim, idxs, values, stream);
  }

  return ret;
}

nvinfer1::DimsExprs SoftmaxTopKPlugin::getOutputDimensions(int output_index, const nvinfer1::DimsExprs* inputs,
                                                           int nb_inputs,
                                                           nvinfer1::IExprBuilder& expr_builder) TRTNOEXCEPT {
  auto output = inputs[0];
  // output.d[output.nbDims - 1] = expr_builder.constant(k_);
  output.d[output.nbDims - 1] = expr_builder.constant(1);
  return output;
}

nvinfer1::IPluginV2DynamicExt* SoftmaxTopKPlugin::clone() const TRTNOEXCEPT {
  return new SoftmaxTopKPlugin(layer_name_, data_type_, axis_dim_, k_);
}

void SoftmaxTopKPlugin::destroy() TRTNOEXCEPT { delete this; }

const char* SoftmaxTopKPlugin::getPluginVersion() const TRTNOEXCEPT { return SOFTMAX_TOPK_PLUGIN_VERSION; }

const char* SoftmaxTopKPlugin::getPluginType() const TRTNOEXCEPT { return SOFTMAX_TOPK_PLUGIN_NAME; }

size_t SoftmaxTopKPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nb_inputs*/,
                                           const nvinfer1::PluginTensorDesc* /*outputs*/,
                                           int /*nb_outputs*/) const TRTNOEXCEPT {
  return 0;
}

const char* SoftmaxTopKPlugin::getPluginNamespace() const TRTNOEXCEPT { return ""; }

int SoftmaxTopKPlugin::getNbOutputs() const TRTNOEXCEPT { return 2; }

void SoftmaxTopKPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                                        nvinfer1::IGpuAllocator* allocator) TRTNOEXCEPT {}

const char* SoftmaxTopKCreator::getPluginName() const TRTNOEXCEPT { return SOFTMAX_TOPK_PLUGIN_NAME; }

const char* SoftmaxTopKCreator::getPluginVersion() const TRTNOEXCEPT { return SOFTMAX_TOPK_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* SoftmaxTopKCreator::getFieldNames() TRTNOEXCEPT { return nullptr; }

nvinfer1::IPluginV2DynamicExt* SoftmaxTopKCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* field_collection) TRTNOEXCEPT {
  assert(field_collection->nbFields >= 1);

  LOG(INFO) << "Creating SoftmaxTopKPlugin...\n";

  int data_type_id = 0, axis_dim = -1, k = 1;

  for (int i = 0; i < field_collection->nbFields; i++) {
    std::string field_name(field_collection->fields[i].name);

    if (field_name.compare("data_type") == 0) {
      data_type_id = static_cast<const int*>(field_collection->fields[i].data)[0];
      LOG(INFO) << "Building data_type_id : " << data_type_id << std::endl;

      if (data_type_id < 0 || data_type_id > 3) {
        LOG(ERROR) << "Invalid type id" << data_type_id << std::endl;
        assert(0);
      }
    }

    // if (field_name.compare("axis_dim") == 0) {
    // axis_dim = static_cast<const int*>(field_collection->fields[i].data)[0];
    // LOG(INFO) << "Building axis_dim: " << axis_dim << std::endl;
    //}

    // if (field_name.compare("k") == 0) {
    // k = static_cast<const int*>(field_collection->fields[i].data)[0];
    // LOG(INFO) << "Building k: " << k << std::endl;
    //}
  }

  LOG(WARNING) << "Only support dim=-1 and k=0 now!" << std::endl;

  DataType type = static_cast<DataType>(data_type_id);
  return new SoftmaxTopKPlugin(string(name), type, axis_dim, k);
}

nvinfer1::IPluginV2DynamicExt* SoftmaxTopKCreator::deserializePlugin(const char* name, const void* serial_data,
                                                                     size_t serial_length) TRTNOEXCEPT {
  return new SoftmaxTopKPlugin(serial_data, serial_length);
}

}  // namespace plugin
}  // namespace nvinfer1
