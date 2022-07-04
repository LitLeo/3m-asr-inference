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

#include "glu_plugin.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <numeric>

#include "common.h"
#include "glu_kernel.h"

// #include "debug.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
// PluginFieldCollection GluPluginCreator::mFC{};
// std::vector<PluginField> GluPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(GluPluginCreator);

GluPlugin::GluPlugin(const std::string name, const nvinfer1::DataType type, const int axis_dim)
    : layer_name_(name), data_type_(type), axis_dim_(axis_dim) {}

GluPlugin::GluPlugin(void const* serial_data, size_t serial_length) {
  deserialize_value(&serial_data, &serial_length, &data_type_);
  deserialize_value(&serial_data, &serial_length, &axis_dim_);

  int tmp = 0;
  deserialize_value(&serial_data, &serial_length, &tmp);
  deserialize_value(&serial_data, &serial_length, &tmp);
  deserialize_value(&serial_data, &serial_length, &tmp);
}

size_t GluPlugin::getSerializationSize() const TRTNOEXCEPT {
  return sizeof(data_type_) + sizeof(axis_dim_) + sizeof(int) * 3;
}

void GluPlugin::serialize(void* buffer) const TRTNOEXCEPT {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, axis_dim_);

  int tmp = 0;
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);
}

bool GluPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
                                          int nb_outputs) TRTNOEXCEPT {
  assert(nb_inputs == 1);
  assert(nb_outputs == 1);

  const PluginTensorDesc& input = in_out[0];
  if (pos == 0) return (input.type == data_type_) && (input.format == TensorFormat::kLINEAR);

  if (pos == 1) {
    const PluginTensorDesc& output = in_out[1];
    return (input.type == output.type) && (output.format == TensorFormat::kLINEAR);
  }
  return false;
}

nvinfer1::DataType GluPlugin::getOutputDataType(int index, const nvinfer1::DataType* input_types,
                                                int nb_inputs) const TRTNOEXCEPT {
  ASSERT(input_types && nb_inputs > 0);
  return input_types[0];
}

int GluPlugin::initialize() TRTNOEXCEPT { return 0; }

void GluPlugin::terminate() TRTNOEXCEPT {}

void GluPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nb_inputs,
                                const nvinfer1::DynamicPluginTensorDesc* out, int nb_outputs) TRTNOEXCEPT {}

int GluPlugin::enqueue(const nvinfer1::PluginTensorDesc* input_desc, const nvinfer1::PluginTensorDesc* output_desc,
                       const void* const* inputs, void* const* outputs, void* workspace,
                       cudaStream_t stream) TRTNOEXCEPT {
  int nb_dims = input_desc[0].dims.nbDims;

  if (axis_dim_ < 0) axis_dim_ = nb_dims - 1;
  if (axis_dim_ == 0 || axis_dim_ == (nb_dims - 1)) {
    LOG(ERROR) << "GluPlugin axis_dim_ == 0! axis_dim_ not support = 0 now! " << endl;
    assert(0);
  }

  if (nb_dims < 3) {
    LOG(ERROR) << "nbDims < 3 not support! " << endl;
    assert(0);
  }

  auto d = input_desc[0].dims.d;

  int batch = accumulate(d, d + axis_dim_, 1, std::multiplies<int>());
  int seq_len = d[axis_dim_];
  int dim = accumulate(d + axis_dim_ + 1, d + nb_dims, 1, std::multiplies<int>());

  if (seq_len % 2 != 0) {
    LOG(ERROR) << "seq_len % 2 != 0 " << endl;
    return -1;
  }

  int split_dim_size = seq_len / 2;

  if (data_type_ == DataType::kFLOAT) {
    const auto input = static_cast<const float*>(inputs[0]);
    auto output = static_cast<float*>(outputs[0]);

    auto ret = ComputeGlu(batch, split_dim_size, dim, input, output, stream);
    return ret;
  } else if (data_type_ == DataType::kHALF) {
    const auto input = static_cast<const half*>(inputs[0]);
    auto output = static_cast<half*>(outputs[0]);

    auto ret = ComputeGlu(batch, split_dim_size, dim, input, output, stream);
    return ret;
  }

  return 0;
}

nvinfer1::DimsExprs GluPlugin::getOutputDimensions(int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
                                                   nvinfer1::IExprBuilder& expr_builder) TRTNOEXCEPT {
  auto output = inputs[0];
  output.d[axis_dim_] =
      expr_builder.operation(DimensionOperation::kCEIL_DIV, *inputs[0].d[axis_dim_], *expr_builder.constant(2));
  return output;
}

nvinfer1::IPluginV2DynamicExt* GluPlugin::clone() const TRTNOEXCEPT {
  return new GluPlugin(layer_name_, data_type_, axis_dim_);
}

void GluPlugin::destroy() TRTNOEXCEPT { delete this; }

const char* GluPlugin::getPluginVersion() const TRTNOEXCEPT { return GLU_PLUGIN_VERSION; }

const char* GluPlugin::getPluginType() const TRTNOEXCEPT { return GLU_PLUGIN_NAME; }

size_t GluPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nb_inputs*/,
                                   const nvinfer1::PluginTensorDesc* /*outputs*/,
                                   int /*nb_outputs*/) const TRTNOEXCEPT {
  return 0;
}

const char* GluPlugin::getPluginNamespace() const TRTNOEXCEPT { return ""; }

int GluPlugin::getNbOutputs() const TRTNOEXCEPT { return 1; }

void GluPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                                nvinfer1::IGpuAllocator* allocator) TRTNOEXCEPT {}

const char* GluPluginCreator::getPluginName() const TRTNOEXCEPT { return GLU_PLUGIN_NAME; }

const char* GluPluginCreator::getPluginVersion() const TRTNOEXCEPT { return GLU_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* GluPluginCreator::getFieldNames() TRTNOEXCEPT {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

nvinfer1::IPluginV2DynamicExt* GluPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* field_collection) TRTNOEXCEPT {
  assert(field_collection->nbFields >= 1);

  LOG(INFO) << "Creating GluPlugin...\n";

  int data_type_id = 0, axis_dim = -1;

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

    if (field_name.compare("axis_dim") == 0) {
      axis_dim = static_cast<const int*>(field_collection->fields[i].data)[0];
      LOG(INFO) << "Building axis_dim: " << axis_dim << std::endl;
    }
  }

  DataType type = static_cast<DataType>(data_type_id);
  return new GluPlugin(string(name), type, axis_dim);
}

nvinfer1::IPluginV2DynamicExt* GluPluginCreator::deserializePlugin(const char* name, const void* serial_data,
                                                                   size_t serial_length) TRTNOEXCEPT {
  return new GluPlugin(serial_data, serial_length);
}

}  // namespace plugin
}  // namespace nvinfer1
