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

#include "dump_tensor_plugin.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <vector>

#include "common.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
PluginFieldCollection DumpTensorPluginCreator::mFC{};
std::vector<PluginField> DumpTensorPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(DumpTensorPluginCreator);

DumpTensorPlugin::DumpTensorPlugin(const std::string name) : layer_name_(name) {}

DumpTensorPlugin::DumpTensorPlugin(const std::string name, const void* data, size_t length) : layer_name_(name) {
  int tmp = 0;
  deserialize_value(&data, &length, &tmp);
  deserialize_value(&data, &length, &tmp);
  deserialize_value(&data, &length, &tmp);

  vector<char> layer_name(kMaxLayerNameSize);
  deserialize_vector(&data, &length, &layer_name);
  layer_name_ = std::string(layer_name.data());

  gLogVerbose << "Starting to deserialize DEBUG plugin: " << layer_name_ << std::endl;
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* DumpTensorPlugin::clone() const TRTNOEXCEPT { return new DumpTensorPlugin(layer_name_); }

nvinfer1::DimsExprs DumpTensorPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                          int nbInputs, nvinfer1::IExprBuilder& exprBuilder) TRTNOEXCEPT {
  return inputs[outputIndex];
}

bool DumpTensorPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                                 int nbOutputs) TRTNOEXCEPT {
  // only support float and int32
  if ((inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kINT32) &&
      inOut[pos].format == TensorFormat::kLINEAR) {
    return true;
  } else {
    return false;
  }
}

void DumpTensorPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) TRTNOEXCEPT {
  data_type_ = in[0].desc.type;
}

size_t DumpTensorPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                          const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const TRTNOEXCEPT {
  return 0;
}

int DumpTensorPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                              const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT {
  for (size_t n = 0; n < 1; n++) {
    const int input_volume = volume(inputDesc[n].dims);
    // remove dim = 1 or 0
    vector<int> v_dims;
    auto in_dims = inputDesc[n].dims;

    for (int i = 0; i < in_dims.nbDims; i++) {
      int d = in_dims.d[i];
      v_dims.push_back(d);
    }

    while (input_volume > 1 && v_dims.back() == 1) {
      v_dims.pop_back();
    }

    // if (in_dims.nbDims > 1) {
    //// batch dim
    // v_dims.push_back(in_dims.d[0]);
    // for (int i = 1; i < in_dims.nbDims; i++) {
    // int d = in_dims.d[i];
    // if (d > 1) v_dims.push_back(d);
    //}
    //}

    const float* input = static_cast<const float*>(inputs[n]);
    float* output = static_cast<float*>(outputs[n]);

    float* arr = new float[input_volume];
    memset(arr, 0, input_volume * sizeof(float));

    cudaMemcpyAsync(arr, input, input_volume * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpyAsync(output, input, input_volume * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    if (data_type_ == DataType::kFLOAT) {
      // int p_size = 100;
      // if (v_dims[v_dims.size()-1] < p_size)
      // p_size = v_dims[v_dims.size()-1];
      /*checkDeviceData(p_size, input, layer_name_.c_str());*/

      p_sum(arr, v_dims, layer_name_);
      p(arr, v_dims);
    } else if (data_type_ == DataType::kINT32) {
      // int p_size = 100;
      // if (v_dims[v_dims.size()-1] < p_size)
      // p_size = v_dims[v_dims.size()-1];
      /*checkDeviceData(p_size, input, layer_name_.c_str());*/

      auto int_arr = reinterpret_cast<int*>(arr);
      p_sum(int_arr, v_dims, layer_name_);
      p(int_arr, v_dims);
    } else {
      delete[] arr;
      assert(false);
    }

    delete[] arr;
  }

  return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType DumpTensorPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                       int nbInputs) const TRTNOEXCEPT {
  // only support float and int32
  assert(inputTypes[index] == DataType::kFLOAT || inputTypes[index] == DataType::kINT32);
  return inputTypes[index];
}

// IPluginV2 Methods

const char* DumpTensorPlugin::getPluginType() const TRTNOEXCEPT { return DUMP_TENSOR_PLUGIN_NAME; }

const char* DumpTensorPlugin::getPluginVersion() const TRTNOEXCEPT { return DUMP_TENSOR_PLUGIN_VERSION; }

int DumpTensorPlugin::getNbOutputs() const TRTNOEXCEPT { return 1; }

int DumpTensorPlugin::initialize() TRTNOEXCEPT { return 0; }

void DumpTensorPlugin::terminate() TRTNOEXCEPT {}

size_t DumpTensorPlugin::getSerializationSize() const TRTNOEXCEPT {
  return sizeof(int) * 3 + sizeof(char) * kMaxLayerNameSize + sizeof(size_t);  // layer_name vector
}

void DumpTensorPlugin::serialize(void* buffer) const TRTNOEXCEPT {
  int tmp = 0;
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);

  vector<char> layer_name(kMaxLayerNameSize);
  memcpy(layer_name.data(), layer_name_.data(), layer_name_.size());
  serialize_vector(&buffer, layer_name);
}

void DumpTensorPlugin::destroy() TRTNOEXCEPT {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void DumpTensorPlugin::setPluginNamespace(const char* libNamespace) TRTNOEXCEPT { namespace_ = libNamespace; }

const char* DumpTensorPlugin::getPluginNamespace() const TRTNOEXCEPT { return namespace_.c_str(); }

///////////////

DumpTensorPluginCreator::DumpTensorPluginCreator() {
  // Fill PluginFieldCollection width PluginField arguments metadata
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* DumpTensorPluginCreator::getPluginName() const TRTNOEXCEPT { return DUMP_TENSOR_PLUGIN_NAME; }

const char* DumpTensorPluginCreator::getPluginVersion() const TRTNOEXCEPT { return DUMP_TENSOR_PLUGIN_VERSION; }

const PluginFieldCollection* DumpTensorPluginCreator::getFieldNames() TRTNOEXCEPT { return &mFC; }

IPluginV2* DumpTensorPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRTNOEXCEPT {
  // if (fc->nbFields != 1 && fc->nbFields != 2) {
  // gLogError << "fc->nbFields != 1 && fc->nbFields != 2" << endl;
  // assert(0);
  //}

  return new DumpTensorPlugin(name);
}

IPluginV2* DumpTensorPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRTNOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call DumpTensorPlugin::destroy()
  return new DumpTensorPlugin(name, serialData, serialLength);
}

void DumpTensorPluginCreator::setPluginNamespace(const char* libNamespace) TRTNOEXCEPT { namespace_ = libNamespace; }

const char* DumpTensorPluginCreator::getPluginNamespace() const TRTNOEXCEPT { return namespace_.c_str(); }

}  // namespace plugin
}  // namespace nvinfer1
