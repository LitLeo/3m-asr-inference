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

#include "fmoe_expert_plugin.h"

#include <cassert>
#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "common.h"
#include "cublas_common.h"
#include "serialize.hpp"
#include "fmoe_expert_kernel.h"

#include "debug.h"

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
PluginFieldCollection FMoEExpertPluginCreator::FC_{};
std::vector<PluginField> FMoEExpertPluginCreator::plugin_attributes_;

REGISTER_TENSORRT_PLUGIN(FMoEExpertPluginCreator);

template<class T>
int compute_fmoe_expert(const T* input, const int* gate_idx, const int input_volume, const int S,
                        const int num_expert, const int idim, const int hidden_units,
                        const T* w1_weight_ptr, const T* w1_bias_ptr,
                        const T* w2_weight_ptr, const T* w2_bias_ptr,
                        std::vector<int>& v_acc_his, void* workspace, T* output,
                        cudaStream_t stream, std::shared_ptr<CudaStreamManager> csm_ptr) {

  auto mapping_size = alignTo<int>(S, kAlignment);
  auto his_size = alignTo<int>(num_expert+1, kAlignment);
  auto input_buffer_size = alignTo<int>(input_volume, kAlignment);

  int* mapping = static_cast<int*>(workspace);
  int* acc_histogram = mapping + mapping_size;

  int status = -1;
  status = ComputeScatterMapping(gate_idx, num_expert, S, mapping, acc_histogram, stream);
  if (status != 0) {
    gLogError << "compute_scatter_mapping error!" << endl;
    return status;
  }

  //print_data(gate_idx, S, "gate_idx");
  //print_data(mapping, S, "mapping");
  //print_data(acc_histogram, num_expert+1, "acc_histogram");
  //cout << "====================" << endl;

  //const size_t word_size = getElementSize(data_type_);
  const size_t word_size = sizeof(T);

  // get buffer from workspace
  float* input_buffer = reinterpret_cast<float*>(acc_histogram + his_size);
  float* hidden_buffer = input_buffer + input_buffer_size;

  status = ComputeScatterMappingCopy(input, mapping, S, idim, input_buffer, stream);
  if (status != 0) {
    gLogError << "ComputeScatterMappingCopy error!" << endl;
    return status;
  }
  //print_data(input_buffer, 10, "reorder_input0");
  //print_data(input_buffer + idim_, 10, "reorder_input1");

  int* h_acc_his = v_acc_his.data();
  cudaMemcpyAsync(h_acc_his, acc_histogram, sizeof(int) * (num_expert+1), cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_T;
  for (int i = 0; i < num_expert; i++) {
    auto cur_stream = csm_ptr->Stream(i);
    auto handle = csm_ptr->CublasHandle(i);
    int m = h_acc_his[i + 1] - h_acc_his[i];
    if (m == 0)
      continue;

    float* input_buffer_ptr = input_buffer + h_acc_his[i] * idim;
    float* hidden_buffer_ptr = hidden_buffer + h_acc_his[i] * hidden_units;

    // Weights offset
    auto w_offset = i * idim * hidden_units;
    auto cur_w1_weight_ptr = w1_weight_ptr + w_offset;
    auto cur_w1_bias_ptr = w1_bias_ptr + i * hidden_units;
    auto cur_w2_weight_ptr = w2_weight_ptr + w_offset;
    auto cur_w2_bias_ptr = w2_bias_ptr + i * idim;

    //print_data(input_buffer_ptr, 10, "input_buffer_ptr");
    // w1 gemm, tmp => output
    CUBLAS_CHECK(cublasGemm(handle, transa, transb,
                            m, hidden_units, idim,
                            1.0f, input_buffer_ptr, cur_w1_weight_ptr,
                            0.0f, hidden_buffer_ptr));

    //print_data(hidden_buffer_ptr, 10, "w1_weight");

    // w1 bias + activate, tmp2
    status = ComputeBiasSilu(hidden_buffer_ptr, cur_w1_bias_ptr, m*hidden_units,
                             hidden_units, hidden_buffer_ptr, cur_stream);
    if (status != 0) {
      gLogError << "ComputeBiasSilu error!" << endl;
      return status;
    }

    //print_data(hidden_buffer_ptr, 10, "silu");

    // w2 gemm tmp2 => tmp1
    CUBLAS_CHECK(cublasGemm(handle, transa, transb,
                            m, idim, hidden_units,
                            1.0f, hidden_buffer_ptr, cur_w2_weight_ptr,
                            0.0f, input_buffer_ptr));

    // w2 bias tmp1
    status = ComputeBias(input_buffer_ptr, cur_w2_bias_ptr, m*idim, idim, input_buffer_ptr, cur_stream);
    if (status != 0) {
      gLogError << "ComputeBias error!" << endl;
      return status;
    }

    //print_data(input_buffer_ptr, 10, "w2");
    //cout << "=================" << endl;
  }

  csm_ptr->SyncAllStream();

  status = ComputeGatherrMappingCopy(input_buffer, mapping, S, idim, output, stream);
  if (status != 0) {
    gLogError << "ComputeGatherrMappingCopy error!" << endl;
    return status;
  }

  //print_data(output, 10, "output");
  //cout << "=================" << endl;

  return status;

}

FMoEExpertPlugin::FMoEExpertPlugin(const std::string& name, const nvinfer1::DataType type, const int num_expert,
                                   const int idim, const int hidden_units, const int act_type)
    : layer_name_(name), data_type_(type), num_expert_(num_expert),
      idim_(idim), hidden_units_(hidden_units), act_type_(act_type) {
  v_acc_his_.resize(num_expert_ + 1);
  cuda_stream_manager_.reset(new CudaStreamManager());
  cuda_stream_manager_->Init();
}

FMoEExpertPlugin::FMoEExpertPlugin(const std::string& name, const nvinfer1::DataType type, const int num_expert,
                                   const int idim, const int hidden_units, const int act_type,
                                   std::shared_ptr<CudaStreamManager> cuda_stream_manager)
    : layer_name_(name), data_type_(type), num_expert_(num_expert),
      idim_(idim), hidden_units_(hidden_units), act_type_(act_type),
      cuda_stream_manager_(cuda_stream_manager) {
  v_acc_his_.resize(num_expert_ + 1);
}

FMoEExpertPlugin::FMoEExpertPlugin(const std::string& name, const void* data, size_t length) : layer_name_(name) {

  // Deserialize in the same order as serialization
  deserialize_value(&data, &length, &data_type_);
  deserialize_value(&data, &length, &num_expert_);
  deserialize_value(&data, &length, &idim_);
  deserialize_value(&data, &length, &hidden_units_);
  deserialize_value(&data, &length, &act_type_);

  int tmp = 0;
  deserialize_value(&data, &length, &tmp);
  deserialize_value(&data, &length, &tmp);
  deserialize_value(&data, &length, &tmp);

  v_acc_his_.resize(num_expert_ + 1);
  cuda_stream_manager_.reset(new CudaStreamManager());
  cuda_stream_manager_->Init();
}

// IPluginV2DynamicExt Methods
IPluginV2DynamicExt* FMoEExpertPlugin::clone() const TRTNOEXCEPT {
  auto ret = new FMoEExpertPlugin(layer_name_, data_type_, num_expert_, idim_,
                                  hidden_units_, act_type_, cuda_stream_manager_);
  return ret;
}

DimsExprs FMoEExpertPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs,
                                          IExprBuilder& exprBuilder) TRTNOEXCEPT {
  assert(nbInputs == 6);
  return inputs[0];
}

bool FMoEExpertPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) TRTNOEXCEPT {
  assert(nbInputs == 6);
  assert(nbOutputs == 1);

  // input, gate_idx, w1_weight, w1_bias, w2_weight, w2_bias
  const PluginTensorDesc& in_out = inOut[pos];
  if (pos == 1)
    return (in_out.type == DataType::kINT32) && (in_out.format == TensorFormat::kLINEAR);
  else
    return (in_out.type == data_type_) && (in_out.format == TensorFormat::kLINEAR);

  return false;
}

void FMoEExpertPlugin::configurePlugin(const DynamicPluginTensorDesc* inputs, int nbInputs,
                                 const DynamicPluginTensorDesc* outputs, int nbOutputs) TRTNOEXCEPT {
  // Validate input arguments
  assert(nbInputs == 6);
  assert(nbOutputs == 1);
  assert(data_type_ == inputs[0].desc.type);
}

size_t FMoEExpertPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
                                    int nbOutputs) const TRTNOEXCEPT {
  const size_t word_size = getElementSize(data_type_);
  const int input_volume = volume(inputs[0].dims);
  const int S = input_volume / idim_;

  auto mapping_size = alignTo<int>(S, kAlignment);
  auto his_size = alignTo<int>(num_expert_+1, kAlignment);
  auto input_buffer_size = alignTo<int>(input_volume, kAlignment);
  auto hidden_buffer_size = alignTo<int>(S * hidden_units_, kAlignment);

  return mapping_size * sizeof(int)         // mapping_size
         + his_size * sizeof(int)           // acc_histogram size
         + input_buffer_size * word_size    // hidden_buffer_size
         + hidden_buffer_size * word_size;  // hidden_buffer_size
}

int FMoEExpertPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT {
  const int input_volume = volume(inputDesc[0].dims);
  const int S = input_volume / idim_;

  int status = -1;

  if (data_type_ == DataType::kFLOAT) {
    // Our plugin outputs only one tensor
    const float* input = static_cast<const float*>(inputs[0]);
    const int* gate_idx = static_cast<const int*>(inputs[1]);
    const float* w1_weight_ptr = static_cast<const float*>(inputs[2]);
    const float* w1_bias_ptr = static_cast<const float*>(inputs[3]);

    const float* w2_weight_ptr = static_cast<const float*>(inputs[4]);
    const float* w2_bias_ptr = static_cast<const float*>(inputs[5]);

    float* output = static_cast<float*>(outputs[0]);

    status = compute_fmoe_expert(input, gate_idx, input_volume, S,
                                 num_expert_, idim_, hidden_units_,
                                 w1_weight_ptr, w1_bias_ptr, w2_weight_ptr, w2_bias_ptr,
                                 v_acc_his_, workspace, output,
                                 stream, cuda_stream_manager_);
  } else if (data_type_ == DataType::kHALF) {
    assert(0);
  }


  return status;
}

// IPluginV2Ext Methods
DataType FMoEExpertPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const TRTNOEXCEPT {
  assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
  return inputTypes[0];
}

// IPluginV2 Methods
const char* FMoEExpertPlugin::getPluginType() const TRTNOEXCEPT { return FMOE_EXPERT_NAME; }

const char* FMoEExpertPlugin::getPluginVersion() const TRTNOEXCEPT { return FMOE_EXPERT_VERSION; }

int FMoEExpertPlugin::getNbOutputs() const TRTNOEXCEPT { return 1; }

int FMoEExpertPlugin::initialize() TRTNOEXCEPT { return 0; }

void FMoEExpertPlugin::terminate() TRTNOEXCEPT {}

size_t FMoEExpertPlugin::getSerializationSize() const TRTNOEXCEPT {
  return sizeof(data_type_) + sizeof(num_expert_) + sizeof(idim_)
         + sizeof(hidden_units_) + sizeof(act_type_) + 3*sizeof(int);
}

void FMoEExpertPlugin::serialize(void* buffer) const TRTNOEXCEPT {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, num_expert_);
  serialize_value(&buffer, idim_);
  serialize_value(&buffer, hidden_units_);
  serialize_value(&buffer, act_type_);

  int tmp = 0;
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);
}

void FMoEExpertPlugin::destroy() TRTNOEXCEPT {
  delete this;
}

void FMoEExpertPlugin::setPluginNamespace(const char* libNamespace) TRTNOEXCEPT { namespace_ = libNamespace; }

const char* FMoEExpertPlugin::getPluginNamespace() const TRTNOEXCEPT { return namespace_.c_str(); }

///////////////////////

FMoEExpertPluginCreator::FMoEExpertPluginCreator() {
  FC_.nbFields = plugin_attributes_.size();
  FC_.fields = plugin_attributes_.data();
}

const char* FMoEExpertPluginCreator::getPluginName() const TRTNOEXCEPT { return FMOE_EXPERT_NAME; }

const char* FMoEExpertPluginCreator::getPluginVersion() const TRTNOEXCEPT { return FMOE_EXPERT_VERSION; }

const PluginFieldCollection* FMoEExpertPluginCreator::getFieldNames() TRTNOEXCEPT { return &FC_; }

IPluginV2* FMoEExpertPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) TRTNOEXCEPT {
  gLogVerbose << "Creating FMoEExpertPlugin...\n";

  int type_id = -1;
  int num_expert = 0, idim = 0, hidden_units = 0, act_type = 0;

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("data_type") == 0) {
      type_id = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building type_id: " << type_id << std::endl;
    }
    if (field_name.compare("num_expert") == 0) {
      num_expert = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building num_expert: " << num_expert << std::endl;
    }
    if (field_name.compare("idim") == 0) {
      idim = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building idim " << idim << std::endl;
    }
    if (field_name.compare("hidden_units") == 0) {
      hidden_units = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building hidden_units " << hidden_units << std::endl;
    }
    if (field_name.compare("act_type") == 0) {
      act_type = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building act_type " << act_type << std::endl;
    }
  }

  if (type_id < 0 || type_id > 1) {
    gLogError << "fmoe: invalid type_id " << type_id << std::endl;
    return nullptr;
  }

  DataType type = static_cast<DataType>(type_id);

  gLogVerbose << "Building the Plugin...\n";
  FMoEExpertPlugin* p = new FMoEExpertPlugin(name, type, num_expert, idim, hidden_units, act_type);
  return p;
}

IPluginV2* FMoEExpertPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRTNOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FMoEExpertPlugin::destroy()
  return new FMoEExpertPlugin(name, serialData, serialLength);
}

void FMoEExpertPluginCreator::setPluginNamespace(const char* libNamespace) TRTNOEXCEPT { namespace_ = libNamespace; }

const char* FMoEExpertPluginCreator::getPluginNamespace() const TRTNOEXCEPT { return namespace_.c_str(); }

}  // namespace plugin
}  // namespace nvinfer1
