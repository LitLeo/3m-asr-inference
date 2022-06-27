#include "cmvn_plugin.h"

#include <cassert>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>

#include "common.cuh"
#include "common.h"

namespace nvinfer1 {
namespace plugin {

//output[idx] = (in - mean[threadIdx.x]) * var[threadIdx.x];
template <typename T>
__global__ void CmvnKernel(const T *input, const int *mask, const T *mean, const T *var, 
                                int seq_len, int feat_dim, int weight_dim,
                                T *output) {
  const int idx = threadIdx.x;
  const int seq_idx = blockIdx.x;
  const int batch_idx = blockIdx.z;
  int compute_seq_len = mask[batch_idx];

  if (seq_idx < compute_seq_len) {
      int offset = (batch_idx * seq_len + seq_idx) * feat_dim;
      const T* input_ptr = input + offset;
      T* output_ptr = output + offset;
      for (int i = idx; i < feat_dim; i += blockDim.x) {
        output_ptr[i] = (input_ptr[i] - mean[i%weight_dim]) * var[i%weight_dim];
      }
  }

}

inline int ComputeCmvn(cudaStream_t stream, int batch_size, int seq_len, 
                       int feat_dim, int weight_dim,
                       const float *input, const int *mask, 
                       const float *mean, const float *var, 
                       float *output) {
  dim3 block_3d(64, 1, 1);
  dim3 grid_3d(seq_len, 1, batch_size);
  CmvnKernel<float><<<grid_3d, block_3d, 0, stream>>>(input, mask, mean, var, 
                                                           seq_len, feat_dim, weight_dim,
                                                           output);
    
  auto ret = cudaPeekAtLastError();
  if (ret != cudaSuccess) {
    gLogError << "CmvnKernel failed! status = " << cudaGetErrorString(ret) << endl;
    return -1;
  }
  return 0;
}

// Static class fields initialization
PluginFieldCollection CmvnPluginCreator::mFC{};
std::vector<PluginField> CmvnPluginCreator::mPluginAttributes;

/*REGISTER_TENSORRT_PLUGIN(CmvnPluginCreator);*/

CmvnPlugin::CmvnPlugin(std::string name, const DataType type, 
                       const nvinfer1::Weights mean, const nvinfer1::Weights var)
    : layer_name_(name), data_type_(type), dim_(mean.count),
      mean_(mean), var_(var) {
  mean_dev_ptr_ = nullptr;
  var_dev_ptr_ = nullptr;

  if (mean.count != var.count)
    gLogError << "mean.count != var.count" << endl;
}

CmvnPlugin::CmvnPlugin(const std::string name, const void *data, size_t length)
    : layer_name_(name) {
  gLogVerbose << "Starting to deserialize DEBUG plugin: " << layer_name_ << std::endl;
  /*char layer_name[100]; */
  /*deserialize_value(&data, &length, layer_name);*/
  /*layer_name_ = std::string(layer_name);*/
  deserialize_value(&data, &length, &data_type_);
  deserialize_value(&data, &length, &dim_);

  const char *d = static_cast<const char*>(data);
  gLogVerbose << "Deserializing mean and var" << std::endl;
  const size_t word_size = getElementSize(data_type_);
  mean_dev_ptr_ = deserToDev<char>(d, dim_ * word_size);
  var_dev_ptr_ = deserToDev<char>(d, dim_ * word_size);

  mean_ = nvinfer1::Weights{DataType::kFLOAT, nullptr, (int64_t)dim_};
  var_ = nvinfer1::Weights{DataType::kFLOAT, nullptr, (int64_t)dim_};
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt *CmvnPlugin::clone() const {
  return new CmvnPlugin(layer_name_, data_type_, mean_, var_);
}

nvinfer1::DimsExprs CmvnPlugin::getOutputDimensions(
    int outputIndex, 
    const nvinfer1::DimsExprs *inputs, 
    int nbInputs, 
    nvinfer1::IExprBuilder &exprBuilder) {
  return inputs[0];
}

bool CmvnPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, 
                                           int nbInputs, int nbOutputs) {
  const PluginTensorDesc &input = inOut[0];
  if (pos == 0) {
    return (input.type == data_type_) && (input.format == TensorFormat::kLINEAR);
  }
  if (pos == 1)
    return true;

  if (pos == 2) {
    const PluginTensorDesc &output = inOut[2];
    return (input.type == output.type) && (output.format == TensorFormat::kLINEAR);
  }
  return false;
}

void CmvnPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) {
  assert(data_type_ == in[0].desc.type);
}

size_t CmvnPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  return 0;
}

int CmvnPlugin::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, 
    const void *const *inputs, void *const *outputs, 
    void *workspace, cudaStream_t stream) {
  const int inputVolume = volume(inputDesc[0].dims);
  const int batch_size = inputDesc[0].dims.d[0];
  const int seq_len = inputDesc[0].dims.d[1];
  const int feat_dim = inputDesc[0].dims.d[2];

  int status = 0;

  if (data_type_ == DataType::kFLOAT) {
    const float *input = static_cast<const float*>(inputs[0]);
    const int *mask = static_cast<const int*>(inputs[1]);
    float *output = static_cast<float*>(outputs[0]);
    const float *mean_ptr = reinterpret_cast<float*>(mean_dev_ptr_);
    const float *var_ptr = reinterpret_cast<float*>(var_dev_ptr_);

    status = ComputeCmvn(stream, batch_size, seq_len, feat_dim, dim_, 
                         input, mask, mean_ptr, var_ptr, output);

  } else if (data_type_ == DataType::kHALF) {
#ifdef __SCORE_HALF__
    const half *input = static_cast<const half*>(inputs[0]);
#endif
  } else {
    assert(false);
  }

  return status;
}

// IPluginV2Ext Methods
nvinfer1::DataType CmvnPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, 
                                                 int nbInputs) const {
  assert(index == 0);
  assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
  return inputTypes[0];
}

// IPluginV2 Methods
const char *CmvnPlugin::getPluginType() const {
  return CMVN_PLUGIN_NAME;
}

const char *CmvnPlugin::getPluginVersion() const {
  return CMVN_PLUGIN_VERSION;
}

int CmvnPlugin::getNbOutputs() const {
  return 1;
}

int CmvnPlugin::initialize() {
  gLogVerbose << "CMVN init start" << std::endl;
  if (mean_.values && var_.values) {
    // target size
    const size_t word_size = getElementSize(data_type_);
    const size_t nb_bytes = mean_.count * word_size;
    CUDA_CHECK(cudaMalloc(&mean_dev_ptr_, nb_bytes));
    CUDA_CHECK(cudaMalloc(&var_dev_ptr_, nb_bytes));

    if (data_type_ == DataType::kFLOAT) {
      convertAndCopyToDevice(mean_, reinterpret_cast<float*>(mean_dev_ptr_));
      convertAndCopyToDevice(var_, reinterpret_cast<float*>(var_dev_ptr_));
    } else {
      convertAndCopyToDevice(mean_, reinterpret_cast<half*>(mean_dev_ptr_));
      convertAndCopyToDevice(var_, reinterpret_cast<half*>(var_dev_ptr_));
    }
  }
  gLogVerbose << "CMVN init done" << std::endl;

  return 0;
}

void CmvnPlugin::terminate()
{
  if (mean_dev_ptr_ && var_dev_ptr_) {
    CUDA_CHECK(cudaFree(mean_dev_ptr_));
    CUDA_CHECK(cudaFree(var_dev_ptr_));

    mean_dev_ptr_ = nullptr;
    var_dev_ptr_ = nullptr;
  }
}

size_t CmvnPlugin::getSerializationSize() const {
  return sizeof(data_type_) + sizeof(dim_) + getElementSize(data_type_) * dim_ * 2;
}

void CmvnPlugin::serialize(void *buffer) const {
  /*serialize_value(&buffer, layer_name_.c_str());*/
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, dim_);

  char *d = static_cast<char*>(buffer);
  const size_t word_size = getElementSize(data_type_);
  serFromDev(d, mean_dev_ptr_, dim_ * word_size);
  serFromDev(d, var_dev_ptr_, dim_ * word_size);
}

void CmvnPlugin::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void CmvnPlugin::setPluginNamespace(const char *libNamespace) {
  namespace_ = libNamespace;
}

const char *CmvnPlugin::getPluginNamespace() const {
  return namespace_.c_str();
}

///////////////

CmvnPluginCreator::CmvnPluginCreator() {
  // Fill PluginFieldCollection with PluginField arguments metadata
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *CmvnPluginCreator::getPluginName() const {
  return CMVN_PLUGIN_NAME;
}

const char *CmvnPluginCreator::getPluginVersion() const {
  return CMVN_PLUGIN_VERSION;
}

const PluginFieldCollection *CmvnPluginCreator::getFieldNames() {
  return &mFC;
}

IPluginV2 *CmvnPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) {
  Weights mean{DataType::kFLOAT, nullptr, 0};
  Weights var{DataType::kFLOAT, nullptr, 0};
  int typeId = -1;
  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);
    if (field_name.compare("type_id") == 0) {
      typeId = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building typeId: " << typeId << std::endl;
    }

    if (field_name.compare("mean") == 0) {
      gLogVerbose << "Building mean...\n";
      mean.values = fc->fields[i].data;
      mean.count = fc->fields[i].length;
      mean.type = fieldTypeToDataType(fc->fields[i].type);
    }

    if (field_name.compare("var") == 0) {
      gLogVerbose << "Building var...\n";
      var.values = fc->fields[i].data;
      var.count = fc->fields[i].length;
      var.type = fieldTypeToDataType(fc->fields[i].type);
    }
  }

  if (typeId < 0 || typeId > 3) {
    gLogError << "CMVN: invalid typeId " << typeId << std::endl;
    return nullptr;
  }
  DataType type = static_cast<DataType>(typeId);
  gLogVerbose << "Creating CmvnPlugin...\n";
  return new CmvnPlugin(name, type, mean, var);
}

IPluginV2 *CmvnPluginCreator::deserializePlugin(const char *name, const void *serialData, 
                                                size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call CmvnPlugin::destroy()
  return new CmvnPlugin(name, serialData, serialLength);
}

void CmvnPluginCreator::setPluginNamespace(const char *libNamespace) {
  namespace_ = libNamespace;
}

const char *CmvnPluginCreator::getPluginNamespace() const {
  return namespace_.c_str();
}

}  // BEGIN_PLUGIN_NAMESPACE
}  // BEGIN_LIB_NAMESPACE

