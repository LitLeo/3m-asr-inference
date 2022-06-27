#include "prior_prob_plugin.h"
#include "prior_prob_kernel.h"

#include <cassert>
#include <cstring>
#include <vector>
/*#include <stdio.h>*/

using namespace std;
using namespace nvinfer1;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
PluginFieldCollection PriorProbPluginCreator::mFC{};
std::vector<PluginField> PriorProbPluginCreator::mPluginAttributes;

/*REGISTER_TENSORRT_PLUGIN(PriorProbPluginCreator);*/

PriorProbPlugin::PriorProbPlugin(std::string name, const DataType type, const size_t dim, 
                                 const nvinfer1::Weights prob)
    : layer_name_(name), data_type_(type), dim_(dim), prob_(prob) {
  prob_ptr_= nullptr;
}

PriorProbPlugin::PriorProbPlugin(const std::string name, const void *data, size_t length)
    : layer_name_(name) {
  gLogVerbose << "Starting to deserialize PriorProbDynamic plugin: " << layer_name_ << std::endl;
  deserialize_value(&data, &length, &data_type_);
  deserialize_value(&data, &length, &dim_);

  const char *d = static_cast<const char*>(data);
  gLogVerbose << "Deserializing prob" << std::endl;
  const size_t word_size = common::GetElementSize(data_type_);
  prob_ptr_ = deserToDev<char>(d, dim_ * word_size);

  prob_ = nvinfer1::Weights{DataType::kFLOAT, nullptr, (int64_t)dim_};
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt *PriorProbPlugin::clone() const {
  return new PriorProbPlugin(layer_name_, data_type_, dim_, prob_);
}

nvinfer1::DimsExprs PriorProbPlugin::getOutputDimensions(
    int outputIndex, 
    const nvinfer1::DimsExprs *inputs, 
    int nbInputs, 
    nvinfer1::IExprBuilder &exprBuilder) {

  DimsExprs ret;

  ret.nbDims = 3;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = inputs[0].d[2];

  return ret;
}

bool PriorProbPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, 
                                                       int nbInputs, int nbOutputs) {
  const PluginTensorDesc &input = inOut[0];
  if (pos == 0)
    return (input.type == data_type_) && (input.format == TensorFormat::kLINEAR);

  if (pos == 1) {
    const PluginTensorDesc &output = inOut[1];
    return (input.type == output.type) && (output.format == TensorFormat::kLINEAR);
  }
  return false;
}

void PriorProbPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) {
  assert(data_type_ == in[0].desc.type);
}

size_t PriorProbPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  return 0;
}

int PriorProbPlugin::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, 
    const void *const *inputs, void *const *outputs, 
    void *workspace, cudaStream_t stream) {
  const int inputVolume = common::Volume(inputDesc[0].dims);

  int status = 0;

  if (data_type_ == DataType::kFLOAT) {
    const float *input = static_cast<const float*>(inputs[0]);
    float *output = static_cast<float*>(outputs[0]);
    const float *prob_ptr = reinterpret_cast<float*>(prob_ptr_);

    int n = inputVolume/dim_;
    status = ComputePriorProb(stream, n, dim_, input, prob_ptr, output);
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
nvinfer1::DataType PriorProbPlugin::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  assert(index == 0);
  assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
  return inputTypes[0];
}

// IPluginV2 Methods
const char *PriorProbPlugin::getPluginType() const {
  return PRIOR_PROB_PLUGIN_NAME;
}

const char *PriorProbPlugin::getPluginVersion() const {
  return PRIOR_PROB_PLUGIN_VERSION;
}

int PriorProbPlugin::getNbOutputs() const {
  return 1;
}

int PriorProbPlugin::initialize() {
  gLogVerbose << "PriorProb init start" << std::endl;
  if (prob_.values) {
    // target size
    const size_t word_size = common::GetElementSize(data_type_);
    const size_t nb_bytes = prob_.count * word_size;
    CUDA_CHECK(cudaMalloc(&prob_ptr_, nb_bytes));

    if (data_type_ == DataType::kFLOAT) {
      convertAndCopyToDevice(prob_, reinterpret_cast<float*>(prob_ptr_));
    } else {
      convertAndCopyToDevice(prob_, reinterpret_cast<half*>(prob_ptr_));
    }
  }
  gLogVerbose << "PriorProb init done" << std::endl;

  return 0;
}

void PriorProbPlugin::terminate()
{
  if (prob_ptr_) {
    CUDA_CHECK(cudaFree(prob_ptr_));
    prob_ptr_ = nullptr;
  }
}

size_t PriorProbPlugin::getSerializationSize() const {
  return sizeof(data_type_) + sizeof(dim_) + 
         common::GetElementSize(data_type_) * dim_;
}

void PriorProbPlugin::serialize(void *buffer) const {
  /*serialize_value(&buffer, layer_name_.c_str());*/
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, dim_);

  char *d = static_cast<char*>(buffer);
  const size_t word_size = common::GetElementSize(data_type_);
  serFromDev(d, prob_ptr_, dim_ * word_size);
}

void PriorProbPlugin::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void PriorProbPlugin::setPluginNamespace(const char *libNamespace) {
  namespace_ = libNamespace;
}

const char *PriorProbPlugin::getPluginNamespace() const {
  return namespace_.c_str();
}

///////////////

PriorProbPluginCreator::PriorProbPluginCreator() {
  // Fill PluginFieldCollection with PluginField arguments metadata
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *PriorProbPluginCreator::getPluginName() const {
  return PRIOR_PROB_PLUGIN_NAME;
}

const char *PriorProbPluginCreator::getPluginVersion() const {
  return PRIOR_PROB_PLUGIN_VERSION;
}

const PluginFieldCollection *PriorProbPluginCreator::getFieldNames() {
  return &mFC;
}

IPluginV2 *PriorProbPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) {
  Weights prob{DataType::kFLOAT, nullptr, 0};
  int typeId = -1;
  size_t dim = 0;
  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);
    if (field_name.compare("type_id") == 0) {
      typeId = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building typeId: " << typeId << std::endl;
    }

    if (field_name.compare("dim") == 0) {
      dim = *static_cast<const size_t*>(fc->fields[i].data);
      gLogVerbose << "Building dim: " << dim << std::endl;
    }

    if (field_name.compare("prob") == 0) {
      gLogVerbose << "Building prob...\n";
      prob.values = fc->fields[i].data;
      prob.count = fc->fields[i].length;
      prob.type = fieldTypeToDataType(fc->fields[i].type);
    }
  }

  if (typeId < 0 || typeId > 3) {
    gLogError << "PriorProb: invalid typeId " << typeId << std::endl;
    return nullptr;
  }
  DataType type = static_cast<DataType>(typeId);
  gLogVerbose << "Creating PriorProbPlugin...\n";
  return new PriorProbPlugin(name, type, dim, prob);
}

IPluginV2 *PriorProbPluginCreator::deserializePlugin(const char *name, const void *serialData, 
                                                            size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call PriorProbPlugin::destroy()
  return new PriorProbPlugin(name, serialData, serialLength);
}

void PriorProbPluginCreator::setPluginNamespace(const char *libNamespace) {
  namespace_ = libNamespace;
}

const char *PriorProbPluginCreator::getPluginNamespace() const {
  return namespace_.c_str();
}

} // namespace plugin
} // namespace nvinfer1

