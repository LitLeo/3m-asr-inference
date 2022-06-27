#include "left_padding_cache_plugin.h"

#include <cuda_fp16.h>
#include <algorithm>

#include "debug.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
// PluginFieldCollection LeftPaddingCachePluginCreator::mFC{};
// std::vector<PluginField> LeftPaddingCachePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(LeftPaddingCachePluginCreator);

LeftPaddingCachePlugin::LeftPaddingCachePlugin(const std::string name, const nvinfer1::DataType type,
                                               const int axis_dim)
    : layer_name_(name), data_type_(type), axis_dim_(axis_dim) {
  cout << "init axis_dim_:" << axis_dim_ << endl;
}

LeftPaddingCachePlugin::LeftPaddingCachePlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &axis_dim_);

  // for future
  int tmp;
  deserialize_value(&serialData, &serialLength, &tmp);
  deserialize_value(&serialData, &serialLength, &tmp);
  deserialize_value(&serialData, &serialLength, &tmp);
}

size_t LeftPaddingCachePlugin::getSerializationSize() const {
  return sizeof(data_type_) + sizeof(axis_dim_) + sizeof(int) * 3;
}

void LeftPaddingCachePlugin::serialize(void* buffer) const {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, axis_dim_);

  int tmp = 0;
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);
}

bool LeftPaddingCachePlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                                       int nbOutputs) {
  assert(nbInputs == 2);
  assert(nbOutputs == 2);

  // all inputs and inputs's type is data_type
  const PluginTensorDesc& inout = inOut[pos];
  return (inout.type == data_type_) && (inout.format == TensorFormat::kLINEAR);
}

nvinfer1::DataType LeftPaddingCachePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                             int nbInputs) const {
  ASSERT(inputTypes && nbInputs > 0);
  return inputTypes[0];
}

int LeftPaddingCachePlugin::initialize() { return 0; }

void LeftPaddingCachePlugin::terminate() {}

void LeftPaddingCachePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                             const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
  auto inputDesc = in[0].desc;
  auto in_cache = in[1].desc;
  auto output_desc = out[0].desc;
  auto out_cache = out[1].desc;

  cout << "configurePlugin axis_dim_:" << axis_dim_ << endl;
  // check axis_dim_
  auto nb_dims = inputDesc.dims.nbDims;
  if (axis_dim_ < 0) axis_dim_ = nb_dims - 1;
  if (axis_dim_ == 0 || axis_dim_ == (nb_dims - 1)) {
    gLogFatal << "LeftPaddingCachePlugin axis_dim_=" << axis_dim_ << " not support now! " << endl;
    assert(0);
  }
  if (nb_dims < 3) {
    gLogFatal << "nbDims < 3 not support! " << endl;
    assert(0);
  }

  // except axis_dim_, all other dimensions are equal
  for (int i = 0; i < inputDesc.dims.nbDims; i++) {
    if (i != axis_dim_) {
      assert(inputDesc.dims.d[i] == in_cache.dims.d[i]);
      assert(inputDesc.dims.d[i] == output_desc.dims.d[i]);
      assert(inputDesc.dims.d[i] == out_cache.dims.d[i]);
    } else {
      assert(inputDesc.dims.d[axis_dim_] + in_cache.dims.d[axis_dim_] == output_desc.dims.d[axis_dim_]);
      assert(in_cache.dims.d[axis_dim_] == out_cache.dims.d[axis_dim_]);
    }
  }
}

int LeftPaddingCachePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                                    void* const* outputs, void* workspace, cudaStream_t stream) {
  int nb_dims = inputDesc[0].dims.nbDims;

  auto d = inputDesc[0].dims.d;

  int batch = accumulate(d, d + axis_dim_, 1, std::multiplies<int>());
  int input_len = d[axis_dim_];
  int dim = accumulate(d + axis_dim_ + 1, d + nb_dims, 1, std::multiplies<int>());

  int cache_len = inputDesc[1].dims.d[axis_dim_];

  if (data_type_ == DataType::kFLOAT) {
    const auto input = static_cast<const float*>(inputs[0]);
    const auto in_cache = static_cast<const float*>(inputs[1]);
    auto output = static_cast<float*>(outputs[0]);
    auto out_cache = static_cast<float*>(outputs[1]);

    auto ret = ComputeLeftPaddingCache(stream, batch, input_len, cache_len, dim, input, in_cache, output, out_cache);

    // print_data(input, ld, "input");
    // print_data(output, ld, "output");

    return ret;
  } else if (data_type_ == DataType::kHALF) {
    const auto input = static_cast<const half*>(inputs[0]);
    const auto in_cache = static_cast<const half*>(inputs[1]);
    auto output = static_cast<half*>(outputs[0]);
    auto out_cache = static_cast<half*>(outputs[1]);

    auto ret = ComputeLeftPaddingCache(stream, batch, input_len, cache_len, dim, input, in_cache, output, out_cache);
    return ret;
  }

  return -1;
}

// inputs are: input and cache
// outputs are: output and cache
nvinfer1::DimsExprs LeftPaddingCachePlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                                int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
  auto input_dims = inputs[0];
  auto cache_dims = inputs[1];
  if (outputIndex == 0) {
    auto output = inputs[0];
    output.d[axis_dim_] =
        exprBuilder.operation(DimensionOperation::kSUM, *input_dims.d[axis_dim_], *cache_dims.d[axis_dim_]);
    return output;
  } else {
    return cache_dims;
  }
}

nvinfer1::IPluginV2DynamicExt* LeftPaddingCachePlugin::clone() const {
  return new LeftPaddingCachePlugin(layer_name_, data_type_, axis_dim_);
}

void LeftPaddingCachePlugin::destroy() {
  // gLogVerbose << "LeftPaddingCachePlugin destroy\n";
  // This gets called when the network containing plugin is destroyed
}

const char* LeftPaddingCachePlugin::getPluginVersion() const { return LEFT_PADDING_CACHE_PLUGIN_VERSION; }

const char* LeftPaddingCachePlugin::getPluginType() const { return LEFT_PADDING_CACHE_PLUGIN_NAME; }

size_t LeftPaddingCachePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
                                                const nvinfer1::PluginTensorDesc* /*outputs*/,
                                                int /*nbOutputs*/) const TRTNOEXCEPT {
  return 0;
}

const char* LeftPaddingCachePlugin::getPluginNamespace() const { return ""; }

int LeftPaddingCachePlugin::getNbOutputs() const { return 2; }

void LeftPaddingCachePlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                                             nvinfer1::IGpuAllocator* allocator) {}

const char* LeftPaddingCachePluginCreator::getPluginName() const { return LEFT_PADDING_CACHE_PLUGIN_NAME; }

const char* LeftPaddingCachePluginCreator::getPluginVersion() const { return LEFT_PADDING_CACHE_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* LeftPaddingCachePluginCreator::getFieldNames() {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

nvinfer1::IPluginV2DynamicExt* LeftPaddingCachePluginCreator::createPlugin(const char* name,
                                                                           const nvinfer1::PluginFieldCollection* fc) {
  assert(fc->nbFields == 2);

  gLogVerbose << "Creating LeftPaddingCachePlugin...\n";

  int data_type_id;
  int axis_dim;

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("data_type") == 0) {
      data_type_id = static_cast<const int*>(fc->fields[i].data)[0];
      gLogVerbose << "Building data_type_id : " << data_type_id << std::endl;
    }
    if (field_name.compare("axis_dim") == 0) {
      axis_dim = static_cast<const int*>(fc->fields[i].data)[0];
      gLogVerbose << "Building axis_dim : " << axis_dim << std::endl;
    }
  }

  if (data_type_id < 0 || data_type_id > 1) {
    gLogError << "Invalid type id" << data_type_id << std::endl;
    assert(0);
  }

  DataType type = static_cast<DataType>(data_type_id);
  return new LeftPaddingCachePlugin(string(name), type, axis_dim);
}

nvinfer1::IPluginV2DynamicExt* LeftPaddingCachePluginCreator::deserializePlugin(const char* name,
                                                                                const void* serialData,
                                                                                size_t serialLength) {
  return new LeftPaddingCachePlugin(serialData, serialLength);
}

}  // namespace plugin
}  // namespace nvinfer1
