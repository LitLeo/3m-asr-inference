#include "cat_split_cache_plugin.h"

#include <cuda_fp16.h>
#include <algorithm>

#include "common.h"
#include "debug.h"
#include "cat_split_cache_kernel.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
// PluginFieldCollection CatSplitCachePluginCreator::mFC{};
// std::vector<PluginField> CatSplitCachePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(CatSplitCachePluginCreator);

CatSplitCachePlugin::CatSplitCachePlugin(const std::string name, const nvinfer1::DataType type, const int axis_dim)
    : layer_name_(name), data_type_(type), axis_dim_(axis_dim) {}

CatSplitCachePlugin::CatSplitCachePlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &axis_dim_);

  // for future
  int tmp;
  deserialize_value(&serialData, &serialLength, &tmp);
  deserialize_value(&serialData, &serialLength, &tmp);
  deserialize_value(&serialData, &serialLength, &tmp);
}

size_t CatSplitCachePlugin::getSerializationSize() const TRTNOEXCEPT {
  return sizeof(data_type_) + sizeof(axis_dim_) + sizeof(int) * 3;
}

void CatSplitCachePlugin::serialize(void* buffer) const TRTNOEXCEPT {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, axis_dim_);

  int tmp = 0;
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);
}

bool CatSplitCachePlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                                    int nbOutputs) TRTNOEXCEPT {
  assert(nbInputs == 2);
  assert(nbOutputs == 2);

  // all inputs and inputs's type is data_type
  const PluginTensorDesc& inout = inOut[pos];
  return (inout.type == data_type_) && (inout.format == TensorFormat::kLINEAR);
}

nvinfer1::DataType CatSplitCachePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                          int nbInputs) const TRTNOEXCEPT {
  ASSERT(inputTypes && nbInputs > 0);
  return inputTypes[0];
}

int CatSplitCachePlugin::initialize() TRTNOEXCEPT { return 0; }

void CatSplitCachePlugin::terminate() TRTNOEXCEPT {}

void CatSplitCachePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                          const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) TRTNOEXCEPT {
  auto in_cache = in[0].desc;
  auto inputDesc = in[1].desc;
  auto output_desc = out[0].desc;
  auto out_cache = out[1].desc;

  // check axis_dim_
  auto nb_dims = inputDesc.dims.nbDims;
  if (axis_dim_ < 0) axis_dim_ = nb_dims - 1;
  if (axis_dim_ == 0) {
    gLogFatal << "CatSplitCachePlugin axis_dim_=" << axis_dim_ << " not support now! " << endl;
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

int CatSplitCachePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                 const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                                 void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT {
  int nb_dims = inputDesc[0].dims.nbDims;

  auto cache_d = inputDesc[0].dims.d;
  auto in_d = inputDesc[1].dims.d;

  int batch = accumulate(in_d, in_d + axis_dim_, 1, std::multiplies<int>());
  int input_dim = accumulate(in_d + axis_dim_, in_d + nb_dims, 1, std::multiplies<int>());
  int cache_dim = accumulate(cache_d + axis_dim_, cache_d + nb_dims, 1, std::multiplies<int>());

  if (data_type_ == DataType::kFLOAT) {
    const auto in_cache = static_cast<const float*>(inputs[0]);
    const auto input = static_cast<const float*>(inputs[1]);
    auto output = static_cast<float*>(outputs[0]);
    auto out_cache = static_cast<float*>(outputs[1]);

    auto ret = ComputeCatSplitCache(stream, batch, cache_dim, input_dim, in_cache, input, output, out_cache);

    // print_data(input, ld, "input");
    // print_data(output, ld, "output");

    return ret;
  } else if (data_type_ == DataType::kHALF) {
    const auto in_cache = static_cast<const half*>(inputs[0]);
    const auto input = static_cast<const half*>(inputs[1]);
    auto output = static_cast<half*>(outputs[0]);
    auto out_cache = static_cast<half*>(outputs[1]);

    auto ret = ComputeCatSplitCache(stream, batch, cache_dim, input_dim, in_cache, input, output, out_cache);
    return ret;
  }

  return -1;
}

// inputs are: input and cache
// outputs are: output and cache
nvinfer1::DimsExprs CatSplitCachePlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                             int nbInputs, nvinfer1::IExprBuilder& exprBuilder) TRTNOEXCEPT {
  auto cache_dims = inputs[0];
  auto input_dims = inputs[1];
  if (outputIndex == 0) {
    auto output = inputs[0];
    output.d[axis_dim_] =
        exprBuilder.operation(DimensionOperation::kSUM, *input_dims.d[axis_dim_], *cache_dims.d[axis_dim_]);
    return output;
  } else {
    return cache_dims;
  }
}

nvinfer1::IPluginV2DynamicExt* CatSplitCachePlugin::clone() const TRTNOEXCEPT {
  return new CatSplitCachePlugin(layer_name_, data_type_, axis_dim_);
}

void CatSplitCachePlugin::destroy() TRTNOEXCEPT {
  delete this;
}

const char* CatSplitCachePlugin::getPluginVersion() const TRTNOEXCEPT { return CAT_SPLIT_CACHE_PLUGIN_VERSION; }

const char* CatSplitCachePlugin::getPluginType() const TRTNOEXCEPT { return CAT_SPLIT_CACHE_PLUGIN_NAME; }

size_t CatSplitCachePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
                                             const nvinfer1::PluginTensorDesc* /*outputs*/,
                                             int /*nbOutputs*/) const TRTNOEXCEPT {
  return 0;
}

const char* CatSplitCachePlugin::getPluginNamespace() const TRTNOEXCEPT { return ""; }

int CatSplitCachePlugin::getNbOutputs() const TRTNOEXCEPT { return 2; }

void CatSplitCachePlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                                          nvinfer1::IGpuAllocator* allocator) TRTNOEXCEPT {}

const char* CatSplitCachePluginCreator::getPluginName() const TRTNOEXCEPT { return CAT_SPLIT_CACHE_PLUGIN_NAME; }

const char* CatSplitCachePluginCreator::getPluginVersion() const TRTNOEXCEPT { return CAT_SPLIT_CACHE_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* CatSplitCachePluginCreator::getFieldNames() TRTNOEXCEPT {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

nvinfer1::IPluginV2DynamicExt* CatSplitCachePluginCreator::createPlugin(const char* name,
                                                                        const nvinfer1::PluginFieldCollection* fc) TRTNOEXCEPT {
  assert(fc->nbFields == 2);

  gLogVerbose << "Creating CatSplitCachePlugin...\n";

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
  return new CatSplitCachePlugin(string(name), type, axis_dim);
}

nvinfer1::IPluginV2DynamicExt* CatSplitCachePluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                                             size_t serialLength) TRTNOEXCEPT {
  return new CatSplitCachePlugin(serialData, serialLength);
}

}  // namespace plugin
}  // namespace nvinfer1
