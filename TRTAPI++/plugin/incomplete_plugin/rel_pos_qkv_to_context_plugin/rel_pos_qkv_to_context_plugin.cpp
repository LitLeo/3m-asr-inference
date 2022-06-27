#include "rel_pos_qkv_to_context_plugin.h"

#include <cuda_fp16.h>
#include <algorithm>

using namespace std;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
// PluginFieldCollection RelPosQKVToContextPluginCreator::mFC{};
// std::vector<PluginField> RelPosQKVToContextPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(RelPosQKVToContextPluginCreator);

RelPosQKVToContextPlugin::RelPosQKVToContextPlugin(
    const std::string name, const nvinfer1::DataType type,
    const nvinfer1::Weights& pos_bias_u, const nvinfer1::Weights& pos_bias_v,
    const int hidden_size, const int num_heads, const bool has_imask)
    : layer_name_(name), data_type_(type), hidden_size_(hidden_size),
      num_heads_(num_heads), has_imask_(has_imask) {
  assert(hidden_size_ % num_heads_ == 0);
  head_size_ = hidden_size_ / num_heads_;
  rsqrt_head_size_ = 1.f / sqrt(float(head_size_));

  pos_bias_u_.convertAndCopy(pos_bias_u, data_type_);
  copyToDevice(pos_bias_u_, getWeightsSize(pos_bias_u_, data_type_), pos_bias_u_ptr_);

  pos_bias_v_.convertAndCopy(pos_bias_v, data_type_);
  copyToDevice(pos_bias_v_, getWeightsSize(pos_bias_v_, data_type_), pos_bias_v_ptr_);
}

RelPosQKVToContextPlugin::RelPosQKVToContextPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &num_heads_);
  deserialize_value(&serialData, &serialLength, &head_size_);
  deserialize_value(&serialData, &serialLength, &rsqrt_head_size_);
  deserialize_value(&serialData, &serialLength, &has_imask_);
  deserialize_value(&serialData, &serialLength, &hidden_size_);

  const char* d = static_cast<const char*>(serialData);

  pos_bias_u.convertAndCopy(d, hidden_size_, data_type_);
  copyToDevice(pos_bias_u, getWeightsSize(pos_bias_u, data_type_), pos_bias_u_ptr_);

  pos_bias_v.convertAndCopy(d, hidden_size_, data_type_);
  copyToDevice(pos_bias_v, getWeightsSize(pos_bias_v, data_type_), pos_bias_v_ptr_);
}

size_t RelPosQKVToContextPlugin::getSerializationSize() const {
  size_t word_size = getElementSize(data_type_);
  return sizeof(data_type_) + sizeof(num_heads_) + sizeof(head_size_)
         + sizeof(rsqrt_head_size_) + sizeof(has_imask_) + sizeof(hidden_size_)
         + word_size * hidden_size_ *  2 // pos_bias_u and pos_bias_v
         ;
}

void RelPosQKVToContextPlugin::serialize(void* buffer) const {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, num_heads_);
  serialize_value(&buffer, head_size_);
  serialize_value(&buffer, rsqrt_head_size_);
  serialize_value(&buffer, has_imask_);
  serialize_value(&buffer, hidden_size_);

  size_t word_size = getElementSize(data_type_);
  char* d = static_cast<char*>(buffer);
  serFromDev(d, static_cast<char*>(pos_bias_u_ptr_.get()), hidden_size_ * word_size);
  serFromDev(d, static_cast<char*>(pos_bias_v_ptr_.get()), hidden_size_ * word_size);
}

bool RelPosQKVToContextPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
  assert(pos >= 0);
  assert(pos < 2 + has_imask_);
  assert(nbInputs == 1 + has_imask_);
  const auto* in = inOut;
  const auto* out = inOut + nbInputs;
  if (pos == 0) {
    // must not check descriptions > pos
    bool ret = false;
    if ((in->type == data_type_) && (in->format == TensorFormat::kLINEAR)) {
      // for trt fully_connected output
      if ((in->dims.nbDims == 5) && ((in->dims.d[ATT_HDIM] % 3) == 0) && ((in->dims.d[3]) == 1) && ((in->dims.d[4]) == 1))
        ret = true;
      if ((in->dims.nbDims == 3) && ((in->dims.d[ATT_HDIM] % 3) == 0))
        ret = true;
    }
    return ret;
  } else { // pos==1
    // has mask
    if ((has_imask_ && pos == 1)) {
      const auto* in_mask = &inOut[1];
      return (in_mask->type == DataType::kINT32) &&     // precision
             (in_mask->format == TensorFormat::kLINEAR) && // format
             (in_mask->dims.nbDims == 2) &&                // num dims
             ((in_mask->dims.d[1]) == in->dims.d[ATT_BDIM])    // check B
             ;
    }
    if (!has_imask_ || (pos == 2)) {
      bool ret = false;
      if ((in->type == out->type) && (out->format == TensorFormat::kLINEAR) &&
          ((out->dims.d[ATT_BDIM]) == in->dims.d[ATT_BDIM]) &&
          ((out->dims.d[ATT_SDIM]) == in->dims.d[ATT_SDIM])) {
        if (out->dims.nbDims == 3)
          ret = true;
        // for fc
        if ((out->dims.nbDims == 5) && ((out->dims.d[3]) == 1) && ((out->dims.d[4]) == 1))
          ret = true;
      }
      return ret;
    }
  }
  return false;
}

nvinfer1::DataType RelPosQKVToContextPlugin::getOutputDataType(int index,
    const nvinfer1::DataType* inputTypes, int nbInputs) const {
  assert(index == 0);
  assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
  return inputTypes[0];
}

int RelPosQKVToContextPlugin::initialize() {
  CUDA_CHECK(cudaStreamCreate(&stream1_));
  return 0;
}

void RelPosQKVToContextPlugin::terminate() {
  CUDA_CHECK(cudaStreamDestory(&stream1_));
}

void RelPosQKVToContextPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) {
  assert(nbInputs == 1 + has_imask_);
  assert(nbOutputs == 1);
  const PluginTensorDesc& in_desc = in[0].desc;
  TRT_UNUSED in_desc;
  const PluginTensorDesc& out_desc = out->desc;
  TRT_UNUSED out_desc;
  assert(mType == in_desc.type);
  assert(mType == out_desc.type);
  assert(in_desc.dims.d[ATT_BDIM] == out_desc.dims.d[ATT_BDIM]);
  assert(in_desc.dims.d[ATT_SDIM] == out_desc.dims.d[ATT_SDIM]);
  assert(in_desc.dims.d[ATT_HDIM] == 4 * out_desc.dims.d[ATT_HDIM]);
  if (mHasImask) {
    const PluginTensorDesc& mask_desc = in[1].desc;
    TRT_UNUSED mask_desc;
    assert(mask_desc.type == DataType::kINT32);
    // [1, B]
    assert(mask_desc.dims.d[0] == 1);
    assert(mask_desc.dims.d[1] == in_desc.dims.d[ATT_BDIM]);
  }
}

// x = x * self.xscale
// pos_emb = self.pe[:, offset:offset + seq_len]
int RelPosQKVToContextPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) {
  auto batch = inputDesc[0].dims.d[0];
  auto seq_len = inputDesc[0].dims.d[1];
  auto dim = inputDesc[0].dims.d[2];

  if (data_type_ == DataType::kFLOAT) {
    const auto input = static_cast<const float*>(inputs[0]);
    const auto masked = static_cast<const int*>(inputs[1]);
    auto output = static_cast<float*>(outputs[0]);

    auto ret = compute_att_masked_softmax(stream, dim, batch, seq_len, scale_, masked, input, output);
    return ret;
  } else if (data_type_ == DataType::kHALF) {
    const auto input = static_cast<const half*>(inputs[0]);
    const auto masked = static_cast<const int*>(inputs[1]);
    auto output = static_cast<half*>(outputs[0]);

    auto ret = compute_att_masked_softmax(stream, dim, batch, seq_len, scale_, masked, input, output);
    return ret;
  }

  return 0;
}

int RelPosQKVToContextPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) {
  const int B = inputDesc->dims.d[ATT_BDIM];
  const int S = inputDesc->dims.d[ATT_SDIM];

  const size_t q_bytes_aligned = alignTo<size_t>(B*S*hidden_size_, kAlignment);
  const size_t score_bytes_aligned = alignTo<size_t>(B*N*S*S, kAlignment);
  char* scratch1 = static_cast<char*>(workspace);
  char* scratch2 = scratch1 + q_bytes_aligned;
  char* scratch3 = scratch2 + score_bytes_aligned;
  char* scratch4 = scratch2 + score_bytes_aligned;

  const int* mask_idx = has_imask_ ? static_cast<const int*>(inputs[1]) : nullptr;

  int status = -1;
  if (data_type_ == DataType::kFLOAT) {
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    float* scr1 = reinterpret_cast<float*>(scratch1);
    float* scr2 = reinterpret_cast<float*>(scratch2);
    float* scr3 = reinterpret_cast<float*>(scratch3);

    status = compute_rel_pos_qkv_to_ctx(cublas_handle_, B, S, num_heads_,
        head_size_, rsqrt_head_size_, input, output, scr1, scr2, scr3, stream, mask_idx);
  } else if (data_type_ == DataType::kHALF) {
    const half* input = static_cast<const half*>(inputs[0]);
    half* output = static_cast<half*>(outputs[0]);
    half* scr1 = reinterpret_cast<half*>(scratch1);
    half* scr2 = reinterpret_cast<half*>(scratch2);
    half* scr3 = reinterpret_cast<half*>(scratch3);

    status = compute_rel_pos_qkv_to_ctx(cublas_handle_, B, S, num_heads_,
        head_size_, rsqrt_head_size_, input, output, scr1, scr2, scr3, stream, mask_idx);
  }

  return status;
}

nvinfer1::DimsExprs RelPosQKVToContextPlugin::getOutputDimensions(int outputIndex,
    const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
  // Input is BxSx4*N*H, output should be BxSxN*H
  // support two input dims, [B, S, 4*N*H] or [B, S, 4*N*H, 1, 1]
  assert(outputIndex == 0);
  // Copy over everything
  DimsExprs output(inputs[0]);
  // Divide last dim by four
  auto four = exprBuilder.constant(4);
  output.d[ATT_HDIM] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV,
                                             *inputs[0].d[ATT_HDIM], *four);
  return output;
}

nvinfer1::IPluginV2DynamicExt* RelPosQKVToContextPlugin::clone() const {
  return new RelPosQKVToContextPlugin(layer_name_, data_type_, pos_bias_u,
      pos_bias_v, hidden_size_, num_heads_);
}

void RelPosQKVToContextPlugin::destroy() {
  // gLogVerbose << "RelPosQKVToContextPlugin destroy\n";
  // This gets called when the network containing plugin is destroyed
}

const char* RelPosQKVToContextPlugin::getPluginVersion() const {
  return REL_POS_QKV_TO_CONTEXT_PLUGIN_VERSION;
}

const char* RelPosQKVToContextPlugin::getPluginType() const {
  return REL_POS_QKV_TO_CONTEXT_PLUGIN_NAME;
}

//size_t RelPosQKVToContextPlugin::scratchSize(const int B, const int S) const {
  //const size_t word_size = getElementSize(data_type_);
  //size_t len = B * num_heads_* S * S;
  //const size_t bytes = len * word_size;

  //return bytes;
//}

size_t RelPosQKVToContextPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const {
  const int B = inputs->dims.d[ATT_BDIM];
  const int S = inputs->dims.d[ATT_SDIM];

  const size_t word_size = getElementSize(data_type_);

  const size_t Q_bytes_aligned = alignTo<size_t>(B*S*hidden_size_, kAlignment)
  const size_t score_bytes_aligned = alignTo<size_t>(B*num_heads_*S*S, kAlignment)

  // [(B,S,N,H)*2 + (B,N,S,S)] * 2 + (B,N,S,S)
  return (Q_bytes_aligned*2+ score_bytes_aligned) * 2;
}

const char* RelPosQKVToContextPlugin::getPluginNamespace() const {
  return "";
}

int RelPosQKVToContextPlugin::getNbOutputs() const {
  return 1;
}

void RelPosQKVToContextPlugin::attachToContext(
    cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) {
  cublas_handle_ = cublas;
}

const char* RelPosQKVToContextPluginCreator::getPluginName() const {
  return REL_POS_QKV_TO_CONTEXT_PLUGIN_NAME;
}

const char* RelPosQKVToContextPluginCreator::getPluginVersion() const {
  return REL_POS_QKV_TO_CONTEXT_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* RelPosQKVToContextPluginCreator::getFieldNames() {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

IPluginV2DynamicExt* RelPosQKVToContextPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) {
  gLogVerbose << "Creating RelPosQKVToContextPlugin...\n";

  int hidden_size = 0;
  int num_heads = 0;
  bool has_imask = false;
  int type_id = -1;
  Weights pos_bias_u;
  Weights pos_bias_v;

  for (int i = 0; i < fc->nbFields; i++){
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("data_type") == 0) {
      type_id = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building type_id: " << type_id << std::endl;
    }
    if (field_name.compare("hidden_size") == 0) {
      hidden_size = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building hidden_size: " << hidden_size << std::endl;
    }
    if (field_name.compare("num_heads") == 0) {
      num_heads = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building num_heads: " << num_heads << std::endl;
    }
    if (field_name.compare("has_mask") == 0) {
      has_imask = *static_cast<const bool*>(fc->fields[i].data);
      gLogVerbose << "Building has_imask: " << has_imask << std::endl;
    }

    if (field_name.compare("pos_bias_u") == 0) {
      gLogVerbose << "Building pos_bias_u...\n";
      pos_bias_u.values = fc->fields[i].data;
      pos_bias_u.count = fc->fields[i].length;
      pos_bias_u.type = fieldTypeToDataType(fc->fields[i].type);
      gLogVerbose << "Is pos_bias_u float32: " << (pos_bias_u.type == DataType::kFLOAT) << endl;
    }

    if (field_name.compare("pos_bias_v") == 0) {
      gLogVerbose << "Building pos_bias_v...\n";
      pos_bias_v.values = fc->fields[i].data;
      pos_bias_v.count = fc->fields[i].length;
      pos_bias_v.type = fieldTypeToDataType(fc->fields[i].type);
      gLogVerbose << "Is pos_bias_v float32: " << (pos_bias_v.type == DataType::kFLOAT) << endl;
    }
  }
  if ((type_id != 0 || type_id != 1) || (hidden_size <= 0) || (num_heads <= 0)) {
    gLogError << "QKV: Invalid params << std::endl;
    assert(0);
  }

  gLogVerbose << "Building the Plugin...\n";
  DataType type = static_cast<DataType>(type_id);
  RelPosQKVToContextPlugin* p = new RelPosQKVToContextPlugin(name, type, pos_bias_u,
      pos_bias_v, hidden_size, num_heads, has_imask);
  return p;
}

nvinfer1::IPluginV2DynamicExt* RelPosQKVToContextPluginCreator::deserializePlugin(const char* name,
    const void* serialData, size_t serialLength) {
  return new RelPosQKVToContextPlugin(serialData, serialLength);
}

}  // namespace plugin
}  // namespace nvinfer1
