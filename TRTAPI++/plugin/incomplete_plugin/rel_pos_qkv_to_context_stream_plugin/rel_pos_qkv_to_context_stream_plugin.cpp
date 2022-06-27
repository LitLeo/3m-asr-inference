#include "rel_pos_qkv_to_context_stream_plugin.h"

#include <cuda_fp16.h>
#include <algorithm>

using namespace std;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
// PluginFieldCollection RelPosQKVToContextStreamPluginCreator::mFC{};
// std::vector<PluginField> RelPosQKVToContextStreamPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(RelPosQKVToContextStreamPluginCreator);

RelPosQKVToContextStreamPlugin::RelPosQKVToContextStreamPlugin(
    const std::string name, const nvinfer1::DataType type,
    const nvinfer1::Weights& pos_bias_u, const nvinfer1::Weights& pos_bias_v,
    const int chunk_size, const int left_chunk_num,
    const int hidden_size, const int num_heads)
    : layer_name_(name), data_type_(type), chunk_size_(chunk_size),
      left_chunk_num_(left_chunk_num), hidden_size_(hidden_size), num_heads_(num_heads) {
  assert(hidden_size_ % num_heads_ == 0);
  head_size_ = hidden_size_ / num_heads_;
  rsqrt_head_size_ = 1.f / sqrt(float(head_size_));

  pos_bias_u_.convertAndCopy(pos_bias_u, data_type_);
  copyToDevice(pos_bias_u_, getWeightsSize(pos_bias_u_, data_type_), pos_bias_u_ptr_);

  pos_bias_v_.convertAndCopy(pos_bias_v, data_type_);
  copyToDevice(pos_bias_v_, getWeightsSize(pos_bias_v_, data_type_), pos_bias_v_ptr_);
}

RelPosQKVToContextStreamPlugin::RelPosQKVToContextStreamPlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &data_type_);
  deserialize_value(&serialData, &serialLength, &chunk_size_);
  deserialize_value(&serialData, &serialLength, &left_chunk_num_);
  deserialize_value(&serialData, &serialLength, &num_heads_);
  deserialize_value(&serialData, &serialLength, &head_size_);
  deserialize_value(&serialData, &serialLength, &rsqrt_head_size_);
  deserialize_value(&serialData, &serialLength, &hidden_size_);

  // space for future to improve compatibility
  int tmp = hidden_size_;
  deserialize_value(&serialData, &serialLength, &tmp);
  deserialize_value(&serialData, &serialLength, &tmp);

  const char* d = static_cast<const char*>(serialData);

  pos_bias_u.convertAndCopy(d, hidden_size_, data_type_);
  copyToDevice(pos_bias_u, getWeightsSize(pos_bias_u, data_type_), pos_bias_u_ptr_);

  pos_bias_v.convertAndCopy(d, hidden_size_, data_type_);
  copyToDevice(pos_bias_v, getWeightsSize(pos_bias_v, data_type_), pos_bias_v_ptr_);
}

size_t RelPosQKVToContextStreamPlugin::getSerializationSize() const {
  size_t word_size = getElementSize(data_type_);
  return sizeof(data_type_) + sizeof(chunk_size_) + sizeof(left_chunk_num_)
         + sizeof(num_heads_) + sizeof(head_size_)
         + sizeof(rsqrt_head_size_) + sizeof(hidden_size_)
         + sizeof(hidden_size_) * 2 // for future
         + word_size * hidden_size_ *  2 // pos_bias_u and pos_bias_v
         ;
}

void RelPosQKVToContextStreamPlugin::serialize(void* buffer) const {
  serialize_value(&buffer, data_type_);
  serialize_value(&buffer, chunk_size_);
  serialize_value(&buffer, left_chunk_num_);
  serialize_value(&buffer, num_heads_);
  serialize_value(&buffer, head_size_);
  serialize_value(&buffer, rsqrt_head_size_);
  serialize_value(&buffer, hidden_size_);

  int tmp;
  serialize_value(&buffer, tmp);
  serialize_value(&buffer, tmp);

  size_t word_size = getElementSize(data_type_);
  char* d = static_cast<char*>(buffer);
  serFromDev(d, static_cast<char*>(pos_bias_u_ptr_.get()), hidden_size_ * word_size);
  serFromDev(d, static_cast<char*>(pos_bias_v_ptr_.get()), hidden_size_ * word_size);
}

bool RelPosQKVToContextStreamPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
  assert(pos >= 0 && pos <= 3);
  assert(nbInputs == 3);
  const auto* in = inOut;
  const auto* out = inOut + nbInputs;

  bool ret = false;
  if (pos == 0) {
    // input [B, C, 4D] or [B, C, 4D, 1, 1]
    if ((in->type == data_type_) && (in->format == TensorFormat::kLINEAR)) {
    }
  } else if (pos == 1) {
    const auto* in_cache = &inOut[1];
    auto dims = in_cache->dims;
    // self_cache: [B, L*C, 3D]
    if ((in_cache->type == data_type_) && (in_cache->format == TensorFormat::kLINEAR)) {
      if ((dims.nbDims == 3) &&
          ((dims.d[ATT_SDIM] == left_chunk_num_ * chunk_size_) &&
           (dims.d[ATT_HDIM] == 3 * hidden_size_)))
        ret = true;
    }
  } else if (pos == 2) {
    // decode_frame_num
    const auto* frame_num = &inOut[2];
    return (frame_num->type == DataType::kINT32) &&     // precision
           (frame_num->format == TensorFormat::kLINEAR) && // format
           (frame_num->dims.nbDims == 2) &&                // num dims
           ((frame_num->dims.d[1]) == in->dims.d[ATT_BDIM])    // check B
           ;
    }
  } else if (pos == 3) {
    // output
    if ((in->type == out->type) && (out->format == TensorFormat::kLINEAR) &&
        ((out->dims.d[ATT_BDIM]) == in->dims.d[ATT_BDIM]) &&
        ((out->dims.d[ATT_SDIM]) == in->dims.d[ATT_SDIM]) &&
        ((out->dims.d[ATT_HDIM]) == hidden_size_)) {
      if (out->dims.nbDims == 3)
        ret = true;
      // for fc
      if ((out->dims.nbDims == 5) && ((out->dims.d[3]) == 1) && ((out->dims.d[4]) == 1))
        ret = true;
    }
  }
  return ret;
}

nvinfer1::DataType RelPosQKVToContextStreamPlugin::getOutputDataType(int index,
    const nvinfer1::DataType* inputTypes, int nbInputs) const {
  assert(index == 0);
  assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
  return inputTypes[0];
}

int RelPosQKVToContextStreamPlugin::initialize() {
  CUDA_CHECK(cudaStreamCreate(&stream1_));
  CUDA_CHECK(cudaStreamCreate(&stream2_));
  return 0;
}

void RelPosQKVToContextStreamPlugin::terminate() {
  CUDA_CHECK(cudaStreamDestory(&stream1_));
  CUDA_CHECK(cudaStreamDestory(&stream2_));
}

void RelPosQKVToContextStreamPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) {
  assert(nbInputs == 3);
  assert(nbOutputs == 1);

  //auto in_desc = in[0].desc;
  //auto out_desc = out->desc;

  //assert(data_type_ == in_desc.type);
}

// x = x * self.xscale
// pos_emb = self.pe[:, offset:offset + seq_len]
int RelPosQKVToContextStreamPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) {

  QKVToContextStreamParams params;
  params.B = inputDesc[0].dims.d[ATT_BDIM];
  params.chunk_size = chunk_size_;
  params.left_chunk_num = left_chunk_num_;
  params.num_heads = num_heads_;
  params.head_size = head_size_;
  params.stream = stream;
  params.stream1 = stream1;
  params.cublas = cublas_handle_;

  const size_t word_size = getElementSize(data_type_);

  const size_t q_bytes = B * num_heads_ * C * head_size_ * word_size;
  const size_t matrix_bytes = B * num_heads_ * C * (left_chunk_num_+1) * C * word_size;
  const size_t trans_kvp_bytes = 3 * B * num_heads_ * (left_chunk_num_+1) * head_size_ * word_size;

  const size_t q_bytes_aligned = alignTo<size_t>(q_bytes, kAlignment)
  const size_t matrix_bytes_aligned = alignTo<size_t>(matrix_bytes, kAlignment)
  //const size_t trans_kvp_bytes_aligned = alignTo<size_t>(trans_kvp_bytes, kAlignment)

  params.trans_q = static_cast<params.(workspace);
  params.tqu = trans_q + q_bytes_aligned;
  params.tqv = tqu + q_bytes_aligned;
  params.matrix_ac = tqv + q_bytes_aligned;
  params.matrix_bd = matrix_ac + matrix_bytes_aligned;
  params.trans_kvp = matrix_bd + matrix_bytes_aligned;

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

int RelPosQKVToContextStreamPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
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

nvinfer1::DimsExprs RelPosQKVToContextStreamPlugin::getOutputDimensions(int outputIndex,
    const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
  // Input is BxCx4*N*H, output should be BxCxN*H
  // support two input dims, [B, C, 4*N*H] or [B, C, 4*N*H, 1, 1]
  assert(outputIndex == 0);
  // Copy over everything
  DimsExprs output(inputs[0]);
  // Divide last dim by four
  auto four = exprBuilder.constant(4);
  output.d[ATT_HDIM] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV,
                                             *inputs[0].d[ATT_HDIM], *four);
  return output;
}

nvinfer1::IPluginV2DynamicExt* RelPosQKVToContextStreamPlugin::clone() const {
  return new RelPosQKVToContextStreamPlugin(layer_name_, data_type_, pos_bias_u,
      pos_bias_v, chunk_size_, left_chunk_num_, hidden_size_, num_heads_);
}

void RelPosQKVToContextStreamPlugin::destroy() {
  // gLogVerbose << "RelPosQKVToContextStreamPlugin destroy\n";
  // This gets called when the network containing plugin is destroyed
}

const char* RelPosQKVToContextStreamPlugin::getPluginVersion() const {
  return REL_POS_QKV_TO_CONTEXT_STREAM_PLUGIN_VERSION;
}

const char* RelPosQKVToContextStreamPlugin::getPluginType() const {
  return REL_POS_QKV_TO_CONTEXT_STREAM_PLUGIN_NAME;
}

//size_t RelPosQKVToContextStreamPlugin::scratchSize(const int B, const int S) const {
  //const size_t word_size = getElementSize(data_type_);
  //size_t len = B * num_heads_* S * S;
  //const size_t bytes = len * word_size;

  //return bytes;
//}

size_t RelPosQKVToContextStreamPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const {

  const int B = inputs->dims.d[ATT_BDIM];
  const int C = inputs->dims.d[ATT_SDIM];

  const size_t word_size = getElementSize(data_type_);

  const size_t q_bytes = B * num_heads_ * C * head_size_ * word_size;
  const size_t matrix_bytes = B * num_heads_ * C * (left_chunk_num_+1) * C * word_size;
  const size_t trans_kvp_bytes = 3 * B * num_heads_ * (left_chunk_num_+1) * head_size_ * word_size;

  const size_t q_bytes_aligned = alignTo<size_t>(q_bytes, kAlignment)
  const size_t matrix_bytes_aligned = alignTo<size_t>(matrix_bytes, kAlignment)
  const size_t trans_kvp_bytes_aligned = alignTo<size_t>(trans_kvp_bytes, kAlignment)

  // (trans_Q + tQu + tQv) + (matrix_ac + matrix_bd) + trans_KVP
  return q_bytes_aligned*3 + matrix_bytes_aligned * 2 + trans_kvp_bytes_aligned;
}

const char* RelPosQKVToContextStreamPlugin::getPluginNamespace() const {
  return "";
}

int RelPosQKVToContextStreamPlugin::getNbOutputs() const {
  return 1;
}

void RelPosQKVToContextStreamPlugin::attachToContext(
    cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) {
  cublas_handle_ = cublas;
}

const char* RelPosQKVToContextStreamPluginCreator::getPluginName() const {
  return REL_POS_QKV_TO_CONTEXT_STREAM_PLUGIN_NAME;
}

const char* RelPosQKVToContextStreamPluginCreator::getPluginVersion() const {
  return REL_POS_QKV_TO_CONTEXT_STREAM_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* RelPosQKVToContextStreamPluginCreator::getFieldNames() {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

IPluginV2DynamicExt* RelPosQKVToContextStreamPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) {
  gLogVerbose << "Creating RelPosQKVToContextStreamPlugin...\n";

  int chunk_size = 0;
  int left_chunk_num = 0;
  int hidden_size = 0;
  int num_heads = 0;
  int type_id = -1;
  Weights pos_bias_u;
  Weights pos_bias_v;

  for (int i = 0; i < fc->nbFields; i++){
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("chunk_size") == 0) {
      chunk_size = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building chunk_size: " << chunk_size << std::endl;
    }
    if (field_name.compare("left_chunk_num_") == 0) {
      left_chunk_num_ = *static_cast<const int*>(fc->fields[i].data);
      gLogVerbose << "Building left_chunk_num_: " << left_chunk_num_ << std::endl;
    }
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
  RelPosQKVToContextStreamPlugin* p = new RelPosQKVToContextStreamPlugin(name, type, pos_bias_u,
      pos_bias_v, chunk_size_, left_chunk_num_, hidden_size, num_heads, has_imask);
  return p;
}

nvinfer1::IPluginV2DynamicExt* RelPosQKVToContextStreamPluginCreator::deserializePlugin(const char* name,
    const void* serialData, size_t serialLength) {
  return new RelPosQKVToContextStreamPlugin(serialData, serialLength);
}

}  // namespace plugin
}  // namespace nvinfer1
