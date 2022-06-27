#ifndef PLUGIN_REL_POS_QKV_TO_CONTEXT_PLUGIN_H_
#define PLUGIN_REL_POS_QKV_TO_CONTEXT_PLUGIN_H_
#include <thrust/device_vector.h>
#include <iostream>
#include <string>

#include <NvInfer.h>

//#include "checkMacrosPlugin.h"
#include "common.h"

namespace nvinfer1 {
namespace plugin {

constexpr const char* REL_POS_QKV_TO_CONTEXT_PLUGIN_VERSION{"1"};
constexpr const char* REL_POS_QKV_TO_CONTEXT_PLUGIN_NAME{"RelPosQKVToContextPluginDynamic"};

int ComputeRelPosQKVToCtx(cublasHandle_t& cublas, const int B, const int S, const int num_heads, const int head_size,
    const float rsqrtHeadSize, const float* input, float* output, float* qkptr, float* pptr, float* tptr, cudaStream_t stream,
    const int* maskIdx);
int ComputeRelPosQKVToCtx(cublasHandle_t& cublas, const int B, const int S, const int num_heads, const int head_size,
    const float rsqrtHeadSize, const half* input, half* output, half* qkptr, half* pptr, half* tptr, cudaStream_t stream,
    const int* maskIdx);

//int compute_rel_pos_qkv_to_context(cudaStream_t stream, const int ld, const int B, const int N,
    //const float rsqrt_head_size, const int* mask_idx, const float* input, float* output);

//int compute_rel_pos_qkv_to_context(cudaStream_t stream, const int ld, const int B, const int N,
    //const float rsqrt_head_size, const int* mask_idx, const half* input, half* output);

class QKVToContextParams {
 public:
  int B, S, num_heads, head_size;
  float rsqrt_head_size;
  char* qu1_ptr;
  char* qu2_ptr;
  char* qv1_ptr;
  char* qv2_ptr;
  char* matrix_ac_ptr;
  char* matrix_bd_ptr;
  cudaStream_t stream;
  cudaStream_t stream1;
  cublasHandle_t cublas
};

class RelPosQKVToContextPlugin final : public nvinfer1::IPluginV2DynamicExt {
 protected:
  // Supress warnings about hiding function names due to overloads and overrides of virtuals.
  using IPluginV2DynamicExt::enqueue;
  using IPluginV2DynamicExt::getOutputDimensions;
  using IPluginV2DynamicExt::getWorkspaceSize;
  using IPluginV2DynamicExt::configurePlugin;

  size_t getSerializationSize() const override;
  void serialize(void* buffer) const override;

 public:
  RelPosQKVToContextPlugin(const std::string name, const nvinfer1::DataType type,
                           const int hidden_size, const int num_heads);
  RelPosQKVToContextPlugin(void const* serialData, size_t serialLength);

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
      int nbOutputs) override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
      int nbInputs) const override;
  int initialize() override;
  void terminate() override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
      const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
  int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
      const void* const* inputs, void* const* outputs, void* workspace,
      cudaStream_t stream) TRTNOEXCEPT override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
      int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;

  nvinfer1::IPluginV2DynamicExt* clone() const override;
  void destroy() override;
  const char* getPluginVersion() const override;
  const char* getPluginType() const override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
      const nvinfer1::PluginTensorDesc* /*outputs*/, int /*nbOutputs*/) const TRTNOEXCEPT override;
  void setPluginNamespace(const char* /*pluginNamespace*/) override {}
  const char* getPluginNamespace() const override;
  int getNbOutputs() const override;

  void attachToContext(cudnnContext* cudnn, cublasContext* cublas,
      nvinfer1::IGpuAllocator* allocator) override;
  void detachFromContext() override {}

 private:
  size_t scratchSize(const int B, const int S) const;

  const std::string layer_name_;
  nvinfer1::DataType data_type_;
  float rsqrt_head_size_;
  int batch_;
  int seq_len_;
  int hidden_size_;
  int head_size_;
  int num_heads_;
  bool has_imask_;
  cublasHandle_t cublas_handle_;
  cudaStream_t stream1_;

  WeightsWithOwnership pos_bias_u_;
  WeightsWithOwnership pos_bias_v_;

  cuda_unique_ptr<void> pos_bias_u_ptr_;
  cuda_unique_ptr<void> pos_bias_v_ptr_;
};

class RelPosQKVToContextPluginCreator : public nvinfer1::IPluginCreator {
 public:
  RelPosQKVToContextPluginCreator() {}
  ~RelPosQKVToContextPluginCreator() {}

  const char* getPluginName() const;
  const char* getPluginVersion() const;
  const nvinfer1::PluginFieldCollection* getFieldNames();
  nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name,
      const nvinfer1::PluginFieldCollection* fc);
  nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData,
      size_t serialLength);

  void setPluginNamespace(const char* libNamespace) {
    m_namespace_ = libNamespace;
  }

  const char* getPluginNamespace() const {
    return m_namespace_.c_str();
  }

 private:
  // static nvinfer1::PluginFieldCollection mFC;
  // static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string m_namespace_;
};

}  // namespace plugin
}  // namespace nvinfer1
#endif  // PLUGIN_REL_POS_QKV_TO_CONTEXT_PLUGIN_H_
