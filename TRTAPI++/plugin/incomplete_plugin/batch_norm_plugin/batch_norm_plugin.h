#ifndef PLUGIN_BATCH_NORM_PLUGIN_BATCH_NORM_PLUGIN_H_
#define PLUGIN_BATCH_NORM_PLUGIN_BATCH_NORM_PLUGIN_H_
#include <thrust/device_vector.h>
#include <iostream>
#include <string>

#include <NvInfer.h>

#include "checkMacrosPlugin.h"
#include "common.h"

namespace nvinfer1 {
namespace plugin {

constexpr const char* BATCH_NORM_PLUGIN_VERSION{"1"};
constexpr const char* BATCH_NORM_PLUGIN_NAME{"BatchNormPluginDynamic"};

class BatchNormPlugin final : public nvinfer1::IPluginV2DynamicExt {
  const std::string layer_name_;
  nvinfer1::DataType data_type_;
  WeightsWithOwnership weight_;
  WeightsWithOwnership bias_;
  WeightsWithOwnership mean_;
  WeightsWithOwnership var_;
  int channel_num_;
  float eps_;
  cuda_unique_ptr<void> weight_dev_ptr_;
  cuda_unique_ptr<void> bias_dev_ptr_;
  cuda_unique_ptr<void> mean_dev_ptr_;
  cuda_unique_ptr<void> var_dev_ptr_;

 protected:
  // Supress warnings about hiding function names due to overloads and overrides of virtuals.
  using IPluginV2DynamicExt::configurePlugin;
  using IPluginV2DynamicExt::enqueue;
  using IPluginV2DynamicExt::getOutputDimensions;
  using IPluginV2DynamicExt::getWorkspaceSize;

  size_t getSerializationSize() const override;
  void serialize(void* buffer) const override;

 public:
  BatchNormPlugin(const std::string name, const nvinfer1::DataType type, const nvinfer1::Weights& weight,
                  const nvinfer1::Weights& bias, const nvinfer1::Weights& mean, const nvinfer1::Weights& var,
                  float eps);
  BatchNormPlugin(void const* serialData, size_t serialLength);

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                 int nbOutputs) override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;
  int initialize() override;
  void terminate() override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
  int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
              void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) override;

  nvinfer1::IPluginV2DynamicExt* clone() const override;
  void destroy() override;
  const char* getPluginVersion() const override;
  const char* getPluginType() const override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
                          const nvinfer1::PluginTensorDesc* /*outputs*/, int /*nbOutputs*/) const TRTNOEXCEPT override;
  void setPluginNamespace(const char* /*pluginNamespace*/) override {}
  const char* getPluginNamespace() const override;
  int getNbOutputs() const override;

  void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) override;
  void detachFromContext() override {}
};

class BatchNormPluginCreator : public nvinfer1::IPluginCreator {
 public:
  BatchNormPluginCreator() {}
  ~BatchNormPluginCreator() {}

  const char* getPluginName() const;
  const char* getPluginVersion() const;
  const nvinfer1::PluginFieldCollection* getFieldNames();
  nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc);
  nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength);

  void setPluginNamespace(const char* libNamespace) { m_namespace_ = libNamespace; }

  const char* getPluginNamespace() const { return m_namespace_.c_str(); }

 private:
  // static nvinfer1::PluginFieldCollection mFC;
  // static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string m_namespace_;
};

}  // namespace plugin
}  // namespace nvinfer1
#endif  // PLUGIN_BATCH_NORM_PLUGIN_BATCH_NORM_PLUGIN_H_
