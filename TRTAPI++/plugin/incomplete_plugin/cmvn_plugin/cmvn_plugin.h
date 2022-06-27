#ifndef PLUGIN_CMVN_PLUGIN_H_
#define PLUGIN_CMVN_PLUGIN_H_

#include <string>
#include <vector>

#include "NvInfer.h"

#include "common/common.h"

namespace nvinfer1 {
namespace plugin {

constexpr const char *CMVN_PLUGIN_VERSION{"1"};
constexpr const char *CMVN_PLUGIN_NAME{"CmvnPluginDynamic"};

class CmvnPlugin : public nvinfer1::IPluginV2DynamicExt {
 public:
  CmvnPlugin(const std::string name, const nvinfer1::DataType type,
             const nvinfer1::Weights mean, const nvinfer1::Weights var);

  CmvnPlugin(const std::string name, const void *data, size_t length);

  // It doesn't make sense to make CmvnPlugin without arguments, so we delete
  // default constructor.
  CmvnPlugin() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                          int nbInputs, nvinfer1::IExprBuilder &exprBuilder) override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut,
                                 int nbInputs, int nbOutputs) override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void* const* inputs, void* const* outputs,
              void *workspace, cudaStream_t stream) override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                       int nbInputs) const override;

  // IPluginV2 Methods
  const char *getPluginType() const override;
  const char *getPluginVersion() const override;
  int getNbOutputs() const override;
  int initialize() override;
  void terminate() override;
  size_t getSerializationSize() const override;
  void serialize(void *buffer) const override;
  void destroy() override;
  void setPluginNamespace(const char *pluginNamespace) override;
  const char *getPluginNamespace() const override;

 private:
  std::string layer_name_;
  std::string namespace_;

  nvinfer1::DataType data_type_;

  size_t dim_;
  nvinfer1::Weights mean_;
  nvinfer1::Weights var_;
  char *mean_dev_ptr_;
  char *var_dev_ptr_;

 protected:
  // To prevent compiler warnings.
  using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
  using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::supportsFormat;
  using nvinfer1::IPluginV2DynamicExt::configurePlugin;
  using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
  using nvinfer1::IPluginV2DynamicExt::enqueue;
};

class CmvnPluginCreator : public nvinfer1::IPluginCreator {
 public:
  CmvnPluginCreator();

  const char *getPluginName() const override;

  const char *getPluginVersion() const override;

  const nvinfer1::PluginFieldCollection *getFieldNames() override;

  nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) override;

  nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override;

  void setPluginNamespace(const char *pluginNamespace) override;

  const char *getPluginNamespace() const override;

private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string namespace_;
};

} // namespace plugin
} // namespace nvinfer1

#endif // PLUGIN_CMVN_PLUGIN_H_
