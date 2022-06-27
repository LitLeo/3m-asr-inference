#ifndef TRT_RESHAPE_PLUGIN_H
#define TRT_RESHAPE_PLUGIN_H
#include <NvInfer.h>

#include "checkMacrosPlugin.h"
#include "serialize.hpp"

#include <iostream>
#include <string>
#include <thrust/device_vector.h>

namespace nvinfer1
{
namespace plugin
{

constexpr const char* RESHAPE_PLUGIN_VERSION{"1"};
constexpr const char* RESHAPE_PLUGIN_NAME{"ReshapePluginDynamic"};

class ReshapePlugin final : public nvinfer1::IPluginV2DynamicExt
{
    int _output_dim_num;
    // size: _output_dim_num * 3
    std::vector<int> _output_dim_params;

protected:
    // Supress warnings about hiding function names due to overloads and overrides of virtuals.
    using IPluginV2DynamicExt::enqueue;
    using IPluginV2DynamicExt::getOutputDimensions;
    using IPluginV2DynamicExt::getWorkspaceSize;
    using IPluginV2DynamicExt::configurePlugin;

    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;

public:

    //ReshapePlugin(int axis, int* const& output_lengths, int noutput)
        //: _axis(axis)
        //, _output_lengths(std::vector<int>(output_lengths, output_lengths + noutput))
    //{
        //assert(axis <= nvinfer1::Dims::MAX_DIMS);
    //}

    ReshapePlugin(std::vector<int> output_dim_params);
    ReshapePlugin(void const* serialData, size_t serialLength);

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;
    int initialize() override;
    void terminate() override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT override;
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;

    nvinfer1::IPluginV2DynamicExt* clone() const override;
    void destroy() override;
    const char* getPluginVersion() const override;
    const char* getPluginType() const override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
        const nvinfer1::PluginTensorDesc* /*outputs*/, int /*nbOutputs*/) const TRTNOEXCEPT override;
    void setPluginNamespace(const char* /*pluginNamespace*/) override {}
    const char* getPluginNamespace() const override;
    int getNbOutputs() const override;

    void attachToContext(
        cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, nvinfer1::IGpuAllocator* /*allocator*/) override {}
    void detachFromContext() override {}
};

class ReshapePluginCreator : public nvinfer1::IPluginCreator
{
public:
    ReshapePluginCreator() {}

    ~ReshapePluginCreator() {}

    const char* getPluginName() const;
    const char* getPluginVersion() const;
    const nvinfer1::PluginFieldCollection* getFieldNames();
    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc);
    nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength);

    void setPluginNamespace(const char* libNamespace)
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const
    {
        return mNamespace.c_str();
    }

private:
    //static nvinfer1::PluginFieldCollection mFC;
    //static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1
#endif // TRT_RESHAPE_PLUGIN_H
