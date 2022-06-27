/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cuda_fp16.h>

#include "reshapePlugin.h"

#include "common.h"

using namespace std;
using namespace nvinfer1;
using nvinfer1::plugin::ReshapePlugin;
using nvinfer1::plugin::ReshapePluginCreator;

// Static class fields initialization
//PluginFieldCollection ReshapePluginCreator::mFC{};
//std::vector<PluginField> ReshapePluginCreator::mPluginAttributes;

//REGISTER_TENSORRT_PLUGIN(ReshapePluginCreator);

ReshapePlugin::ReshapePlugin(std::vector<int> output_dim_params)
    : _output_dim_num(output_dim_params.size() / 3)
    , _output_dim_params(output_dim_params)
{
    assert(_output_dim_params.size() % 3 == 0);
}

ReshapePlugin::ReshapePlugin(void const* serialData, size_t serialLength)
{
    deserialize_vector(&serialData, &serialLength, &_output_dim_params);
    _output_dim_num = _output_dim_params.size() / 3;
    assert(_output_dim_num * 3 == (int)_output_dim_params.size());
}

size_t ReshapePlugin::getSerializationSize() const
{
    return serialized_size(_output_dim_params);
}

void ReshapePlugin::serialize(void* buffer) const
{
    //serialize_value(&buffer, _output_dim_num);
    serialize_vector(&buffer, _output_dim_params);
}

bool ReshapePlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
}

nvinfer1::DataType ReshapePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
  ASSERT(inputTypes && nbInputs > 0);
  return inputTypes[0];
}

int ReshapePlugin::initialize()
{
  return 0;
}

void ReshapePlugin::terminate()
{

}

void ReshapePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                  const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
}

int ReshapePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
                         const void* const* inputs, void* const* outputs,
                         void* workspace,
                         cudaStream_t stream)
{
  float const* idata = reinterpret_cast<float  const*>(inputs[0]);
  float* odata = reinterpret_cast<float*>(outputs[0]);
  auto input_size = volume(inputDesc->dims);
  //auto output_size = volume(outputDesc->dims);
  cudaError_t cuda_status =
    cudaMemcpyAsync(odata, idata, input_size * sizeof(float*),
                    cudaMemcpyHostToDevice, stream);
  return cuda_status != cudaSuccess;
}

enum DimOp {
    kConstant = 0,
    kIndex = 1,
    kIndexAddIndex = 2,
    kIndexSubIndex = 3,
    kIndexMulIndex = 4,
};

nvinfer1::DimsExprs ReshapePlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    nvinfer1::DimsExprs output;
    output.nbDims = _output_dim_num;
    for (int i = 0; i < _output_dim_num; i++) {
        auto dim_op = _output_dim_params[i*3];
        auto operand1 = _output_dim_params[i*3 + 1];
        auto operand2 = _output_dim_params[i*3 + 2];
        if (operand1 < 0 || operand1 < 0) {
            gLogError << "operand1 < 0 || operand1 < 0! " << endl;
            assert(0);
        }
        switch (dim_op) {
            case kConstant: {
                output.d[i] = exprBuilder.constant(operand1);
                break;
            }
            case kIndex: {
                output.d[i] = inputs[0].d[operand1];
                break;
            }
            case kIndexAddIndex: {
                auto op = DimensionOperation::kSUM;
                auto first = inputs[0].d[operand1];
                auto second = inputs[0].d[operand2];
                output.d[i] = exprBuilder.operation(op, *first, *second);
                break;
            }
            case kIndexSubIndex: {
                auto op = DimensionOperation::kSUB;
                auto first = inputs[0].d[operand1];
                auto second = inputs[0].d[operand2];
                output.d[i] = exprBuilder.operation(op, *first, *second);
                break;
            }
            case kIndexMulIndex: {
                auto op = DimensionOperation::kPROD;
                auto first = inputs[0].d[operand1];
                auto second = inputs[0].d[operand2];
                output.d[i] = exprBuilder.operation(op, *first, *second);
                break;
            }
            default: {
                gLogError << "dim_op not support! " << dim_op << endl;
                assert(0);
            }
        }
    }
    return output;
}

nvinfer1::IPluginV2DynamicExt* ReshapePlugin::clone() const
{
    return new ReshapePlugin(_output_dim_params);
}
void ReshapePlugin::destroy()
{
    delete this;
}
const char* ReshapePlugin::getPluginVersion() const
{
    return RESHAPE_PLUGIN_VERSION;
}
const char* ReshapePlugin::getPluginType() const
{
    return RESHAPE_PLUGIN_NAME;
}
size_t ReshapePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
    const nvinfer1::PluginTensorDesc* /*outputs*/, int /*nbOutputs*/) const TRTNOEXCEPT
{
    return 0;
}
const char* ReshapePlugin::getPluginNamespace() const
{
    return "";
}
int ReshapePlugin::getNbOutputs() const
{
    return 1;
}

const char* ReshapePluginCreator::getPluginName() const
{
    return RESHAPE_PLUGIN_NAME;
}

const char* ReshapePluginCreator::getPluginVersion() const
{
    return RESHAPE_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* ReshapePluginCreator::getFieldNames()
{
    std::cerr << "Function not implemented" << std::endl;
    return nullptr;
}

nvinfer1::IPluginV2DynamicExt* ReshapePluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    assert(fc->nbFields == 1);

    vector<int> output_dim_params;
    output_dim_params.resize(fc->fields[0].length);
    memcpy(output_dim_params.data(), fc->fields[0].data,
           fc->fields[0].length * sizeof(float));

    return new ReshapePlugin(output_dim_params);
}

nvinfer1::IPluginV2DynamicExt* ReshapePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    return new ReshapePlugin(serialData, serialLength);
}

