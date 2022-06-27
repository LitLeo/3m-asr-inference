// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "mask_conv2d_sample_plugin.h"

#include <algorithm>
#include <numeric>

#include "common.h"
#include "mask_conv2d_sample_kernel.h"

using namespace std;

namespace nvinfer1 {
namespace plugin {

// Static class fields initialization
// PluginFieldCollection MaskConv2dSamplePluginCreator::mFC{};
// std::vector<PluginField> MaskConv2dSamplePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(MaskConv2dSamplePluginCreator);

MaskConv2dSamplePlugin::MaskConv2dSamplePlugin(const std::string name, const int left_padding, const int stride)
    : layer_name_(name), left_padding_(left_padding), stride_(stride) {}

MaskConv2dSamplePlugin::MaskConv2dSamplePlugin(void const* serialData, size_t serialLength) {
  deserialize_value(&serialData, &serialLength, &left_padding_);
  deserialize_value(&serialData, &serialLength, &stride_);
}

size_t MaskConv2dSamplePlugin::getSerializationSize() const TRTNOEXCEPT { return sizeof(left_padding_) + sizeof(stride_); }

void MaskConv2dSamplePlugin::serialize(void* buffer) const TRTNOEXCEPT {
  serialize_value(&buffer, left_padding_);
  serialize_value(&buffer, stride_);
}

bool MaskConv2dSamplePlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                                       int nbOutputs) TRTNOEXCEPT {
  if (nbInputs != 1 || nbOutputs != 1) return false;
  if (inOut[pos].type != nvinfer1::DataType::kINT32) return false;
  return true;
}

nvinfer1::DataType MaskConv2dSamplePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                                             int nbInputs) const TRTNOEXCEPT {
  return inputTypes[0];
}

int MaskConv2dSamplePlugin::initialize() TRTNOEXCEPT { return 0; }

void MaskConv2dSamplePlugin::terminate() TRTNOEXCEPT {}

void MaskConv2dSamplePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                             const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) TRTNOEXCEPT {}

// x = x * self.xscale
// pos_emb = self.pe[:, offset:offset + seq_len]
int MaskConv2dSamplePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                                    void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT {
  int batch = volume(inputDesc[0].dims);

  const auto input = static_cast<const int*>(inputs[0]);
  auto output = static_cast<int*>(outputs[0]);

  auto ret = compute_mask_conv2d_sample(input, batch, left_padding_, stride_, output, stream);
  return ret;
}

nvinfer1::DimsExprs MaskConv2dSamplePlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                                                int nbInputs, nvinfer1::IExprBuilder& exprBuilder) TRTNOEXCEPT {
  return inputs[0];
}

nvinfer1::IPluginV2DynamicExt* MaskConv2dSamplePlugin::clone() const TRTNOEXCEPT {
  return new MaskConv2dSamplePlugin(layer_name_, left_padding_, stride_);
}

void MaskConv2dSamplePlugin::destroy() TRTNOEXCEPT {
  delete this;
}

const char* MaskConv2dSamplePlugin::getPluginVersion() const TRTNOEXCEPT { return MASK_CONV2D_SAMPLE_PLUGIN_VERSION; }

const char* MaskConv2dSamplePlugin::getPluginType() const TRTNOEXCEPT { return MASK_CONV2D_SAMPLE_PLUGIN_NAME; }

size_t MaskConv2dSamplePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* /*inputs*/, int /*nbInputs*/,
                                                const nvinfer1::PluginTensorDesc* /*outputs*/,
                                                int /*nbOutputs*/) const TRTNOEXCEPT {
  return 0;
}

const char* MaskConv2dSamplePlugin::getPluginNamespace() const TRTNOEXCEPT { return ""; }

int MaskConv2dSamplePlugin::getNbOutputs() const TRTNOEXCEPT { return 1; }

void MaskConv2dSamplePlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                                             nvinfer1::IGpuAllocator* allocator) TRTNOEXCEPT {}

const char* MaskConv2dSamplePluginCreator::getPluginName() const TRTNOEXCEPT { return MASK_CONV2D_SAMPLE_PLUGIN_NAME; }

const char* MaskConv2dSamplePluginCreator::getPluginVersion() const TRTNOEXCEPT { return MASK_CONV2D_SAMPLE_PLUGIN_VERSION; }

const nvinfer1::PluginFieldCollection* MaskConv2dSamplePluginCreator::getFieldNames() TRTNOEXCEPT {
  std::cerr << "Function not implemented" << std::endl;
  return nullptr;
}

nvinfer1::IPluginV2DynamicExt* MaskConv2dSamplePluginCreator::createPlugin(const char* name,
                                                                           const nvinfer1::PluginFieldCollection* fc) TRTNOEXCEPT {
  assert(fc->nbFields == 2);

  gLogVerbose << "Creating MaskConv2dSamplePlugin...\n";

  int left_padding = 0, stride = 1;

  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("left_padding") == 0) {
      left_padding = static_cast<const int*>(fc->fields[i].data)[0];
      gLogVerbose << "Building left_padding : " << left_padding << std::endl;
    }

    if (field_name.compare("stride") == 0) {
      stride = static_cast<const int*>(fc->fields[i].data)[0];
      gLogVerbose << "Building stride: " << stride << std::endl;
    }
  }

  return new MaskConv2dSamplePlugin(string(name), left_padding, stride);
}

nvinfer1::IPluginV2DynamicExt* MaskConv2dSamplePluginCreator::deserializePlugin(const char* name,
                                                                                const void* serialData,
                                                                                size_t serialLength) TRTNOEXCEPT {
  return new MaskConv2dSamplePlugin(serialData, serialLength);
}

}  // namespace plugin
}  // namespace nvinfer1
