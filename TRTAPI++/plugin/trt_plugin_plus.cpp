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

#include "trt_plugin_plus.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "NvInferRuntimeCommon.h"
#include "common.h"
#include "logger.h"

#include "glu_plugin.h"
#include "layer_norm_plugin.h"
//#include "silu_plugin.h"

#ifdef BUILD_LIBTORCH_PLUGINS
#include "batch_norm_plugin.h"
#include "celu_plugin.h"
#include "group_norm_plugin.h"
#endif  // BUILD_LIBTORCH_PLUGINS

// your self plugin
#ifdef BUILD_SELF_PLUGINS
#include "att_masked_softmax_plugin.h"
#include "att_stream_softmax_plugin.h"
#include "cat_split_cache_plugin.h"
#include "dump_tensor_plugin.h"
#include "fmoe_expert_plugin.h"
#include "mask_conv2d_sample_plugin.h"
#include "masked_fill_plugin.h"
#include "rel_positional_encoding_plugin.h"
#endif  // BUILD_SELF_PLUGINS

using namespace std;

namespace nvinfer1 {
namespace plugin {

// This singleton ensures that each plugin is only registered once for a given
// namespace and type, and attempts of duplicate registration are ignored.
class PluginCreatorRegistry {
 public:
  static PluginCreatorRegistry& get_instance() {
    static PluginCreatorRegistry instance;
    return instance;
  }

  template <typename CreatorType>
  void add_plugin_creator(void* logger, const char* libNamespace) {
    // Make accesses to the plugin creator registry thread safe
    std::lock_guard<std::mutex> lock(m_registry_lock_);

    std::string errorMsg;
    std::string verboseMsg;

    std::unique_ptr<CreatorType> pluginCreator{new CreatorType{}};
    pluginCreator->setPluginNamespace(libNamespace);

    auto logger_ptr = static_cast<nvinfer1::ILogger*>(logger);
    string pluginType = string{pluginCreator->getPluginNamespace()} + "::" +
                        string{pluginCreator->getPluginName()} + " version " +
                        string{pluginCreator->getPluginVersion()};

    if (m_registry_list_.find(pluginType) == m_registry_list_.end()) {
      bool status = getPluginRegistry()->registerCreator(*pluginCreator, libNamespace);
      if (status) {
        m_registry_.push(std::move(pluginCreator));
        m_registry_list_.insert(pluginType);
        verboseMsg = "Registered plugin creator - " + pluginType;
      } else {
        errorMsg = "Could not register plugin creator -  " + pluginType;
      }
    } else {
      verboseMsg = "Plugin creator already registered - " + pluginType;
    }

    if (logger) {
      if (!errorMsg.empty()) {
        logger_ptr->log(ILogger::Severity::kERROR, errorMsg.c_str());
      }
      if (!verboseMsg.empty()) {
        logger_ptr->log(ILogger::Severity::kVERBOSE, verboseMsg.c_str());
      }
    }
  }

  ~PluginCreatorRegistry() {
    std::lock_guard<std::mutex> lock(m_registry_lock_);

    // Release pluginCreators in LIFO order of registration.
    while (!m_registry_.empty()) {
      m_registry_.pop();
    }
    m_registry_list_.clear();
  }

 private:
  PluginCreatorRegistry() {}

  std::mutex m_registry_lock_;
  std::stack<std::unique_ptr<IPluginCreator>> m_registry_;
  std::unordered_set<std::string> m_registry_list_;

 public:
  PluginCreatorRegistry(PluginCreatorRegistry const&) = delete;
  void operator=(PluginCreatorRegistry const&) = delete;
};

template <typename CreatorType>
void initialize_plugin(void* logger, const char* libNamespace) {
  PluginCreatorRegistry::get_instance().add_plugin_creator<CreatorType>(logger, libNamespace);
}

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

template <typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

bool init_trt_plugin_plus(void* logger, const char* libNamespace) {
  initialize_plugin<nvinfer1::plugin::GluPluginCreator>(logger, libNamespace);
  initialize_plugin<nvinfer1::plugin::LayerNormPluginCreator>(logger, libNamespace);
  //initialize_plugin<nvinfer1::plugin::SiluPluginCreator>(logger, libNamespace);

#ifdef BUILD_LIBTORCH_PLUGINS
  initialize_plugin<nvinfer1::plugin::CeluPluginCreator>(logger, libNamespace);
  initialize_plugin<nvinfer1::plugin::GroupNormPluginCreator>(logger, libNamespace);
#endif  // BUILD_LIBTORCH_PLUGINS

// your self plugin
#ifdef BUILD_SELF_PLUGINS
  initialize_plugin<nvinfer1::plugin::AttMaskedSoftmaxPluginCreator>(logger, libNamespace);
  initialize_plugin<nvinfer1::plugin::AttStreamSoftmaxPluginCreator>(logger, libNamespace);
  initialize_plugin<nvinfer1::plugin::CatSplitCachePluginCreator>(logger, libNamespace);
  initialize_plugin<nvinfer1::plugin::DumpTensorPluginCreator>(logger, libNamespace);
  initialize_plugin<nvinfer1::plugin::FMoEExpertPluginCreator>(logger, libNamespace);
  initialize_plugin<nvinfer1::plugin::MaskConv2dSamplePluginCreator>(logger, libNamespace);
  initialize_plugin<nvinfer1::plugin::MaskedFillPluginCreator>(logger, libNamespace);
  initialize_plugin<nvinfer1::plugin::RelPositionalEncodingPluginCreator>(logger, libNamespace);
#endif  // BUILD_SELF_PLUGINS

  return true;
}

IPluginCreator* get_plugin_creator(const char* plugin_name, const char* plugin_version) {
  auto creator = getPluginRegistry()->getPluginCreator(plugin_name, plugin_version);
  if (!creator) {
    gLogError << "plugin_name:" << plugin_name << ", version:" << plugin_version << "not found!" << endl;
    return {};
  }
  return creator;
}

/*******************************add_glu_plugin************************************/
ITensorVector AddGluPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                           const nvinfer1::DataType type, const int axis_dim) {
  if (input_tensors.size() != 1) {
    gLogFatal << "input_tensors.size() != 1. glu only support 1 input!" << endl;
    assert(0);
  }
  // find creator
  auto creator = get_plugin_creator(GLU_PLUGIN_NAME, GLU_PLUGIN_VERSION);

  // create plugin
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("data_type", &type, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("axis_dim", &axis_dim, nvinfer1::PluginFieldType::kINT32, 1);

  const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
  const auto plugin_obj = InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("GluPlugin", &plugin_data));

  auto plugin = network->addPluginV2(&input_tensors[0], 1, *plugin_obj);
  if (plugin == nullptr) {
    gLogError << "Create Network: Fail to create [GluPlugin] layer.";
    return {};
  }

  plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [GluPlugin]")).c_str());

  return {plugin->getOutput(0)};
}

/*******************************AddLayerNormPlugin************************************/
ITensorVector AddLayerNormPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                    const nvinfer1::DataType type, const size_t dim, const float eps) {
  // find creator
  auto creator = get_plugin_creator(LAYER_NORM_NAME, LAYER_NORM_VERSION);

  // create plugin
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("data_type", &type, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("dim", &dim, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("eps", &eps, nvinfer1::PluginFieldType::kFLOAT32, 1);

  const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
  const auto plugin_obj =
      InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("LayerNormDynamicPlugin", &plugin_data));

  auto plugin = network->addPluginV2(&input_tensors[0], 3, *plugin_obj);
  if (plugin == nullptr) {
    gLogError << "Create Network: Fail to create [LayerNormDynamicPlugin] layer.";
    return {};
  }

  plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [LayerNormDynamicPlugin]")).c_str());

  return {plugin->getOutput(0)};
}

//[>******************************add_silu_plugin***********************************<]
//ITensorVector AddSiluPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                            //const nvinfer1::DataType type) {
  //if (input_tensors.size() != 1) {
    //gLogFatal << "input_tensors.size() != 1. silu only support 1 input!" << endl;
    //assert(0);
  //}
  //// find creator
  //auto creator = get_plugin_creator(SILU_PLUGIN_NAME, SILU_PLUGIN_VERSION);

  //// create plugin
  //std::vector<nvinfer1::PluginField> field_data;
  //field_data.emplace_back("data_type", &type, nvinfer1::PluginFieldType::kINT32, 1);

  //const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
  //const auto plugin_obj = InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("SiluPlugin", &plugin_data));

  //auto plugin = network->addPluginV2(&input_tensors[0], 1, *plugin_obj);
  //if (plugin == nullptr) {
    //gLogError << "Create Network: Fail to create [SiluPlugin] layer.";
    //return {};
  //}

  //plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [SiluPlugin]")).c_str());

  //return {plugin->getOutput(0)};
//}

#ifdef BUILD_LIBTORCH_PLUGINS

/*******************************add_celu_plugin************************************/
ITensorVector add_celu_plugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                              const nvinfer1::DataType type, const float alpha) {
  if (input_tensors.size() != 1) {
    gLogFatal << "input_tensors.size() != 1. celu only support 1 input!" << endl;
    assert(0);
  }
  // find creator
  auto creator = get_plugin_creator(CELU_PLUGIN_NAME, CELU_PLUGIN_VERSION);

  // create plugin
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("data_type", &type, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("alpha", &alpha, nvinfer1::PluginFieldType::kFLOAT32, 1);

  const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
  const auto plugin_obj = InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("CeluPlugin", &plugin_data));

  auto plugin = network->addPluginV2(&input_tensors[0], 1, *plugin_obj);
  if (plugin == nullptr) {
    gLogError << "Create Network: Fail to create [CeluPlugin] layer.";
    return {};
  }

  plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [CeluPlugin]")).c_str());

  return {plugin->getOutput(0)};
}

// [>******************************add_batch_norm_plugin***********************************<]
// ITensorVector add_batch_norm_plugin(INetworkDefinition* network, const ITensorVector& input_tensors,
// const nvinfer1::DataType type, const nvinfer1::Weights& weight, const nvinfer1::Weights& bias,
// const nvinfer1::Weights& mean, const nvinfer1::Weights& var, const float eps) {
// if (input_tensors.size() != 1) {
// gLogFatal << "input_tensors.size() != 1. batch_norm only support 1 input!" << endl;
// assert(0);
//}
//// find creator
// auto creator = get_plugin_creator(BATCH_NORM_PLUGIN_NAME, BATCH_NORM_PLUGIN_VERSION);

//// create plugin
// std::vector<nvinfer1::PluginField> field_data;
// field_data.emplace_back("data_type", &type, nvinfer1::PluginFieldType::kINT32, 1);
// field_data.emplace_back("weight", weight.values, nvinfer1::PluginFieldType::kFLOAT32,
// static_cast<int>(weight.count)); field_data.emplace_back("bias", bias.values, nvinfer1::PluginFieldType::kFLOAT32,
// static_cast<int>(bias.count)); field_data.emplace_back("mean", mean.values, nvinfer1::PluginFieldType::kFLOAT32,
// static_cast<int>(mean.count)); field_data.emplace_back("var", var.values, nvinfer1::PluginFieldType::kFLOAT32,
// static_cast<int>(var.count)); field_data.emplace_back("eps", &eps, nvinfer1::PluginFieldType::kFLOAT32, 1);

// const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
// const auto plugin_obj = InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin(
// "BatchNormPlugin", &plugin_data));

// auto plugin = network->addPluginV2(&input_tensors[0], 1, *plugin_obj);
// if (plugin == nullptr) {
// gLogError << "Create Network: Fail to create [BatchNormPlugin] layer.";
// return {};
//}

// plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [BatchNormPlugin]")).c_str());

// return {plugin->getOutput(0)};
//}

/*******************************add_group_norm_plugin************************************/
ITensorVector add_group_norm_plugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                    const nvinfer1::DataType type, const int num_groups,
                                    const nvinfer1::Weights& weight, const nvinfer1::Weights& bias, const float eps) {
  if (input_tensors.size() != 1) {
    gLogFatal << "input_tensors.size() != 1. group_norm only support 1 input!" << endl;
    assert(0);
  }
  // find creator
  auto creator = get_plugin_creator(GROUP_NORM_PLUGIN_NAME, GROUP_NORM_PLUGIN_VERSION);

  // create plugin
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("data_type", &type, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("num_groups", &num_groups, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("weight", weight.values, nvinfer1::PluginFieldType::kFLOAT32, static_cast<int>(weight.count));
  field_data.emplace_back("bias", bias.values, nvinfer1::PluginFieldType::kFLOAT32, static_cast<int>(bias.count));
  field_data.emplace_back("eps", &eps, nvinfer1::PluginFieldType::kFLOAT32, 1);

  const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
  const auto plugin_obj = InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("GroupNormPlugin", &plugin_data));

  auto plugin = network->addPluginV2(&input_tensors[0], 1, *plugin_obj);
  if (plugin == nullptr) {
    gLogError << "Create Network: Fail to create [GroupNormPlugin] layer.";
    return {};
  }

  plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [GroupNormPlugin]")).c_str());

  return {plugin->getOutput(0)};
}

#endif  // BUILD_LIBTORCH_PLUGINS

#ifdef BUILD_SELF_PLUGINS

/*******************************yourself plugin************************************/
/*******************************AddAttMaskedSoftmaxPlugin************************************/
ITensorVector AddAttMaskedSoftmaxPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                        const nvinfer1::DataType type, const float scale) {
  if (input_tensors.size() != 2) {
    gLogFatal << "input_tensors.size() != 2. MaskedSoftmax only support 2 input!" << endl;
    assert(0);
  }

  // find creator
  auto creator = get_plugin_creator(ATT_MASKED_SOFTMAX_PLUGIN_NAME, ATT_MASKED_SOFTMAX_PLUGIN_VERSION);

  // create plugin
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("data_type", &type, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("scale", &scale, nvinfer1::PluginFieldType::kFLOAT32, 1);

  const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
  const auto plugin_obj =
      InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("AttMaskedSoftmaxPlugin", &plugin_data));

  auto plugin = network->addPluginV2(&input_tensors[0], 2, *plugin_obj);
  if (plugin == nullptr) {
    gLogError << "Create Network: Fail to create [AttMaskedSoftmaxPlugin] layer.";
    return {};
  }

  plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [AttMaskedSoftmaxPlugin]")).c_str());

  return {plugin->getOutput(0)};
}

/*******************************add_att_stream_softmax_plugin************************************/
ITensorVector AddAttStreamSoftmaxPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                        const nvinfer1::DataType type, const float scale, const int cache_len) {
  if (input_tensors.size() != 3) {
    gLogFatal << "input_tensors.size() != 3. AttStreamSoftmaxPlugin only support 3 input!" << endl;
    assert(0);
  }

  // find creator
  auto creator = get_plugin_creator(ATT_STREAM_SOFTMAX_PLUGIN_NAME, ATT_STREAM_SOFTMAX_PLUGIN_VERSION);

  // create plugin
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("data_type", &type, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("scale", &scale, nvinfer1::PluginFieldType::kFLOAT32, 1);
  field_data.emplace_back("cache_len", &cache_len, nvinfer1::PluginFieldType::kINT32, 1);

  const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
  const auto plugin_obj =
      InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("AttStreamSoftmaxPlugin", &plugin_data));

  auto plugin = network->addPluginV2(&input_tensors[0], 3, *plugin_obj);
  if (plugin == nullptr) {
    gLogError << "Create Network: Fail to create [AttStreamSoftmaxPlugin] layer.";
    return {};
  }

  plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [AttStreamSoftmaxPlugin]")).c_str());

  return {plugin->getOutput(0)};
}

/*******************************add_masked_fill_plugin************************************/
ITensorVector AddMaskedFillPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                  const nvinfer1::DataType type, const float fill) {
  if (input_tensors.size() != 2) {
    gLogFatal << "input_tensors.size() != 2" << endl;
    assert(0);
  }
  // find creator
  auto creator = get_plugin_creator(MASKED_FILL_PLUGIN_NAME, MASKED_FILL_PLUGIN_VERSION);

  // create plugin
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("data_type", &type, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("fill", &fill, nvinfer1::PluginFieldType::kFLOAT32, 1);

  const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
  const auto plugin_obj = InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("MaskedFillPlugin", &plugin_data));

  auto plugin = network->addPluginV2(&input_tensors[0], 2, *plugin_obj);
  if (plugin == nullptr) {
    gLogError << "Create Network: Fail to create [MaskedFillPlugin] layer.";
    return {};
  }

  plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [MaskedFillPlugin]")).c_str());

  return {plugin->getOutput(0)};
}

/*******************************AddCatSplitCachePlugin************************************/
ITensorVector AddCatSplitCachePlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                     const nvinfer1::DataType type, const int axis_dim) {
  if (input_tensors.size() != 2) {
    gLogFatal << "input_tensors.size() != 2. AddCatSplitCachePlugin only support 2 input!" << endl;
    assert(0);
  }

  // find creator
  auto creator = get_plugin_creator(CAT_SPLIT_CACHE_PLUGIN_NAME, CAT_SPLIT_CACHE_PLUGIN_VERSION);

  // create plugin
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("data_type", &type, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("axis_dim", &axis_dim, nvinfer1::PluginFieldType::kINT32, 1);

  const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
  const auto plugin_obj =
      InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("atSplitCachePlugin", &plugin_data));

  auto plugin = network->addPluginV2(&input_tensors[0], 2, *plugin_obj);
  if (plugin == nullptr) {
    gLogError << "Create Network: Fail to create [CatSplitCachePlugin] layer.";
    return {};
  }

  plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [CatSplitCachePlugin]")).c_str());

  return {plugin->getOutput(0), plugin->getOutput(1)};
}

/*******************************AddDumpTensorPlugin************************************/
ITensorVector AddDumpTensorPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                  const nvinfer1::DataType type) {
  // find creator
  auto creator = get_plugin_creator(DUMP_TENSOR_PLUGIN_NAME, DUMP_TENSOR_PLUGIN_VERSION);

  // create plugin
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("data_type", &type, nvinfer1::PluginFieldType::kINT32, 1);

  const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
  const auto plugin_obj = InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("DumpTensorPlugin", &plugin_data));

  auto plugin = network->addPluginV2(&input_tensors[0], 1, *plugin_obj);
  if (plugin == nullptr) {
    gLogError << "Create Network: Fail to create [DumpTensorPlugin] layer.";
    return {};
  }

  plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [DumpTensorPlugin]")).c_str());

  return {plugin->getOutput(0)};
}

/*******************************AddFMoEExpertPlugin************************************/
ITensorVector AddFMoEExpertPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                  const nvinfer1::DataType type, const int num_expert, const int idim,
                                  const int hidden_units, const int act_type) {
  // find creator
  auto creator = get_plugin_creator(FMOE_EXPERT_NAME, FMOE_EXPERT_VERSION);

  // create plugin
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("data_type", &type, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("num_expert", &num_expert, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("idim", &idim, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("hidden_units", &hidden_units, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("act_type", &act_type, nvinfer1::PluginFieldType::kINT32, 1);

  const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
  const auto plugin_obj =
      InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("FMoEDynamicPlugin", &plugin_data));

  auto plugin = network->addPluginV2(&input_tensors[0], 6, *plugin_obj);
  if (plugin == nullptr) {
    gLogError << "Create Network: Fail to create [FMoEDynamicPlugin] layer.";
    return {};
  }

  plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [FMoEDynamicPlugin]")).c_str());

  return {plugin->getOutput(0)};
}

/*******************************add_mask_conv2d_sample_plugin************************************/
ITensorVector AddMaskConv2dSamplePlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                        const int left_padding, const int stride) {
  if (input_tensors.size() != 1) {
    gLogFatal << "input_tensors.size() != 1. MaskConv2dSample only support 1 input!" << endl;
    assert(0);
  }

  // find creator
  auto creator = get_plugin_creator(MASK_CONV2D_SAMPLE_PLUGIN_NAME, MASK_CONV2D_SAMPLE_PLUGIN_VERSION);

  // create plugin
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("left_padding", &left_padding, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("stride", &stride, nvinfer1::PluginFieldType::kINT32, 1);

  const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
  const auto plugin_obj =
      InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("MaskConv2dSamplePlugin", &plugin_data));

  auto plugin = network->addPluginV2(&input_tensors[0], 1, *plugin_obj);
  if (plugin == nullptr) {
    gLogError << "Create Network: Fail to create [MaskConv2dSamplePlugin] layer.";
    return {};
  }

  plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [MaskConv2dSamplePlugin]")).c_str());

  return {plugin->getOutput(0)};
}

/*******************************AddRelPositionalEncodingPlugin************************************/
ITensorVector AddRelPositionalEncodingPlugin(INetworkDefinition* network, const ITensorVector& input_tensors,
                                             const nvinfer1::DataType type, const float scale, int max_len, int dim,
                                             const int streaming) {
  // find creator
  auto creator = get_plugin_creator(RELPOSITIONAL_ENCODING_PLUGIN_NAME, RELPOSITIONAL_ENCODING_PLUGIN_VERSION);

  // create plugin
  std::vector<nvinfer1::PluginField> field_data;
  field_data.emplace_back("data_type", &type, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("scale", &scale, nvinfer1::PluginFieldType::kFLOAT32, 1);
  field_data.emplace_back("max_len", &max_len, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("dim", &dim, nvinfer1::PluginFieldType::kINT32, 1);
  field_data.emplace_back("streaming", &streaming, nvinfer1::PluginFieldType::kINT32, 1);

  const PluginFieldCollection plugin_data{static_cast<int>(field_data.size()), field_data.data()};
  const auto plugin_obj =
      InferUniquePtr<nvinfer1::IPluginV2>(creator->createPlugin("RelPositionalEncoding", &plugin_data));
  int input_num = 2;
  if (streaming) input_num ++;

  auto plugin = network->addPluginV2(&input_tensors[0], input_num, *plugin_obj);
  if (plugin == nullptr) {
    gLogError << "Create Network: Fail to create [RelPositionalEncoding] layer.";
    return {};
  }

  plugin->setName((std::to_string(network->getNbLayers()) + std::string(" [RelPositionalEncoding]")).c_str());

  if (streaming) {
    return {plugin->getOutput(0), plugin->getOutput(1), plugin->getOutput(2)};
  } else {
    return {plugin->getOutput(0), plugin->getOutput(1)};
  }
}

#endif  // BUILD_SELF_PLUGINS

}  // namespace plugin
}  // namespace nvinfer1
