# Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import tensorrt as trt
import numpy as np
import ctypes
import math
import time

from typing import Optional, Tuple

from trt_helper.torch_network_helper import TorchNetworkHelper

class NetworkHelper(TorchNetworkHelper):
    """
    TensorRT Network Definition helper,
    this contains some functions that are related to TensorRT.
    For example addInput, addConstant and so on.
    """

    # def __init__(self, network, plugin_registry, config, logger):
        # self.network = network
        # self.plugin_registry = plugin_registry
        # self.config = config
        # self.logger = logger

        # self.input_num = 0

    # Advanced API
    def addActivation(self, layer, x, layer_name=None, precision=None):
        if isinstance(layer, torch.nn.SiLU):
            self.addSiLU(x, layer_name, precision);
        elif isinstance(layer, torch.nn.Sigmoid):
            self.addSigmoid(x, layer_name, precision);
        elif isinstance(layer, torch.nn.ReLU):
            self.addReLU(x, layer_name, precision);
        else:
            raise RuntimeError("activatio not support now! " + str(layer))

    #############Plugins#################
    def addAttMaskedSoftmaxPlugin(self, x: trt.ITensor, x_len: trt.ITensor, scale: float,
                                  layer_name: str = None, precision=None) -> trt.ITensor:
        """plugin AttMaskedSoftmax"""
        plg_creator = self.plugin_registry.get_plugin_creator("AttMaskedSoftmaxPluginDynamic", "1", "")
        if not plg_creator:
            raise RuntimeError("Could not find AttMaskedSoftmaxPluginDynamic")

        if layer_name is None:
            layer_name = "AttMaskedSoftmaxPlugin"
        else:
            layer_name = "AttMaskedSoftmaxPlugin." + layer_name

        data_type = trt.PluginField("data_type", np.array([self.config.plugin_data_type], dtype=np.int32), trt.PluginFieldType.INT32)
        scale = trt.PluginField("scale", np.array([scale], dtype=np.float32), trt.PluginFieldType.FLOAT32)
        pfc = trt.PluginFieldCollection([data_type, scale])
        plugin = plg_creator.create_plugin(layer_name, pfc)
        if not plugin:
            raise RuntimeError("Could not create_plugin AttMaskedSoftmaxPluginDynamic")

        layer = self.network.add_plugin_v2([x, x_len], plugin)

        self.layer_post_process(layer, layer_name, precision)

        x = layer.get_output(0)
        return x

    def addCatSplitCache(self, cache: trt.ITensor, x: trt.ITensor, dim: int,
        layer_name: Optional[str] = None, precision=None):
        """plugin CatSplitCache"""
        plg_creator = self.plugin_registry.get_plugin_creator("CatSplitCachePluginDynamic", "1", "")
        if not plg_creator:
            raise RuntimeError("Could not find CatSplitCachePluginDynamic")

        data_type = trt.PluginField("data_type", np.array([self.config.plugin_data_type], dtype=np.int32), trt.PluginFieldType.INT32)
        dim = trt.PluginField("axis_dim", np.array([dim], dtype=np.int32), trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection([data_type, dim])
        plugin = plg_creator.create_plugin("CatSplitCachePluginDynamic", pfc)
        if not plugin:
            raise RuntimeError("Could not create_plugin CatSplitCachePluginDynamic")

        layer = self.network.add_plugin_v2([cache, x], plugin)

        if layer_name is None:
            layer_name = "CatSplitCachePlugin"
        else:
            layer_name = "CatSplitCachePlugin." + layer_name

        self.layer_post_process(layer, layer_name, precision)

        x = layer.get_output(0)
        out_cache = layer.get_output(1)
        return [x, out_cache]

    def addDumpTensor(self, x: trt.ITensor, layer_name: Optional[str] = None):
        """DumpTensorPlugin"""
        plg_creator = self.plugin_registry.get_plugin_creator("DumpTensorPluginDynamic", "1", "")
        if not plg_creator:
            raise RuntimeError("Could not find DumpTensorPluginDynamic")

        if layer_name is None:
            layer_name = "DumpTensorPlugin"
        else:
            layer_name = "DumpTensorPlugin." + layer_name

        # data_type = trt.PluginField("data_type", np.array([data_type], dtype=np.int32), trt.PluginFieldType.INT32)
        # pfc = trt.PluginFieldCollection([data_type])
        pfc = trt.PluginFieldCollection([])
        plugin = plg_creator.create_plugin(layer_name, pfc)
        if not plugin:
            raise RuntimeError("Could not create_plugin DumpTensorPluginDynamic")

        layer = self.network.add_plugin_v2([x], plugin)

        self.layer_post_process(layer, layer_name, None)

        x = layer.get_output(0)
        return x

    # def addLeftPaddingCache(self, x: trt.ITensor, cache: trt.ITensor, dim: int,
        # layer_name: Optional[str] = None, precision=None):
        # """plugin addLeftPaddingCache"""
        # plg_creator = self.plugin_registry.get_plugin_creator("LeftPaddingCachePluginDynamic", "1", "")
        # if not plg_creator:
            # raise RuntimeError("Could not find LeftPaddingCachePluginDynamic")

        # data_type = trt.PluginField("data_type", np.array([self.config.plugin_data_type], dtype=np.int32), trt.PluginFieldType.INT32)
        # dim = trt.PluginField("axis_dim", np.array([dim], dtype=np.int32), trt.PluginFieldType.INT32)
        # pfc = trt.PluginFieldCollection([data_type, dim])
        # plugin = plg_creator.create_plugin("LeftPaddingCachePluginDynamic", pfc)
        # if not plugin:
            # raise RuntimeError("Could not create_plugin LeftPaddingCachePluginDynamic")

        # layer = self.network.add_plugin_v2([x, cache], plugin)

        # if layer_name is None:
            # layer_name = "plugin.LeftPaddingCachePlugin"

        # self.layer_post_process(layer, layer_name, precision)

        # x = layer.get_output(0)
        # out_cache = layer.get_output(1)
        # return [x, out_cache]

