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

from trt_helper.base_network_helper import BaseNetworkHelper

class TrtNetworkHelper(BaseNetworkHelper):
    """TensorRT Network Definition helper, include base funcs"""

    # def __init__(self, network, plugin_registry, config, logger):
        # self.network = network
        # self.plugin_registry = plugin_registry
        # self.config = config
        # self.logger = logger

        # self.input_num = 0

    def addInput(self, name, dtype, shape):
        if name is None:
            name = "input" + str(self.input_num)

        self.input_num = self.input_num + 1

        trt_input = self.network.add_input(name=name, dtype=dtype, shape=shape)
        if not trt_input:
            raise RuntimeError("addInput failed!")

        self.logger.log(trt.Logger.INFO, "[Network] add input:" + name + ", shape=" + str(shape))

        return trt_input

    def markOutput(self, x: trt.ITensor):
        self.network.mark_output(x)
        self.logger.log(trt.Logger.INFO, "[Network] mark output:" + x.name + ", shape=" + str(x.shape))

    def addConstant_(self, x: np.array, layer_name: Optional[str] = None) -> trt.ITensor:
        trt_layer = self.network.add_constant(x.shape, x)

        if layer_name is None:
            layer_name = "trt.Constant"
        else:
            layer_name = "trt.Constant." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)
        x = trt_layer.get_output(0)
        return x

    def addConstant(self, x: torch.Tensor, layer_name: Optional[str] = None) -> trt.ITensor:
        x_ = x.detach().numpy()
        return self.addConstant_(x_, layer_name)

    def addShape(
        self,
        x: trt.ITensor,
        layer_name: str = None
    ) -> trt.ITensor:
        trt_layer = self.network.add_shape(x)

        if layer_name is None:
            layer_name = "trt.shape"
        else:
            layer_name = "trt.shape." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)
        return trt_layer.get_output(0)

    def addShuffle(
        self,
        x: trt.ITensor,
        first_transpose: trt.Permutation,
        reshape_dims: trt.Dims,
        second_transpose: trt.Permutation,
        layer_name: str = None
    ) -> trt.ITensor:
        """"""
        trt_layer = self.network.add_shuffle(x)
        if first_transpose is not None:
            trt_layer.first_transpose = first_transpose

        if reshape_dims is not None:
            trt_layer.reshape_dims = reshape_dims

        if second_transpose is not None:
            trt_layer.second_transpose = second_transpose

        if layer_name is None:
            layer_name = "trt.Shuffle"
        else:
            layer_name = "trt.Shuffle." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x

    def addTopK(
        self,
        x: trt.ITensor,
        op: trt.TopKOperation,
        k: int = 1,
        dim: int = -1,
        layer_name: str = None
    ) -> trt.ITensor:
        """"""
        input_len = len(x.shape)
        # NCHW tensor as input (three non-batch dimensions).
        # Bit 0 corresponds to the C dimension boolean.
        # Bit 1 corresponds to the H dimension boolean.
        # Bit 2 corresponds to the W dimension boolean.
        assert (dim == -1) or (dim > 0 and dim < input_len)

        if dim == -1:
            dim = input_len - 1

        axes = 1 << dim
        # print(dim)
        # print(axes)

        trt_layer = self.network.add_topk(x, op, k, axes)

        if layer_name is None:
            layer_name = "trt.TopK"
        else:
            layer_name = "trt.TopK." + layer_name

        self.layer_post_process(trt_layer, layer_name, None)

        values  = trt_layer.get_output(0)
        idxs = trt_layer.get_output(1)
        return [idxs, values]

