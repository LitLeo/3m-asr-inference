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

import tensorrt as trt
import numpy as np
import ctypes
import math
import time

from typing import Optional, Tuple

class BaseNetworkHelper():
    """TensorRT Network Definition helper, include base funcs"""
    def __init__(self, network, plugin_registry, config, logger):
        self.network = network
        self.plugin_registry = plugin_registry
        self.config = config
        self.logger = logger

        self.input_num = 0

    def set_layer_name(self, layer, name):
        """
        Tool function. Set the name of trt layer or plugin and print output shapes.
        """
        if not layer:
            raise RuntimeError("Could not name")

        layer.name = str(self.network.num_layers) + "_" + name
        for i in range(0, layer.num_outputs):
            tensor = layer.get_output(i)
            tensor.name = layer.name + str(i)
            shape = tensor.shape
            self.logger.log(trt.Logger.INFO, "[Network] " + layer.name + ", output[" + str(i) + "] shape= " + str(shape))

        return None

    def check_trt_layer(self, trt_layer):
        """
        Tool function. check trt layer,
        """
        if not trt_layer:
            raise RuntimeError("add " + str(trt_layer) + " failed!")

        for i in range(0, trt_layer.num_outputs):
            shape = trt_layer.get_output(i).shape
            # print(trt.volume(shape))

            # if len(shape) is 1:
                # raise RuntimeError("add " + layer.name + " failed!")

    def layer_post_process(self, trt_layer, layer_name, precision):
        """
        Tool function. set precision, set_layer_name and check_trt_layer
        """
        if precision is not None:
            trt_layer.precision = precision

        self.set_layer_name(trt_layer, layer_name)
        self.check_trt_layer(trt_layer)

    def broadcast_matrix(self, mat: np.array, nb_dims: int):
        mat_nb_dims = len(mat.shape)
        if mat_nb_dims >= nb_dims:
            raise RuntimeError("broadcast_tensor mat_nb_dims >= nb_dims")

        new_shape = np.ones([nb_dims], dtype=np.int)
        new_shape[-mat_nb_dims:] = mat.shape

        new_mat = mat.reshape(new_shape)
        self.logger.log(trt.Logger.INFO, "[Network] broadcast_matrix " + \
                                          str(mat.shape) + " to " + str(new_mat.shape))

        return new_mat

    def broadcast_tensor(self, x: trt.ITensor, nb_dims: int):
        x_nb_dims = len(x.shape)
        if x_nb_dims >= nb_dims:
            raise RuntimeError("broadcast_tensor x_nb_dims >= nb_dims")

        new_shape = np.ones([nb_dims], dtype=np.int)
        new_shape[-x_nb_dims:] = x.shape

        trt_layer = self.network.add_shuffle(x)
        trt_layer.reshape_dims = new_shape

        layer_name = "util.broadcast_tensor." + x.name

        self.layer_post_process(trt_layer, layer_name, None)

        x = trt_layer.get_output(0)
        return x

    def broadcast_2tensors(self, t1: trt.ITensor, t2: trt.ITensor):
        t1_nb_dim = len(t1.shape)
        t2_nb_dim = len(t2.shape)

        if t1_nb_dim > t2_nb_dim:
            t2 = self.broadcast_tensor(t2, t1_nb_dim)

        if t2_nb_dim > t1_nb_dim:
            t1 = self.broadcast_tensor(t1, t2_nb_dim)

        return t1, t2


