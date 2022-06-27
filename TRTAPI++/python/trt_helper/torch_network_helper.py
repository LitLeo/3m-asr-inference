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
import random
import ctypes
import math
import time

from typing import Optional, Tuple

from trt_helper.tensor_network_helper import TensorNetworkHelper

class TorchNetworkHelper(TensorNetworkHelper):
    """TensorRT Network Definition helper for Pytorch"""

    # def __init__(self, network, plugin_registry, config, logger):
        # self.network = network
        # self.plugin_registry = plugin_registry
        # self.config = config
        # self.logger = logger

        # self.input_num = 0

    def addAdaptiveAvgPool1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.AdaptiveAvgPool1d not support!")

        # if precision is not None:
           # trt_layer.precision = precision

        # if layer_name is None:
            # layer_name = "nn.AdaptiveAvgPool1d"
        # set_layer_name(network, trt_layer, layer_name)

        # x = trt_layer.get_output(0)

        # return [x]

    def addAdaptiveAvgPool2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.AdaptiveAvgPool2d not support!")

    def addAdaptiveAvgPool3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.AdaptiveAvgPool3d not support!")

    def addAdaptiveMaxPool1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.AdaptiveMaxPool1d not support!")

    def addAdaptiveMaxPool2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.AdaptiveMaxPool2d not support!")

    def addAdaptiveMaxPool3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.AdaptiveMaxPool3d not support!")

    def addAlphaDropout(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.AlphaDropout not support!")

    def addAvgPool1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.AvgPool1d not support!")

    def addAvgPool2d(self, layer, x, layer_name=None, precision=None):
        """AvgPool2d"""
        if layer_name is None:
            layer_name = "nn.AvgPool2d"

        # TODO(spikechu): support 3d input
        # TODO(spikechu): support padding, ceil_mode and divisor_override

        # torch layer.kernel_size like: kernel_size=3 or kernel_size=(1,3)
        if isinstance(layer.kernel_size, int):
            trt_layer = self.network.add_pooling(
                x, type=trt.PoolingType.AVERAGE,
                window_size=trt.DimsHW(layer.kernel_size, layer.kernel_size))
        else:
            trt_layer = self.network.add_pooling(
                x, type=trt.PoolingType.AVERAGE,
                window_size=trt.DimsHW(layer.kernel_size[0], layer.kernel_size[1]))

        # torch layer.stride like: stride=1 or stride=(1,2)
        if isinstance(layer.stride, int):
            trt_layer.stride = trt.DimsHW(layer.stride, layer.stride)
        else:
            trt_layer.stride = trt.DimsHW(layer.stride[0], layer.stride[1])

        trt_layer.average_count_excludes_padding = layer.count_include_pad

        # torch layer.padding like: padding=1 or padding=(1,2)
        if isinstance(layer.padding, int):
            trt_layer.padding = trt.DimsHW(layer.padding, layer.padding)
        else:
            trt_layer.padding = trt.DimsHW(layer.padding[0], layer.padding[1])

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addAvgPool3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.AvgPool3d not support!")

    def addBatchNorm1d(self, layer, x, layer_name=None, precision=None):
        """BatchNorm1d"""
        if layer_name is None:
            layer_name = "nn.BatchNorm1d"

        if len(x.shape) != 3:
            raise RuntimeError("addBatchNorm1d x.shape.size must = 3")

        if layer.affine:
            weight = layer.weight.detach().numpy()
            bias = layer.bias.detach().numpy()
        else:
            weight = torch.ones(x.shape[1]).detach().numpy()
            bias = torch.zeros(x.shape[1]).detach().numpy()
        mean = layer.running_mean.detach().numpy()
        var = layer.running_var.detach().numpy()
        eps = layer.eps
        var = np.sqrt(var + eps)

        scale = weight / var
        shift = bias - mean / var * weight

        # shuffle 1d input to 2d
        shuffle_layer_1 = self.network.add_shuffle(x)
        shuffle_layer_1.reshape_dims = (x.shape[0], x.shape[1], x.shape[2], 1)
        self.layer_post_process(shuffle_layer_1, layer_name+".shuffle_to_2d", precision)

        # compute batchnorm
        trt_layer = self.network.add_scale(shuffle_layer_1.get_output(0), mode=trt.ScaleMode.CHANNEL, shift=shift, scale=scale)
        self.layer_post_process(trt_layer, layer_name+".scale", precision)

        # shuffle 2d output to 1d
        shuffle_layer_2 = self.network.add_shuffle(trt_layer.get_output(0))
        shuffle_layer_2.reshape_dims = (x.shape[0], x.shape[1], x.shape[2])
        self.layer_post_process(shuffle_layer_2, layer_name+".shuffle_to_1d", precision)

        return shuffle_layer_2.get_output(0)

    def addBatchNorm2d(self, layer, x, layer_name=None, precision=None):
        """BatchNorm2d"""
        if layer_name is None:
            layer_name = "nn.BatchNorm2d"

        if len(x.shape) != 4:
            raise RuntimeError("addBatchNorm2d x.shape.size must = 4")

        if layer.affine:
            weight = layer.weight.detach().numpy()
            bias = layer.bias.detach().numpy()
        else:
            weight = torch.ones(x.shape[1]).detach().numpy()
            bias = torch.zeros(x.shape[1]).detach().numpy()
        mean = layer.running_mean.detach().numpy()
        var = layer.running_var.detach().numpy()
        eps = layer.eps
        var = np.sqrt(var + eps)

        scale = weight / var
        shift = bias - mean / var * weight

        trt_layer = self.network.add_scale(x, mode=trt.ScaleMode.CHANNEL, shift=shift, scale=scale)
        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addBatchNorm3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.BatchNorm3d not support!")

    def addBilinear(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Bilinear not support!")

    def addCELU(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.CELU not support!")

    def addChannelShuffle(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.ChannelShuffle not support!")

    def addConstantPad1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.ConstantPad1d not support!")

    def addConstantPad2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.ConstantPad2d not support!")

    def addConstantPad3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.ConstantPad3d not support!")

    def addConv1d(self, layer, x, layer_name=None, precision=None):
        """Conv1d"""
        weight = trt.Weights(layer.weight.detach().numpy())
        bias = trt.Weights(layer.bias.detach().numpy()) if not layer.bias is None else None

        trt_layer = self.network.add_convolution_nd(
            x, num_output_maps=layer.out_channels,
            kernel_shape=(1, layer.kernel_size[0]),
            kernel=weight, bias=bias)

        trt_layer.stride = (1, layer.stride[0])
        trt_layer.padding = (0, layer.padding[0])
        trt_layer.dilation = (1, layer.dilation[0])
        trt_layer.num_groups = layer.groups

        # TODO(leowgyang): conv padding_mode(zeros, reflect, replicate, circular)
        if layer.padding_mode is not "zeros":
            raise RuntimeError("padding_mode only support zeros now! padding_mode=" + str(layer.padding_mode))

        if layer_name is None:
            layer_name = "nn.Conv1d"
        else:
            layer_name = "nn.Conv1d." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addConv2d(self, layer, x, layer_name=None, precision=None):
        """Conv2d"""
        weight = trt.Weights(layer.weight.detach().numpy())
        bias = trt.Weights(layer.bias.detach().numpy()) if not layer.bias is None else None

        trt_layer = self.network.add_convolution_nd(
            x, num_output_maps=layer.out_channels,
            kernel_shape=(layer.kernel_size[0], layer.kernel_size[1]),
            kernel=weight, bias=bias)

        trt_layer.stride = (layer.stride[0], layer.stride[1])
        trt_layer.padding = (layer.padding[0], layer.padding[1])
        trt_layer.dilation = (layer.dilation[0], layer.dilation[1])
        trt_layer.num_groups = layer.groups

        # TODO(spikechu): conv padding_mode(zeros, reflect, replicate, circular)
        if layer.padding_mode is not "zeros":
            raise RuntimeError("padding_mode only support zeros now! padding_mode=" + str(layer.padding_mode))

        if layer_name is None:
            layer_name = "nn.Conv2d"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addConv3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Conv3d not support!")

    def addConvTranspose1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.ConvTranspose1d not support!")

    def addConvTranspose2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.ConvTranspose2d not support!")

    def addConvTranspose3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.ConvTranspose3d not support!")

    def addCosineSimilarity(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.CosineSimilarity not support!")

    # def addDropout(self, layer, x, layer_name=None, precision=None):
        # # do nothing
        # return x

    # def addDropout2d(self, layer, x, layer_name=None, precision=None):
        # # do nothing
        # return x

    # def addDropout3d(self, layer, x, layer_name=None, precision=None):
        # # do nothing
        # return x

    def addELU(self, layer, x, layer_name=None, precision=None):
        """ELU"""
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.ELU)
        trt_layer.alpha = layer.alpha

        if layer_name is None:
            layer_name = "nn.ELU"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addEmbedding(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Embedding not support!")

    def addEmbeddingBag(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.EmbeddingBag not support!")

    # TODO(leowgyang): not test
    def addFlatten(self, layer, x, layer_name=None, precision=None):
        if len(x) != 1:
            raise RuntimeError("nn.Flatten only support 1 input")

        input_len = len(x[0].shape)

        start_dim = layer.start_dim
        end_dim = layer.end_dim
        if start_dim == 0 and (end_dim == -1 or end_dim == input_len-1):
            trt_layer = self.network.add_shuffle(x[0])
            trt_layer.reshape_dims = (1)
        elif start_dim == 1 and (end_dim == -1 or end_dim == input_len-1):
            trt_layer = self.network.add_shuffle(x[0])
            trt_layer.reshape_dims = (1, -1)
        else:
            raise RuntimeError("nn.Flatten not support!")

        if layer_name is None:
            layer_name = "nn.Flatten"

        # layer_name = layer_name + "start" + str(start_dim) + "_end" + str(end_dim)
        layer_name = layer_name + "_start{s}_end{e}".format(s=start_dim, e=end_dim)

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addFold(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Fold not support!")

    def addFractionalMaxPool2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.FractionalMaxPool2d not support!")

    def addFractionalMaxPool3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.FractionalMaxPool3d not support!")

    def addGELU(self, x, layer_name=None, precision=None):
        dims = []
        for i in range(0, len(x.shape)):
            dims.append(1)

        dims = tuple(dims)

        POW = self.network.add_constant(dims, trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
        MULTIPLY = self.network.add_constant(dims, trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
        SQRT = self.network.add_constant(dims, trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
        ONE = self.network.add_constant(dims, trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
        HALF = self.network.add_constant(dims, trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
        X_pow = self.network.add_elementwise(x, POW.get_output(0), trt.ElementWiseOperation.POW)
        X_pow_t = X_pow.get_output(0)
        X_mul = self.network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
        X_add = self.network.add_elementwise(x, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
        X_sqrt = self.network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
        X_sqrt_tensor = X_sqrt.get_output(0)
        X_tanh = self.network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
        X_tanh_tensor = X_tanh.get_output(0)
        X_one = self.network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
        CDF = self.network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
        gelu_layer = self.network.add_elementwise(CDF.get_output(0), x, trt.ElementWiseOperation.PROD)

        if layer_name is None:
            layer_name = "nn.GELU"
        else:
            layer_name = "nn.GELU." + layer_name

        self.layer_post_process(gelu_layer, layer_name, precision)

        return gelu_layer.get_output(0)

    def addGLU(self, x: trt.ITensor, axis_dim:int = -1,
               layer_name: Optional[str] = None, precision: trt.DataType = None):
        """torch.nn.GLU"""
        plg_creator = self.plugin_registry.get_plugin_creator("GluPluginDynamic", "1", "")
        if not plg_creator:
            raise RuntimeError("Could not find GluPluginDynamic")

        data_type = trt.PluginField("data_type", np.array([self.config.plugin_data_type], dtype=np.int32), trt.PluginFieldType.INT32)
        axis_dim = trt.PluginField("axis_dim", np.array([axis_dim], dtype=np.int32), trt.PluginFieldType.INT32)
        pfc = trt.PluginFieldCollection([data_type, axis_dim])
        plugin = plg_creator.create_plugin("GluPluginDynamic", pfc)
        if not plugin:
            raise RuntimeError("Could not create_plugin GluPluginDynamic")

        layer = self.network.add_plugin_v2([x], plugin)

        if layer_name is None:
            layer_name = "nn.GLU"

        self.layer_post_process(layer, layer_name, precision)

        x = layer.get_output(0)
        return x

    # plugin
    def addGroupNorm(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.GroupNorm not support!")

    def addGRU(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.GRU not support!")

    def addGRUCell(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.GRUCell not support!")

    def addHardshrink(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Hardshrink not support!")

    def addHardsigmoid(self, layer, x, layer_name=None, precision=None):
        """Hardsigmoid"""
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.HARD_SIGMOID)
        trt_layer.alpha = 1 / 6
        trt_layer.beta = 1 / 2

        if layer_name is None:
            layer_name = "nn.Hardsigmoid"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    # plugin
    def addHardswish(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Hardswish not support!")

    # plugin
    def addHardtanh(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Hardtanh not support!")

    def addIdentity(self, layer, x, layer_name=None, precision=None):
        """Identity"""
        if layer_name is None:
            layer_name = "nn.Identity"

        trt_layer = self.network.add_identity(x)
        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addInstanceNorm1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.InstanceNorm1d not support!")

    def addInstanceNorm2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.InstanceNorm2d not support!")

    def addInstanceNorm3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.InstanceNorm3d not support!")

    def addLayerNorm(self, layer, x, layer_name=None, precision=None):
        """LayerNorm"""
        plg_creator = self.plugin_registry.get_plugin_creator("LayerNormPluginDynamic", "1", "")
        if not plg_creator:
            raise RuntimeError("Could not find LayerNormPluginDynamic")

        # TODO(spikechu): 2d layer.weight && elementwise_affine=False

        dim = layer.weight.size(0)
        eps = layer.eps
        gamma = layer.weight
        beta = layer.bias
        data_type = trt.PluginField("data_type", np.array([self.config.plugin_data_type], dtype=np.int32), trt.PluginFieldType.INT32)
        dim = trt.PluginField("dim", np.array([dim], dtype=np.int32), trt.PluginFieldType.INT32)
        eps = trt.PluginField("eps", np.array([eps], dtype=np.float32), trt.PluginFieldType.FLOAT32)
        # gamma_w = trt.PluginField("gamma", gamma.detach().numpy(), trt.PluginFieldType.FLOAT32)
        # beta_w = trt.PluginField("beta", beta.detach().numpy(), trt.PluginFieldType.FLOAT32)
        pfc = trt.PluginFieldCollection([data_type, dim, eps])
        plugin = plg_creator.create_plugin("LayerNormPluginDynamic", pfc)
        if not plugin:
            raise RuntimeError("Could not create_plugin LayerNormPluginDynamic")

        gamma_w = self.addConstant(gamma)
        beta_w = self.addConstant(beta)
        trt_layer = self.network.add_plugin_v2([x, gamma_w, beta_w], plugin)

        if layer_name is None:
            layer_name = "nn.LayerNorm"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addLazyBatchNorm1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LazyBatchNorm1d not support!")

    def addLazyBatchNorm2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LazyBatchNorm2d not support!")

    def addLazyBatchNorm3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LazyBatchNorm3d not support!")

    def addLazyConv1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LazyConv1d not support!")

    def addLazyConv2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LazyConv2d not support!")

    def addLazyConv3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LazyConv3d not support!")

    def addLazyConvTranspose1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LazyConvTranspose1d not support!")

    def addLazyConvTranspose2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LazyConvTranspose2d not support!")

    def addLazyConvTranspose3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LazyConvTranspose3d not support!")

    def addLazyLinear(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LazyLinear not support!")

    def addLeakyReLU(self, layer, x, layer_name=None, precision=None):
        """LeakyReLU"""
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.LEAKY_RELU)
        trt_layer.alpha = layer.negative_slope

        if layer_name is None:
            layer_name = "nn.LeakyReLU"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    # def addLinear_(self, x, weight, bias=None, layer_name=None, precision=None):
        # """Linear"""
        # input_len = len(x.shape)

        # # TODO(spikechu): support input_len = 2
        # if input_len < 3:
            # raise RuntimeError("addLinear x.shape.size must >= 3")

        # if layer_name is None:
            # layer_name = "nn.Linear"

        # # calc pre_reshape_dims and after_reshape_dims
        # pre_reshape_dims = trt.Dims()
        # after_reshape_dims = trt.Dims()
        # if input_len == 3:
            # pre_reshape_dims = (0, 0, 0, 1, 1)
            # after_reshape_dims = (0, 0, 0)
        # elif input_len == 4:
            # pre_reshape_dims = (0, 0, 0, 0, 1, 1)
            # after_reshape_dims = (0, 0, 0, 0)
        # elif input_len == 5:
            # pre_reshape_dims = (0, 0, 0, 0, 0, 1, 1)
            # after_reshape_dims = (0, 0, 0, 0, 0)
        # else:
            # raise RuntimeError("addLinear x.shape.size > 5 not support!")

        # # add pre_reshape layer
        # trt_layer = self.network.add_shuffle(x)
        # trt_layer.reshape_dims = pre_reshape_dims

        # self.layer_post_process(trt_layer, layer_name+"_pre_reshape", precision)

        # x = trt_layer.get_output(0)

        # out_features = weight.shape[0]
        # # add Linear layer
        # weight = trt.Weights(weight)
        # bias = None
        # if bias is not None:
            # bias = trt.Weights(bias)

        # trt_layer = self.network.add_fully_connected(x, out_features, weight, bias)
        # self.layer_post_process(trt_layer, layer_name, precision)
        # x = trt_layer.get_output(0)

        # # add after_reshape layer
        # trt_layer = self.network.add_shuffle(x)
        # trt_layer.reshape_dims = after_reshape_dims
        # self.layer_post_process(trt_layer, layer_name+"_after_reshape", precision)
        # x = trt_layer.get_output(0)

        # return x

    def addLinear_(self, x, weight, bias=None, layer_name=None, precision=None):
        """Linear"""
        # If input B is a constant, we transpose at parse time if necessary,
        # because In some cases, A * Bt is much slower than A * B.
        weight = np.copy(weight.transpose(1, 0), order='C')
        weight = self.broadcast_matrix(weight, len(x.shape))

        weight_layer = self.network.add_constant(weight.shape, trt.Weights(weight))
        weight = weight_layer.get_output(0)
        trt_layer = self.network.add_matrix_multiply(x, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.NONE)
        x = trt_layer.get_output(0)

        if layer_name is None:
            layer_name = "Linear"
        else:
            layer_name = "Linear." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        if bias is not None:
            bias = self.broadcast_matrix(bias, len(x.shape))
            bias_layer = self.network.add_constant(bias.shape, trt.Weights(bias))
            bias = bias_layer.get_output(0)
            trt_layer = self.network.add_elementwise(x, bias, trt.ElementWiseOperation.SUM)
            x = trt_layer.get_output(0)

            if layer_name is None:
                layer_name = "Linear.bias"
            else:
                layer_name = "Linear.bias." + layer_name
            self.layer_post_process(trt_layer, layer_name, precision)

        return x


    def addLinear(self, layer, x, layer_name=None, precision=None):
        weight = layer.weight.detach().numpy()
        bias = None
        if layer.bias is not None:
            bias = layer.bias.detach().numpy()

        return self.addLinear_(x, weight, bias, layer_name, precision)

    def addLocalResponseNorm(self, layer, x, layer_name=None, precision=None):
        """LocalResponseNorm"""
        if layer_name is None:
            layer_name = "nn.LocalResponseNorm"

        # TODO(spikechu): support 3d input

        trt_layer = self.network.add_lrn(x, layer.size, layer.alpha, layer.beta, layer.k)

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addLogSigmoid(self, layer, x, layer_name=None, precision=None):
        """LogSigmoid"""
        if layer_name is None:
            layer_name = "nn.LogSigmoid"

        trt_layer = self.network.add_activation(x, type=trt.ActivationType.SIGMOID)
        self.layer_post_process(trt_layer, layer_name+".sigmoid", precision)

        trt_layer = self.network.add_unary(trt_layer.get_output(0), op=trt.UnaryOperation.LOG)
        self.layer_post_process(trt_layer, layer_name+".unary.log", precision)

        return trt_layer.get_output(0)

    def addLogSoftmax(self, layer, x, layer_name=None, precision=None):
        """LogSoftmax"""
        if layer_name is None:
            layer_name = "nn.LogSoftmax"

        trt_layer = self.network.add_softmax(x)
        trt_layer.axes = int(math.pow(2, layer.dim))
        self.layer_post_process(trt_layer, layer_name+".softmax", precision)

        trt_layer = self.network.add_unary(trt_layer.get_output(0), op=trt.UnaryOperation.LOG)
        self.layer_post_process(trt_layer, layer_name+".unary.log", precision)

        return trt_layer.get_output(0)

    def addLPPool1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LPPool1d not support!")

    def addLPPool2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LPPool2d not support!")

    def addLSTM(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LSTM not support!")

    def addLSTMCell(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.LSTMCell not support!")

    def addMaxPool1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.MaxPool1d not support!")

    def addMaxPool2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.MaxPool2d not support!")

    def addMaxPool3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.MaxPool3d not support!")

    def addMaxUnpool1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.MaxUnpool1d not support!")

    def addMaxUnpool2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.MaxUnpool2d not support!")

    def addMaxUnpool3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.MaxUnpool3d not support!")

    def addMish(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Mish not support!")

    def addMultiheadAttention(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.MultiheadAttention not support!")

    def addPairwiseDistance(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.PairwiseDistance not support!")

    def addPixelShuffle(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.PixelShuffle not support!")

    def addPixelUnshuffle(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.PixelUnshuffle not support!")

    def addPReLU(self, layer, x, layer_name=None, precision=None):
        """PReLU"""
        if layer_name is None:
            layer_name = "nn.PReLU"

        if len(x.shape) < 2:
            raise RuntimeError("addPReLU x.shape.size < 2 not support!")
        if len(x.shape) > 5:
            raise RuntimeError("addPReLU x.shape.size > 5 not support!")

        if len(x.shape) == 2:
            slope_shape = (1, layer.weight.shape[0])
        if len(x.shape) == 3:
            slope_shape = (1, layer.weight.shape[0], 1)
        if len(x.shape) == 4:
            slope_shape = (1, layer.weight.shape[0], 1, 1)
        if len(x.shape) == 5:
            slope_shape = (1, layer.weight.shape[0], 1, 1, 1)

        slope = trt.Weights(layer.weight.detach().numpy())

        constant_layer = self.network.add_constant(slope_shape, slope)
        self.layer_post_process(constant_layer, layer_name+".constant", precision)

        trt_layer = self.network.add_parametric_relu(x, constant_layer.get_output(0))
        self.layer_post_process(trt_layer, layer_name+".parametric_relu", precision)

        return trt_layer.get_output(0)

    def addReflectionPad1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.ReflectionPad1d not support!")

    def addReflectionPad2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.ReflectionPad2d not support!")

    def addReLU(self, x, layer_name=None, precision=None):
        """ReLU"""
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.RELU)

        if layer_name is None:
            layer_name = "nn.ReLU"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addReLU6(self, layer, x, layer_name=None, precision=None):
        """ReLU6"""
        if layer_name is None:
            layer_name = "nn.ReLU6"

        relu = self.network.add_activation(x, type=trt.ActivationType.RELU)
        self.layer_post_process(relu, layer_name+".activation.relu", precision)

        shape = (1, ) * len(x.shape)
        tensor = 6.0 * torch.ones(shape, dtype=torch.float32).cpu().numpy()

        trt_6 = self.network.add_constant(shape, tensor)
        self.layer_post_process(trt_6, layer_name+".constant", precision)

        relu_6 = self.network.add_elementwise(relu.get_output(0), trt_6.get_output(0), trt.ElementWiseOperation.MIN)
        self.layer_post_process(relu_6, layer_name+".elementwise.min", precision)

        return relu_6.get_output(0)

    def addReplicationPad1d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.ReplicationPad1d not support!")

    def addReplicationPad2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.ReplicationPad2d not support!")

    def addReplicationPad3d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.ReplicationPad3d not support!")

    def addRNN(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.RNN not support!")

    def addRNNBase(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.RNNBase not support!")

    def addRNNCell(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.RNNCell not support!")

    def addRReLU(self, layer, x, layer_name=None, precision=None):
        """RReLU"""
        if layer_name is None:
            layer_name = "nn.RReLU"

        if len(x.shape) < 2:
            raise RuntimeError("addPReLU x.shape.size < 2 not support!")
        if len(x.shape) > 5:
            raise RuntimeError("addPReLU x.shape.size > 5 not support!")

        lower = layer.lower
        upper = layer.upper

        rand = random.uniform(lower, upper)



    def addSELU(self, layer, x, layer_name=None, precision=None):
        """SELU"""
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.SELU)
        # alpha and beta are fixed value corresponding with pytorch
        trt_layer.alpha = 1.6732632423543772848170429916717
        trt_layer.beta = 1.0507009873554804934193349852946

        if layer_name is None:
            layer_name = "nn.SELU"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addSigmoid(self, layer, x, layer_name=None, precision=None):
        """Sigmoid"""
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.SIGMOID)

        if layer_name is None:
            layer_name = "nn.Sigmoid"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    # Swish ＝ x*sigmod(x)
    def addSiLU(self, x, layer_name=None, precision=None):
        """SiLU"""
        if layer_name is None:
            layer_name = "nn.SiLU"

        # Sigmoid
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.SIGMOID)
        self.layer_post_process(trt_layer, layer_name+".sigmoid", precision)
        sigmod_x = trt_layer.get_output(0)

        # prod
        trt_layer = self.network.add_elementwise(x, sigmod_x, trt.ElementWiseOperation.PROD)
        self.layer_post_process(trt_layer, layer_name+".elementwise.prod", precision)

        return trt_layer.get_output(0)

    def addSoftmax(self, x: trt.ITensor, dim: int = -1, layer_name=None, precision=None) -> trt.ITensor:
        trt_layer = self.network.add_softmax(x)

        # input_len = len(x.shape)
        # if dim is -1:
            # dim = input_len
        # trt_layer.axes = int(math.pow(2, input_len-1))

        input_len = len(x.shape)
        if dim < 0:
            dim = input_len + dim
        # trt_layer.axes = 1 << (dim - 1)
        trt_layer.axes = 1 << (dim)

        layer_name_prefix = "nn.Softmax[dim=" + str(dim) + "]"
        if layer_name is None:
            layer_name = layer_name_prefix
        else:
            layer_name = layer_name_prefix + "." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addSoftmax2d(self, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Softmax2d not support!")

    def addSoftmin(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Softmin not support!")

    def addSoftplus(self, layer, x, layer_name=None, precision=None):
        """Softplus"""
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.SOFTPLUS)
        trt_layer.alpha = 1 / layer.beta
        trt_layer.beta = layer.beta

        # TODO(spikechu): support threshold

        if layer_name is None:
            layer_name = "nn.Softplus"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addSoftshrink(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Softshrink not support!")

    def addSoftsign(self, layer, x, layer_name=None, precision=None):
        """Softsign"""
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.SOFTSIGN)

        if layer_name is None:
            layer_name = "nn.Softsign"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addSyncBatchNorm(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.SyncBatchNorm not support!")

    def addTanh(self, layer, x, layer_name=None, precision=None):
        """Tanh"""
        trt_layer = self.network.add_activation(x, type=trt.ActivationType.TANH)

        if layer_name is None:
            layer_name = "nn.Tanh"

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addTanhshrink(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Tanhshrink not support!")

    def addThreshold(self, layer, x, layer_name=None, precision=None):
        """Threshold"""
        if layer_name is None:
            layer_name = "nn.Threshold"

        # TODO(spikechu): support layer.value
        if layer.value != 0:
            raise RuntimeError("nn.Threshold only support value = 0 now!")

        trt_layer = self.network.add_activation(x, type=trt.ActivationType.THRESHOLDED_RELU)
        trt_layer.alpha = layer.threshold

        self.layer_post_process(trt_layer, layer_name, precision)

        return trt_layer.get_output(0)

    def addTransformer(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Transformer not support!")

    def addTransformerDecoder(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.TransformerDecoder not support!")

    def addTransformerDecoderLayer(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.TransformerDecoderLayer not support!")

    def addTransformerEncoder(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.TransformerEncoder not support!")

    def addTransformerEncoderLayer(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.TransformerEncoderLayer not support!")

    def addUnflatten(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Unflatten not support!")

    def addUnfold(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Unfold not support!")

    def addUpsample(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.Upsample not support!")

    def addUpsamplingBilinear2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.UpsamplingBilinear2d not support!")

    def addUpsamplingNearest2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.UpsamplingNearest2d not support!")

    def addZeroPad2d(self, layer, x, layer_name=None, precision=None):
        raise RuntimeError("nn.ZeroPad2d not support!")
