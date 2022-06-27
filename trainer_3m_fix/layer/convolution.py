#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
"""ConvolutionModule definition."""

from typing import Optional, Tuple

import torch
from torch import nn
import numpy as np

import tensorrt as trt
import trt_helper


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""
    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(
        self,
        network_helper,
        x: trt.ITensor,
        x_len: trt.ITensor,
        cache: Optional[torch.Tensor] = None,
    ) -> trt.ITensor:

        def add_masked_fill_plugin(network_helper, x, xs_len, fill):
            plg_creator = network_helper.plugin_registry.get_plugin_creator("MaskedFillPluginDynamic", "1", "")
            if not plg_creator:
                raise RuntimeError("Could not find MaskedFillPluginDynamic")

            data_type = trt.PluginField("data_type", np.array([network_helper.config.plugin_data_type], dtype=np.int32), trt.PluginFieldType.INT32)
            fill = trt.PluginField("fill", np.array([fill], dtype=np.float32), trt.PluginFieldType.FLOAT32)
            pfc = trt.PluginFieldCollection([data_type, fill])
            plugin = plg_creator.create_plugin("MaskedFillPluginDynamic", pfc)
            if not plugin:
                raise RuntimeError("Could not create_plugin MaskedFillPluginDynamic")

            layer = network_helper.network.add_plugin_v2([x, xs_len], plugin)
            network_helper.set_layer_name(layer, "MaskedFillPluginDynamic")
            x = layer.get_output(0)
            return x

        # exchange the temporal dimension and the feature dimension
        # x = x.transpose(1, 2)  # (#batch, channels, time)
        x = network_helper.addShuffle(x, (0, 2, 1), None, None, "conv_trans")

        # mask batch padding
        if x_len is not None:
            x = add_masked_fill_plugin(network_helper, x, x_len, 0.0)
            # x.masked_fill_(~mask_pad, 0.0)

        # reshape x from 3d to 4d for trt conv
        # trt conv require a 4-dimensional input tensor
        x = network_helper.addShuffle(x, None, (0, 0, 1, -1), None, "conv_trans_3d_to_4d")

        if self.lorder > 0:
            # x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            layer = network_helper.network.add_padding(x, pre_padding=(0, self.lorder), post_padding=(0, 0))
            network_helper.set_layer_name(layer, "conv_pad")
            x = layer.get_output(0)

        # GLU mechanism
        # x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = network_helper.addConv1d(self.pointwise_conv1, x)

        # x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)
        x = network_helper.addGLU(x, 1)
        # assert(0)

        # # 1D Depthwise Conv
        # x = self.depthwise_conv(x)
        # if self.use_layer_norm:
            # x = x.transpose(1, 2)
        x = network_helper.addConv1d(self.depthwise_conv, x)
        if self.use_layer_norm:
            # x = x.transpose(1, 2)
            x = network_helper.addShuffle(x, (0, 3, 2, 1), None, None, "use_layer_norm_trans")

        # x = self.activation(self.norm(x))
        x = network_helper.addLayerNorm(self.norm, x)

        # # swish: x* sigmoid(x)
        x = network_helper.addSiLU(x)

        if self.use_layer_norm:
            # x = x.transpose(1, 2)
            x = network_helper.addShuffle(x, (0, 3, 2, 1), None, None, "use_layer_norm_trans")

        # x = self.pointwise_conv2(x)
        x = network_helper.addConv1d(self.pointwise_conv2, x)

        x = network_helper.addShuffle(x, None, (0, 0, -1), None, "conv_trans_4d_to_3d")

        # mask batch padding
        if x_len is not None:
            # x.masked_fill_(~mask_pad, 0.0)
            x = add_masked_fill_plugin(network_helper, x, x_len, 0.0)

        # x.transpose(1, 2)
        x = network_helper.addShuffle(x, (0, 2, 1), None, None, "use_layer_norm_trans")

        return x
