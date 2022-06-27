#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
"""Encoder self-attention layer definition."""

from typing import Optional, Tuple

import torch
from torch import nn


class FmoeConformerLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size,
                                          eps=1e-12)  # for the CNN module
            self.norm_final = nn.LayerNorm(
                size, eps=1e-12)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear = nn.Linear(size + size, size)

    def forward(
        self,
        network_helper,
        x, embed, x_len, pos_emb
    ) :
        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                # x = self.norm_ff_macaron(x)
                x = network_helper.addLayerNorm(self.norm_ff_macaron, x)

            # x = residual + self.ff_scale * self.dropout(
                # self.feed_forward_macaron(x))
            x = self.feed_forward_macaron(network_helper, x)

            # x = residual + self.ff_scale * self.dropout(
            x = network_helper.addScale(x, self.ff_scale)

            x = network_helper.addAdd(x, residual)

            if not self.normalize_before:
                # x = self.norm_ff_macaron(x)
                x = network_helper.addLayerNorm(self.norm_ff_macaron, x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            # x = self.norm_mha(x)
            x = network_helper.addLayerNorm(self.norm_mha, x)

        # return x, x_len
        # x = network_helper.addDumpTensor(x, "before att")
        x_att = self.self_attn(network_helper, x, x, x, x_len, pos_emb)

        # x_att = network_helper.addDumpTensor(x_att, "after att")
        # return x_att, x_len
        if self.concat_after:
            assert(0)
            # x_concat = torch.cat((x, x_att), dim=-1)
            # x = residual + self.concat_linear(x_concat)
        else:
            # x = residual + self.dropout(x_att)
            x = network_helper.addAdd(residual, x_att)
            # set_layer_name(network, att_residual_layer, "att_residual_layer")
            # x = att_residual_layer.get_output(0)
        if not self.normalize_before:
            # x = self.norm_mha(x)
            x = network_helper.addLayerNorm(self.norm_mha, x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        # new_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                # x = self.norm_conv(x)
                x = network_helper.addLayerNorm(self.norm_conv, x)

            x = self.conv_module(network_helper, x, x_len)

            # x = network_helper.addDumpTensor(x, "after conv")

            # x = residual + self.dropout(x)
            x = network_helper.addAdd(residual, x, "conv_residual_layer")
            # set_layer_name(network, conv_residual_layer, "conv_residual_layer")
            # x = conv_residual_layer.get_output(0)

            if not self.normalize_before:
                # x = self.norm_conv(x)
                x = network_helper.addLayerNorm(self.norm_conv, x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = network_helper.addLayerNorm(self.norm_ff, x)
            # x = self.norm_ff(x)

        # x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        # network_helper.markOutput(x)
        x = self.feed_forward(network_helper, x, embed)

        x = network_helper.addScale(x, self.ff_scale)

        # x = residual + self.dropout(x)
        x = network_helper.addAdd(residual, x, "residual_layer")

        if not self.normalize_before:
            # x = self.norm_ff(x)
            x = network_helper.addLayerNorm(self.norm_ff, x)

        if self.conv_module is not None:
            # x = self.norm_final(x)
            x = network_helper.addLayerNorm(self.norm_final, x)

        # if output_cache is not None:
            # x = torch.cat([output_cache, x], dim=1)
        return x, x_len
