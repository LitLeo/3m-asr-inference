#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import tensorrt as trt
import numpy as np
import ctypes
import math
import time
import sys

from typing import Optional, Tuple, List

from trt_helper.base_network_helper import BaseNetworkHelper

MAX_INT = sys.maxsize
MIN_INT = -sys.maxsize - 1

class TensorNetworkHelper(BaseNetworkHelper):
    """TensorRT Network Definition helper for Pytorch"""
    # def __init__(self, network, plugin_registry, config, logger):
        # super(NetworkHelper, self).__init__(network, plugin_registry, config, logger)

    def addExpand(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addExpandAs(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addIndex(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addNewEmpty(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addRepeat(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addReshape(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addSelect(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def _slice_calc_size(self, strt, end):
        """
        start  end      size            dynamic_size  support
          +     +   end - start             N            Y
          +     -   (end - start) + shape   Y            Y
          +   None  (-start) + shape        Y            Y
          -     +   end - (shape + start)   Y            N
          -     -   start - end             N            Y
          -   None  -start                  N            Y
        """
        size = 0
        if start >= 0 and end >= 0:
            size = end - start
        if start >= 0 and end < 0:
            size = end - start
        if start >= 0 and end is None:
            size = -start
        if start < 0 and end >= 0:
            raise RuntimeError("not support now!")
        if start < 0 and end < 0:
            size = start - end
        if start < 0 and end is None:
            size = -start
        return size


    def addSlice(self, x: trt.ITensor, params: [[]], layer_name=None, precision=None):
        """"""
        dynamic_start = False
        dynamic_size = False

        input_shape_len = len(x.shape)
        if input_shape_len == len(params):
          raise RuntimeError("The len of slice params is not match input'c shape!")

        start = np.zeros(input_shape_len, dtype=int)
        size = np.zeros(input_shape_len, dtype=int)
        stride = np.zeros(input_shape_len, dtype=int)
        # get start, end and stride
        for i in range(0, input_shape_len):
            p = params[i] if p is not None else [0, None, 1]

            if len(params[i]) != 3:
                raise RuntimeError("The slice params pairs must be [start, end, stride]!")

            start[i] = p[0] if p[0] is not None else 0
            end = p[1]
            stride[i] = p[2]

            if start[i] < 0:
                dynamic_start = True

            if start[i] >= 0 and (end is None or end < 0):
                dynamic_size = True

            size[i] = self._slice_calc_size(start[i], end)

        # create static slice layer
        slice_layer = self.network.add_slice(x, start, size, stride)

        if dynamic_start or dynamic_size:
            # calc start, size and stride
            shape_tensor = self.network.add_shape(x).get_output(0)
            if dynamic_start:
                start_tensor = self.network.add_constant(start.shape, start).get_output(0)
                start_tensor = self.network.add_elementwise(  \
                    shape_tensor, start_tensor, trt.ElementWiseOperation.SUM) \
                    .get_output(0)

                slice_layer.set_input(1, start_tensor)

            if dynamic_size:
                size_tensor = self.network.add_constant(size.shape, size).get_output(0)
                size_tensor = self.network.add_elementwise( \
                    shape_tensor, size_tensor, trt.ElementWiseOperation.SUM) \
                    .get_output(0)

                slice_layer.set_input(2, size_tensor)

        if layer_name is None:
            layer_name = "tensor.Slice"
        else:
            layer_name = "tensor.Slice." + layer_name

        self.layer_post_process(slice_layer, layer_name, None)
        x = slice_layer.get_output(0)
        return x


    def addView(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addAddmm(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addAmax(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addAmin(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addArange(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addArgmax(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addArgmin(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addCat(self, inputs: List[trt.ITensor], dim=0, layer_name=None, precision=None):
        trt_layer = self.network.add_concatenation(inputs)
        trt_layer.axis = dim

        if layer_name is None:
            layer_name = "torch.cat"
        else:
            layer_name = "torch.cat." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addChunk(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addClamp(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addClone(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addDequantize(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addEmpty(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addEmptyLike(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addFlatten(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addFlip(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addFull(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addFullLike(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addLogsumexp(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addMatmul(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addMean(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addNorm(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addNormal(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addOnes(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addOnesLike(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addPermute(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addProd(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    # def add_quantize_per_tensor(self, x: trt.ITensor, layer_name=None, precision=None):
        # raise RuntimeError("Not support now!")

    def addRandn(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addRandnLike(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addRoll(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addSplit(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addSqueeze(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addStack(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addSum(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addTranspose(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addUnbind(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addUnsqueeze(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addVar(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addZeros(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")

    def addZerosLike(self, x: trt.ITensor, layer_name=None, precision=None):
        raise RuntimeError("Not support now!")


    ################## unary op ###################
    def addLog(self, x: trt.ITensor, layer_name=None, precision=None):
        trt_layer = self.network.add_unary(x, trt.UnaryOperation.LOG)
        if layer_name is None:
            layer_name = "unary.log"
        else:
            layer_name = "unary.log." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    ################## unary op ###################
    def addLog(self, x: trt.ITensor, layer_name=None, precision=None):
        trt_layer = self.network.add_unary(x, trt.UnaryOperation.LOG)
        if layer_name is None:
            layer_name = "unary.log"
        else:
            layer_name = "unary.log." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    ################## elementwise op ###################
    def addAdd(self, a, b, layer_name=None, precision=None):
        trt_layer = self.network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)
        if layer_name is None:
            layer_name = "elementwise.sum"
        else:
            layer_name = "elementwise.sum." + layer_name

        self.layer_post_process(trt_layer, layer_name, precision)

        x = trt_layer.get_output(0)
        return x

    def addProd(
        self,
        a: trt.ITensor,
        b: trt.ITensor,
        layer_name: Optional[str] = None,
        precision: Optional[trt.DataType] = None
    ) -> trt.ITensor:
        """"""
        trt_layer = self.network.add_elementwise(a, b, trt.ElementWiseOperation.PROD)

        if layer_name is None:
            layer_name = "elementwise.prod"

        self.layer_post_process(trt_layer, layer_name, precision)

        x = layer.get_output(0)
        return x

    # tensor and scalar op
    def addScale(
            self,
            x: trt.ITensor,
            scale: float,
            layer_name: str = None,
            precision: trt.DataType = None
    ) -> trt.ITensor:
        """scale"""
        input_len = len(x.shape)
        if input_len < 3:
            raise RuntimeError("input_len < 3 not support now! ")

        if layer_name is None:
            layer_name = "Scale"

        # The input dimension must be greater than or equal to 4
        if input_len is 3:
            trt_layer = self.network.add_shuffle(x)
            trt_layer.reshape_dims = (0, 0, 0, 1)
            self.layer_post_process(trt_layer, layer_name+".3dto4d", precision)
            x = trt_layer.get_output(0)

        np_scale = trt.Weights(np.array([scale], dtype=np.float32))
        trt_layer = self.network.add_scale(x, mode=trt.ScaleMode.UNIFORM,
                                      shift=None, scale=np_scale, power=None)
        self.layer_post_process(trt_layer, layer_name, precision)
        x = trt_layer.get_output(0)

        if input_len is 3:
            trt_layer = self.network.add_shuffle(x)
            trt_layer.reshape_dims = (0, 0, 0)
            self.layer_post_process(trt_layer, layer_name+".4dto3d", precision)
            x = trt_layer.get_output(0)

        return x

