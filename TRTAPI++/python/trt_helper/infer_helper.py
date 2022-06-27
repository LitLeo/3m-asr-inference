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

def trt_dtype_to_np_dtype(trt_dtype):
    if trt_dtype == trt.DataType.FLOAT:
      return np.float32
    if trt_dtype == trt.DataType.INT32:
      return np.int32
    if trt_dtype == trt.DataType.HALF:
      return np.float16
    if trt_dtype == trt.DataType.INT8:
      return np.int8
    if trt_dtype == trt.DataType.BOOL:
      return np.bool_

class InferHelper():
    """"""
    def __init__(self, plan_name, trt_logger):
        """"""
        self.logger = trt_logger
        self.runtime = trt.Runtime(trt_logger)
        self.plan_name = plan_name
        with open(plan_name, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.context.active_optimization_profile = 0

    def infer(self, inputs: list, base_outputs: list):
        """"""
        # set trt bindings(inputs and outputs)
        bindings = []
        trt_inputs = []
        for i in inputs:
            trt_i = i.cuda()
            trt_inputs.append(trt_i)
            bindings.append(trt_i.data_ptr())

        trt_outputs = []
        for o in base_outputs:
            trt_o = torch.zeros_like(o).contiguous().cuda()
            trt_outputs.append(trt_o)
            bindings.append(trt_o.data_ptr())

        # context.set_binding_shape
        for i in range(0, len(inputs)):
            self.context.set_binding_shape(i, tuple(inputs[i].shape))

        for i in range(len(inputs), self.engine.num_bindings):
            trt_output_shape = self.context.get_binding_shape(i)
            output_idx = i - len(inputs)
            if not (list(trt_output_shape) == list(base_outputs[output_idx].shape)):
                self.logger.log(trt.Logger.ERROR, "[Infer] output shape is error!")
                self.logger.log(trt.Logger.ERROR, "trt_output.shape = " + str(trt_output_shape))
                self.logger.log(trt.Logger.ERROR, "base_output.shape = " + str(base_outputs[output_idx].shape))
                assert(0)

        self.context.execute_v2(bindings)

        for i in range(0, len(base_outputs)):
            trt_outputs[i] = trt_outputs[i].cpu()

            base_output = base_outputs[i]
            trt_output = trt_outputs[i]

            print("base_output.shape:" + str(base_output.shape))
            print("base_output.sum:" + str(base_output.sum()))
            print(base_output.view(-1)[0:10])

            print("trt_output.shape:" + str(trt_output.shape))
            print("trt_output.sum:" + str(trt_output.sum()))
            print(trt_output.view(-1)[0:10])
            print("torch.allclose result:" + str(torch.allclose(base_output, trt_output, 1e-05, 1e-03)))
            print("====================")

        T1 = time.perf_counter()
        for i in range(0, 10):
          self.context.execute_v2(bindings)
        T2 =time.perf_counter()
        print("time=" + str((T2-T1) * 1000 / 10) + "ms")

        # return trt_outputs
        return torch.allclose(base_output, trt_output, 1e-05, 1e-03)

    def infer1(self, inputs: list, dump_output = False):
        """"""
        # set input binding_shape and get output binding_shape
        for i in range(0, len(inputs)):
            self.context.set_binding_shape(i, tuple(inputs[i].shape))

        output_names = []
        output_dtypes = []
        print("===============infer1 info===================")
        print("plan_name: " + self.plan_name)
        for i in range(0, self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.context.get_binding_shape(i)

            if self.engine.binding_is_input(i):
                print(f"input[{i}] name={name}, shape={shape}")
            else:
                print(f"output[{i}] name={name}, shape={shape}")
                output_names.append(name)
                output_dtypes.append(self.engine.get_binding_dtype(i))
        print("=============================================")

        outputs = []
        for i in range(len(inputs), self.engine.num_bindings):
            trt_dtype = self.engine.get_binding_dtype(i)
            np_dtype = trt_dtype_to_np_dtype(trt_dtype)
            o = np.zeros(self.context.get_binding_shape(i), dtype=np_dtype)
            o = torch.from_numpy(o).cuda()
            outputs.append(o)

        # set trt bindings(inputs and outputs)
        bindings = []
        trt_inputs = []
        for i in inputs:
            trt_i = i.cuda()
            trt_inputs.append(trt_i)
            bindings.append(trt_i.data_ptr())

        for o in outputs:
          bindings.append(o.data_ptr())

        T1 = time.perf_counter()
        self.context.execute_v2(bindings)
        T2 =time.perf_counter()
        print("time=" + str((T2-T1) * 1000) + "ms")

        if dump_output:
            for i in range(len(outputs)):
                o = outputs[i]
                name = output_names[i]
                print(f"output name={name}, dtype={str(output_dtypes[i])}, shape={str(o.shape)}")
                # print("output.shape:" + str(o.shape))
                print("output.sum:" + str(o.sum()))
                print(o)

        # return trt_outputs
        return outputs
