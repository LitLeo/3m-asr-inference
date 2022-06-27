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

import ctypes
from typing import Optional

import torch
import tensorrt as trt

from trt_helper.network_helper import NetworkHelper

def init_trt_plugin(severity=None, lib_name=None, logger=None):
    """
    TensorRT Initialization
    """
    if severity is None:
        severity = trt.Logger.INFO

    if logger is None:
        logger = trt.Logger(severity)

    if lib_name is None:
        lib_name = "libtrt_plugin_plus.so"

    handle = ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
    if not handle:
        raise RuntimeError("Could not load plugin library. Is " + lib_name + " on your LD_LIBRARY_PATH?")

    trt.init_libnvinfer_plugins(logger, "")

    logger.log(logger.INFO, "[TrtHelper LOG] tensorrt plugin init done!")

    return logger

class HelperConfig():
    """ TensorRT helper config

    """
    def __init__(self):
        self.use_fp16 = False
        self.use_int8 = False
        # 0: float, 1: half, 2: int8
        self.plugin_data_type = 0
        self.dynamic_shape = True
        self.max_workspace_size = 3 # 3G

    def log(self):
        print("=========TrtHelperConfig===========")
        print("use_fp16: " + str(self.use_fp16))
        print("use_int8: " + str(self.use_int8))
        print("plugin_data_type: " + str(self.plugin_data_type))
        print("dynamic_shape: " + str(self.dynamic_shape))
        print("max_workspace_size: " + str(self.max_workspace_size) + "G")
        print("=========TrtHelperConfig===========")

class BuilderHelper():
    """TensorRT builder helper for Pytorch"""
    def __init__(self, config, logger, calibrator=None):
        if not logger:
            self.logger = trt.Logger(trt.Logger.INFO)
        else:
            self.logger = logger

        self.config = config
        self.config.log()

        self.log_prefix = "[Builder] "

        # get_plugin_registry
        self.plugin_registry = trt.get_plugin_registry()
        if not self.plugin_registry:
            raise RuntimeError("get_plugin_registry failed! Please call init_trt_plugin first")
        self.logger.log(trt.Logger.INFO, self.log_prefix + "trt.get_plugin_registry() done")

        # create builder, network and builder_config
        self.builder = trt.Builder(self.logger)
        if not self.builder:
            raise RuntimeError("create trt.Builder failed!")

        # create static or dynamic shape flag
        if self.config.dynamic_shape:
            flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            self.logger.log(trt.Logger.INFO, self.log_prefix + "create_network in dynamic shape")
        else:
            raise RuntimeError("Not support static shape now!")
            self.logger.log(trt.Logger.INFO, self.log_prefix + "create_network in static shape")
        self.network = self.builder.create_network(flag)

        self.builder_config = self.builder.create_builder_config()
        if not self.builder_config:
            raise RuntimeError("create_builder_config failed!")

        self.builder_config.max_workspace_size = self.config.max_workspace_size * (1024 * 1024 * 1024)
        log_str = self.log_prefix + "max_workspace_size=" + str(self.config.max_workspace_size)
        self.logger.log(trt.Logger.INFO, log_str)

        # fp16
        if self.config.use_fp16:
            self.builder_config.set_flag(trt.BuilderFlag.FP16)
            self.logger.log(trt.Logger.INFO, self.log_prefix + "config use fp16")

        # int8
        if self.config.use_int8:
            if calibrator is None:
                raise RuntimeError("config.use_int8 is true, but calibrator is None!")
            self.builder_config.set_flag(trt.BuilderFlag.INT8)
            # calibrator = AsrCalibrator("np_inputs/np_feat.list",
            # "np_inputs/np_feat_len.list", "conformer.int8.cache", 10)
            self.builder_config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
            self.builder_config.int8_calibrator = calibrator
            self.logger.log(trt.Logger.INFO, self.log_prefix + "config use int8")

        self.profile = self.builder.create_optimization_profile()
        # self.max_profile_size = 8
        # self.profiles = []
        # for i in range(0, self.max_profile_size):
            # p = self.builder.create_optimization_profile()
            # self.profiles.append(p)

        self.network_helper = NetworkHelper(self.network, self.plugin_registry, self.config, self.logger)

    def get_network_helper(self):
        return self.network_helper

    def add_profile(self, name, min_shape, opt_shape, max_shape):
        self.profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)

        log = "add_profile, name=" + name + \
              ", min=" + str(min_shape) + ", opt=" + str(opt_shape) + ", max=" + str(max_shape)
        self.logger.log(trt.Logger.INFO, "[Builder] " + log)

        return None

    def build_engine(self, engine_name: Optional[str] = None):
        self.logger.log(trt.Logger.INFO, "=====================build engine start.=====================")

        self.builder_config.add_optimization_profile(self.profile)

        engine = self.builder.build_engine(self.network, self.builder_config)
        if not engine:
            raise RuntimeError("build_engine failed")

        if engine_name:
            self.logger.log(trt.Logger.INFO, "Serializing Engine...")
            serialized_engine = engine.serialize()
            if serialized_engine is None:
                raise RuntimeError("serialize failed")

            self.logger.log(trt.Logger.INFO, "Saving Engine to {:}".format(engine_name))
            with open(engine_name, "wb") as fout:
                fout.write(serialized_engine)

        self.logger.log(trt.Logger.INFO, "=====================build engine done.=====================")

        return engine

    def test_infer(engine_name, inputs, base_outputs):
        TRT_LOGGER.log(TRT_LOGGER.INFO, "=====================infer_trt Start.=====================")

        # infer
        with open(engine_name, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, \
            runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

            # select engine profile
            context.active_optimization_profile = 0

            batch = feat.size(0)
            seq_len = feat.size(1)
            input_dim = feat.size(2)

            feat = feat.cuda()
            feat_len = feat_len.cuda()

            output_score = torch.zeros_like(base_score).contiguous().cuda()
            output_len = feat_len.clone().zero_().cuda()

            context.set_binding_shape(0, (batch, seq_len, input_dim))
            context.set_binding_shape(1, (1, batch))

            bindings = [feat.data_ptr(), feat_len.data_ptr(),
                        output_score.data_ptr()]

            print(context.get_binding_shape(0))
            print(context.get_binding_shape(1))
            print(context.get_binding_shape(2))
            # print(context.get_binding_shape(3))
            context.execute_v2(bindings)

            output_score = output_score.cpu()
            output_len = output_len.cpu()

            print("base_score.shape:" + str(base_score.shape))
            print("base_score.sum:" + str(base_score.sum()))
            print(base_score)

            print("output_score.shape:" + str(output_score.shape))
            print("output_score.sum:" + str(output_score.sum()))
            print(output_score)
            print(output_len)
            print("torch.allclose result:" + str(torch.allclose(base_score, output_score)))

        TRT_LOGGER.log(TRT_LOGGER.INFO, "=====================infer_trt Done.=====================")
        return output_score, output_len
