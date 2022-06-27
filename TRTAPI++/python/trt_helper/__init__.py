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

import trt_helper.builder_helper
import trt_helper.infer_helper
import trt_helper.network_helper
import trt_helper.version

init_trt_plugin = trt_helper.builder_helper.init_trt_plugin
BuilderHelper = trt_helper.builder_helper.BuilderHelper
HelperConfig = trt_helper.builder_helper.HelperConfig
NetworkHelper = trt_helper.network_helper.NetworkHelper
InferHelper = trt_helper.infer_helper.InferHelper

__version__ = trt_helper.version.__version__
print("==============================================")
print("[trt_helper] debug info: __version__=" + __version__)
print("==============================================")
