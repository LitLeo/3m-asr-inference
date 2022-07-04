// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef PLUGIN_DEBUG_H_
#define PLUGIN_DEBUG_H_
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "NvInfer.h"
#include "cuda_fp16.h"

void print_data(const float* ptr, int len, std::string message);
void print_data(const int* ptr, int len, std::string message);
void print_data(const half* ptr, int len, std::string message);

#ifdef BUILD_LIBTORCH_PLUGINS
#include "torch/script.h"
#include "torch/torch.h"

void print_tensor(torch::Tensor tensor, std::string message, bool print_value);
void print_inttensor(torch::Tensor tensor, std::string message, bool print_value);

void print_tensor(const float *ptr, nvinfer1::Dims dims, std::string message, bool print_value);
void print_tensor(const half *ptr, nvinfer1::Dims dims, std::string message, bool print_value);
#endif  // BUILD_LIBTORCH_PLUGINS

#endif  // PLUGIN_DEBUG_H_
