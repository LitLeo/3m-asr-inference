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

#ifndef PLUGIN_COMMON_CUBLAS_COMMON_H_
#define PLUGIN_COMMON_CUBLAS_COMMON_H_

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "common.h"

namespace nvinfer1 {
namespace plugin {

cublasStatus_t cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k,
                          const float alpha, const float* A, const float* B,
                          const float beta, float* C);

cublasStatus_t cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k,
                          const half alpha, const half* A, const half* B,
                          const half beta, half* C);

cublasStatus_t cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k,
                          const float alpha, const char* A, const char* B,
                          const float beta, int* C);

cublasStatus_t cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k,
                          const float alpha, const float* A, int lda, const float* B, int ldb,
                          const float beta, float* C, int ldc);

cublasStatus_t cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k,
                          const half alpha, const half* A, int lda, const half* B, int ldb,
                          const half beta, half* C, int ldc);

cublasStatus_t cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k,
                          const float alpha, const char* A, int lda, const char* B, int ldb,
                          const float beta, int* C, int ldc);


}  // namespace plugin
}  // namespace nvinfer1

#endif  // PLUGIN_COMMON_CUBLAS_COMMON_H_
