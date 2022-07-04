/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PLUGIN_COMMON_CHECKMACROSPLUGIN_H_
#define PLUGIN_COMMON_CHECKMACROSPLUGIN_H_

#include <sstream>
#include "NvInfer.h"

#include "NvInferVersion.h"
#include "cuda.h"

#ifndef TRT_CHECK_MACROS_H
#ifndef TRT_TUT_HELPERS_H

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif
#if __cplusplus < 201103L
#define OVERRIDE
#else
#define OVERRIDE override
#endif

#if NV_TENSORRT_MAJOR > 7
#undef TRTNOEXCEPT
#define TRTNOEXCEPT noexcept
#endif

#endif  // TRT_TUT_HELPERS_H
#endif    // TRT_CHECK_MACROS_H

namespace nvinfer1 {
namespace plugin {

void reportAssertion(const char* msg, const char* file, int line);
void logError(const char* msg, const char* file, const char* fn, int line);

void CudaError(const char* file, const char* function, int line, int status, const char* msg = nullptr);
void CudnnError(const char* file, const char* function, int line, int status, const char* msg = nullptr);
void CublasError(const char* file, const char* function, int line, int status, const char* msg = nullptr);

}  // namespace plugin

}  // namespace nvinfer1

#ifndef TRT_CHECK_MACROS_H
#ifndef TRT_TUT_HELPERS_H

#define API_CHECK(condition)                                               \
{                                                                        \
  if ((condition) == false) {                                            \
    nvinfer1::plugin::logError(#condition, __FILE__, FN_NAME, __LINE__); \
    return;                                                              \
  }                                                                      \
}

#define API_CHECK_RETVAL(condition, retval)                                \
{                                                                        \
  if ((condition) == false) {                                            \
    nvinfer1::plugin::logError(#condition, __FILE__, FN_NAME, __LINE__); \
    return retval;                                                       \
  }                                                                      \
}

#define API_CHECK_WEIGHTS(Name)        \
API_CHECK((Name).values != nullptr); \
API_CHECK((Name).count > 0);         \
API_CHECK(static_cast<int>((Name).type) >= 0 && static_cast<int>((Name).type) < EnumMax<DataType>());

#define API_CHECK_WEIGHTS0(Name)                                                         \
API_CHECK((Name).count >= 0);                                                          \
API_CHECK((Name).count > 0 ? ((Name).values != nullptr) : ((Name).values == nullptr)); \
API_CHECK(static_cast<int>((Name).type) >= 0 && static_cast<int>((Name).type) < EnumMax<DataType>());

#define API_CHECK_WEIGHTS_RETVAL(Name, retval)        \
API_CHECK_RETVAL((Name).values != nullptr, retval); \
API_CHECK_RETVAL((Name).count > 0, retval);         \
API_CHECK_RETVAL(static_cast<int>((Name).type) >= 0 && static_cast<int>((Name).type) < EnumMax<DataType>(), retval);

#define API_CHECK_WEIGHTS0_RETVAL(Name, retval)                                                         \
API_CHECK_RETVAL((Name).count >= 0, retval);                                                          \
API_CHECK_RETVAL((Name).count > 0 ? ((Name).values != nullptr) : ((Name).values == nullptr), retval); \
API_CHECK_RETVAL(static_cast<int>((Name).type) >= 0 && static_cast<int>((Name).type) < EnumMax<DataType>(), retval);

#define API_CHECK_NULL(param) API_CHECK((param) != nullptr)
#define API_CHECK_NULL_RETVAL(param, retval) API_CHECK_RETVAL((param) != nullptr, retval)
#define API_CHECK_NULL_RET_NULL(ptr) API_CHECK_NULL_RETVAL(ptr, nullptr)

#define API_CHECK_ENUM_RANGE(Type, val) API_CHECK(static_cast<int>(val) >= 0 && static_cast<int>(val) < EnumMax<Type>())
#define API_CHECK_ENUM_RANGE_RETVAL(Type, val, retval) \
API_CHECK_RETVAL(static_cast<int>(val) >= 0 && static_cast<int>(val) < EnumMax<Type>(), retval)

#define CUBLASASSERTMSG(status_, msg)                                           \
{                                                                             \
  auto s_ = status_;                                                          \
  if (s_ != CUBLAS_STATUS_SUCCESS) {                                          \
    nvinfer1::plugin::CublasError(__FILE__, FN_NAME, __LINE__, s_, msg); \
  }                                                                           \
}

#define CUBLASASSERT(status_)                                              \
{                                                                        \
  auto s_ = status_;                                                     \
  if (s_ != CUBLAS_STATUS_SUCCESS) {                                     \
    nvinfer1::plugin::CublasError(__FILE__, FN_NAME, __LINE__, s_); \
  }                                                                      \
}

#define CUDNNASSERTMSG(status_, msg)                                           \
{                                                                            \
  auto s_ = status_;                                                         \
  if (s_ != CUDNN_STATUS_SUCCESS) {                                          \
    nvinfer1::plugin::CudnnError(__FILE__, FN_NAME, __LINE__, s_, msg); \
  }                                                                          \
}

#define CUDNNASSERT(status_)                                                   \
{                                                                            \
  auto s_ = status_;                                                         \
  if (s_ != CUDNN_STATUS_SUCCESS) {                                          \
    const char* msg = cudnnGetErrorString(s_);                               \
    nvinfer1::plugin::CudnnError(__FILE__, FN_NAME, __LINE__, s_, msg); \
  }                                                                          \
}

#define CUASSERTMSG(status_, msg)                                             \
{                                                                           \
  auto s_ = status_;                                                        \
  if (s_ != cudaSuccess) {                                                  \
    nvinfer1::plugin::CudaError(__FILE__, FN_NAME, __LINE__, s_, msg); \
  }                                                                         \
}

#define CUASSERT(status_)                                                     \
{                                                                           \
  auto s_ = status_;                                                        \
  if (s_ != cudaSuccess) {                                                  \
    const char* msg = cudaGetErrorString(s_);                               \
    nvinfer1::plugin::CudaError(__FILE__, FN_NAME, __LINE__, s_, msg); \
  }                                                                         \
}

#define ASSERT(assertion)                                                \
{                                                                      \
  if (!(assertion)) {                                                  \
    nvinfer1::plugin::reportAssertion(#assertion, __FILE__, __LINE__); \
  }                                                                    \
}

#define CUERRORMSG(status_)                                                          \
{                                                                                  \
  auto s_ = status_;                                                               \
  if (s_ != 0) {                                                                   \
    nvinfer1::plugin::logError(#status_ " failure.", __FILE__, FN_NAME, __LINE__); \
  }                                                                                \
}

#endif  // TRT_TUT_HELPERS_H
#endif    // TRT_CHECK_MACROS_H

#endif  // PLUGIN_COMMON_CHECKMACROSPLUGIN_H_
