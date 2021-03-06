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

#include "check_macros_plugin.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdlib>

#include "trt_logger.h"

namespace nvinfer1 {
namespace plugin {

// break-pointable
void CudaError(const char* file, const char* function, int line, int status, const char* msg) {
  //CudaError error(file, function, line, status, msg);
  //error.log(LOG(ERROR));
  //throw error;
}

// break-pointable
void CublasError(const char* file, const char* function, int line, int status, const char* msg) {
  if (msg == nullptr) {
    auto s_ = static_cast<cublasStatus_t>(status);
    switch (s_) {
      case CUBLAS_STATUS_SUCCESS:
        msg = "CUBLAS_STATUS_SUCCESS";
        break;
      case CUBLAS_STATUS_NOT_INITIALIZED:
        msg = "CUBLAS_STATUS_NOT_INITIALIZED";
        break;
      case CUBLAS_STATUS_ALLOC_FAILED:
        msg = "CUBLAS_STATUS_ALLOC_FAILED";
        break;
      case CUBLAS_STATUS_INVALID_VALUE:
        msg = "CUBLAS_STATUS_INVALID_VALUE";
        break;
      case CUBLAS_STATUS_ARCH_MISMATCH:
        msg = "CUBLAS_STATUS_ARCH_MISMATCH";
        break;
      case CUBLAS_STATUS_MAPPING_ERROR:
        msg = "CUBLAS_STATUS_MAPPING_ERROR";
        break;
      case CUBLAS_STATUS_EXECUTION_FAILED:
        msg = "CUBLAS_STATUS_EXECUTION_FAILED";
        break;
      case CUBLAS_STATUS_INTERNAL_ERROR:
        msg = "CUBLAS_STATUS_INTERNAL_ERROR";
        break;
      case CUBLAS_STATUS_NOT_SUPPORTED:
        msg = "CUBLAS_STATUS_NOT_SUPPORTED";
        break;
      case CUBLAS_STATUS_LICENSE_ERROR:
        msg = "CUBLAS_STATUS_LICENSE_ERROR";
        break;
    }
  }
  //CublasError error(file, function, line, status, msg);
  //error.log(LOG(ERROR));
  //throw error;
}

// break-pointable
void CudnnError(const char* file, const char* function, int line, int status, const char* msg) {
  //CudnnError error(file, function, line, status, msg);
  //error.log(LOG(ERROR));
  //throw error;
}

void logError(const char* msg, const char* file, const char* fn, int line) {
  LOG(ERROR) << "Parameter check failed at: " << file << "::" << fn << "::" << line;
  LOG(ERROR) << ", condition: " << msg << std::endl;
}

// break-pointable
void reportAssertion(const char* msg, const char* file, int line) {
  std::ostringstream stream;
  stream << "Assertion failed: " << msg << std::endl << file << ':' << line << std::endl << "Aborting..." << std::endl;
  getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
  cudaDeviceReset();
  abort();
}

//void TRTException::log(std::ostream& logStream) const {
  //logStream << file << " (" << line << ") - " << name << " Error in " << function << ": " << status;
  //if (message != nullptr) {
    //logStream << " (" << message << ")";
  //}
  //logStream << std::endl;
//}

}  // namespace plugin
}  // namespace nvinfer1
