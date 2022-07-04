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

#ifndef CUDA_STREAM_MANAGER_H
#define CUDA_STREAM_MANAGER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace nvinfer1 {
namespace plugin {

#define SMGR_N_STREAMS 8

class CudaStreamManager {
 public:
  int device;
  cublasHandle_t* handles_;
  cudaStream_t* streams_;
  bool init_;

 public:
  CudaStreamManager(): init_(false) {}

  void Init();
  int SyncStream(int idx = 0);
  int SyncRangeStream(int start_idx, int end_idx);
  int SyncAllStream();
  void Destroy();

  cudaStream_t Stream(size_t idx = 0) {
    return this->streams_[idx % SMGR_N_STREAMS];
  }
  cublasHandle_t CublasHandle(size_t idx = 0) {
    return handles_[idx % SMGR_N_STREAMS];
  }

  ~CudaStreamManager() {
    this->Destroy();
  }
};

}  // namespace plugin
}  // namespace nvinfer1

#endif  // CUDA_STREAM_MANAGER
