#include <unordered_map>
#include <mutex>
#include <cassert>
#include <thread>
#include <iostream>

#include "cuda_stream_manager.h"

namespace nvinfer1 {
namespace plugin {

int CudaStreamManager::SyncStream(int idx) {
  //if (!init_) return -1;
  if (!init_) assert(0);
    if (idx < 0 || idx >= SMGR_N_STREAMS)
        return -1;

    cudaStreamSynchronize(streams_[idx]);
    return 0;
}

int CudaStreamManager::SyncAllStream() {
    for (int i = 0; i < SMGR_N_STREAMS; ++i)
        cudaStreamSynchronize(streams_[i]);
    return 0;
}

int CudaStreamManager::SyncRangeStream(int start_idx, int end_idx) {
    if (start_idx < 0 || end_idx >= SMGR_N_STREAMS || end_idx <= start_idx)
        return -1;

    for (int i = start_idx; i < end_idx; ++i)
        cudaStreamSynchronize(streams_[i]);
    return 0;
}

void CudaStreamManager::Init() {
    streams_ = new cudaStream_t[SMGR_N_STREAMS];
    handles_ = new cublasHandle_t[SMGR_N_STREAMS];
    for (size_t i = 0; i < SMGR_N_STREAMS; ++i) {
        cudaStreamCreate(streams_ + i);
        cublasCreate(handles_ + i);
        cublasSetStream(handles_[i], streams_[i]);
    }
}

void CudaStreamManager::Destroy() {
    for (size_t i = 0; i < SMGR_N_STREAMS; ++i) {
        cudaStreamDestroy(streams_[i]);
        cublasDestroy(handles_[i]);
    }
    delete[] streams_;
    delete[] handles_;
}

}  // end plugin
}  // end nvinfer1

