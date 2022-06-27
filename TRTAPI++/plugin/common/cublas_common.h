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


//template <typename T>
//cublasStatus_t inline cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                                 //cublasOperation_t transb, int m, int n, int k, const T alpha,
                                                 //const T* A, int lda, long long int strideA, const T* B, int ldb,
                                                 //long long int strideB, const T beta, T* C, int ldc,
                                                 //long long int strideC, int batchCount, cublasGemmAlgo_t algo);

//template <>
//cublasStatus_t inline cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                                 //cublasOperation_t transb, int m, int n, int k, const float alpha,
                                                 //const float* A, int lda, long long int strideA, const float* B,
                                                 //int ldb, long long int strideB, const float beta, float* C, int ldc,
                                                 //long long int strideC, int batchCount, cublasGemmAlgo_t algo) {
  //return ::cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &alpha, A, CUDA_R_32F, lda, strideA, B,
                                      //CUDA_R_32F, ldb, strideB, &beta, C, CUDA_R_32F, ldc, strideC, batchCount,
                                      //CUDA_R_32F, algo);
//}

//template <>
//cublasStatus_t inline cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                                                 //cublasOperation_t transb, int m, int n, int k, const half alpha,
                                                 //const half* A, int lda, long long int strideA, const half* B, int ldb,
                                                 //long long int strideB, const half beta, half* C, int ldc,
                                                 //long long int strideC, int batchCount, cublasGemmAlgo_t algo) {
  //return ::cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, &alpha, A, CUDA_R_16F, lda, strideA, B,
                                      //CUDA_R_16F, ldb, strideB, &beta, C, CUDA_R_16F, ldc, strideC, batchCount,
                                      //CUDA_R_16F, algo);
//}

//template <typename T>
//cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                               //cublasOperation_t transb, int m, int n, int k, const T alpha, const T* A,
                                               //int lda, long long int strideA, const T* B, int ldb,
                                               //long long int strideB, const T beta, T* C, int ldc,
                                               //long long int strideC, int batchCount);

//template <>
//cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                               //cublasOperation_t transb, int m, int n, int k, const float alpha,
                                               //const float* A, int lda, long long int strideA, const float* B, int ldb,
                                               //long long int strideB, const float beta, float* C, int ldc,
                                               //long long int strideC, int batchCount) {
  //return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C,
                                   //ldc, strideC, batchCount);
//}

//template <>
//cublasStatus_t inline cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                               //cublasOperation_t transb, int m, int n, int k, const half alpha,
                                               //const half* A, int lda, long long int strideA, const half* B, int ldb,
                                               //long long int strideB, const half beta, half* C, int ldc,
                                               //long long int strideC, int batchCount) {
  //return cublasHgemmStridedBatched(handle, transa, transb, m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C,
                                   //ldc, strideC, batchCount);
//}


}  // namespace plugin
}  // namespace nvinfer1

#endif  // PLUGIN_COMMON_CUBLAS_COMMON_H_
