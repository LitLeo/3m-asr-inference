#ifndef PLUGIN_COMMON_COMMON_H_
#define PLUGIN_COMMON_COMMON_H_

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "check_macros_plugin.h"
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "logger.h"
#include "serialize.hpp"
#include "def.h"

using half = __half;

// CUDA: various checks for different function calls.
#ifndef CUDA_CHECK
#define CUDA_CHECK(status)                                                     \
    if (status != cudaSuccess) {                                                 \
      gLogError << "Cuda failure! Error=" << cudaGetErrorString(status) << endl; \
    }
#endif

// cublas: various checks for different function calls.
#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(status)                                 \
    if (status != CUBLAS_STATUS_SUCCESS) {                     \
      gLogError << "Cublas failure! Error=" << status << endl; \
    }
#endif

namespace nvinfer1 {
namespace plugin {

constexpr size_t kAlignment = 256;

constexpr size_t kMaxLayerNameSize = 128;

// att input dim is [B, S, H]
// qkv_to_context input dim is [B, S, 3H]
constexpr uint32_t ATT_BDIM = 0;  // batch dimension
constexpr uint32_t ATT_SDIM = 1;  // seq len dimension
constexpr uint32_t ATT_HDIM = 2;  // hidden dimension

inline int getSMVersion() {
  int device{-1};
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  return props.major * 10 + props.minor;
}

// inline int getMHAMaskPackedSize(int smVersion, nvinfer1::DataType dataType, int sequenceLength) {
//   // this code must match EmbLayerNormPluginDynamic::getOutputDimensions in embLayerNormPlugin.cpp
//   int packedSize = unfusedMaskSize;
//   if ((smVersion == kSM_75 || smVersion == kSM_80 || smVersion == kSM_86)
//       && (dataType == nvinfer1::DataType::kINT8 || dataType == nvinfer1::DataType::kHALF)) {
//     if (sequenceLength == 64) {
//       packedSize = (dataType == nvinfer1::DataType::kHALF ? packedMaskSize64 : packedSize);
//     } else if (sequenceLength == 96) {
//       packedSize = (dataType == nvinfer1::DataType::kHALF ? packedMaskSize96 : packedSize);
//     } else if (sequenceLength == 128) {
//       packedSize = packedMaskSize128;
//     } else if (sequenceLength == 384) {
//       packedSize = packedMaskSize384;
//     }
//   }
//   return packedSize;
// }

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

inline int64_t getWeightsSize(const nvinfer1::Weights& w, nvinfer1::DataType type) {
  return w.count * getElementSize(type);
}

inline int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline std::string Dims2String(const nvinfer1::Dims& d) {
  std::string str = "[";
  for (int i = 0; i < d.nbDims - 1; i++) {
    str += std::to_string(d.d[i]) + ", ";
  }
  str += std::to_string(d.d[d.nbDims - 1]) + "]";
  return str;
}

template <typename IntType>
constexpr IntType ceildiv(IntType a, IntType b) {
  return (a + b - 1) / b;
}
template <typename IntType>
constexpr IntType alignTo(IntType a, IntType b) {
  return ceildiv(a, b) * b;
}

template <typename T>
inline T* deserToDev(const char*& buffer, size_t nbElem) {
  void* dev{nullptr};
  const size_t len = sizeof(T) * nbElem;
  CUASSERT(cudaMalloc(&dev, len));
  CUASSERT(cudaMemcpy(dev, buffer, len, cudaMemcpyHostToDevice));

  buffer += len;
  return static_cast<T*>(dev);
}

template <typename T>
inline void serFromDev(char*& buffer, const T* data, size_t nbElem) {
  const size_t len = sizeof(T) * nbElem;
  CUASSERT(cudaMemcpy(buffer, static_cast<const void*>(data), len, cudaMemcpyDeviceToHost));
  buffer += len;
}

template <typename T>
inline void serFromHost(char*& buffer, const T* data, size_t nbElem) {
  const size_t len = sizeof(T) * nbElem;
  memcpy(buffer, static_cast<const void*>(data), len);
  buffer += len;
}

template <typename T>
inline T* devToDev(const T* data, size_t nbElem) {
  void* dev{nullptr};
  const size_t len = sizeof(T) * nbElem;
  CUASSERT(cudaMalloc(&dev, len));
  CUASSERT(cudaMemcpy(dev, static_cast<const void*>(data), len, cudaMemcpyDeviceToDevice));
  return static_cast<T*>(dev);
}

struct CublasConfigHelper {
  cublasPointerMode_t pm;
  cublasMath_t mm;
  cublasHandle_t cublas;
  explicit CublasConfigHelper(cublasHandle_t cublas_) : cublas(cublas_) {
    cublasGetPointerMode(cublas, &pm);
    cublasGetMathMode(cublas, &mm);
    cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST);
    cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH);
  }
  ~CublasConfigHelper() {
    cublasSetMathMode(cublas, mm);
    cublasSetPointerMode(cublas, pm);
  }
};

template <typename T>
struct CudaDeleter {
  void operator()(T* buf) { if (buf) CUASSERT(cudaFree(buf)); }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter<T>>;

template <typename T>
using cuda_shared_ptr = std::shared_ptr<T>;

template <typename T>
void make_cuda_shared(cuda_shared_ptr<T>& ptr, void* cudaMem) {
  ptr.reset(static_cast<T*>(cudaMem), CudaDeleter<T>());
}

struct WeightsWithOwnership : public nvinfer1::Weights {
  WeightsWithOwnership() {
    values = nullptr;
    count = 0;
  }
  ~WeightsWithOwnership() { operator delete[](const_cast<void*>(values)); }

  WeightsWithOwnership(const WeightsWithOwnership&) = delete;
  WeightsWithOwnership operator=(const WeightsWithOwnership&) = delete;
  WeightsWithOwnership(const WeightsWithOwnership&&) = delete;
  WeightsWithOwnership operator=(const WeightsWithOwnership&&) = delete;

  void convertAndCopy(const nvinfer1::Weights& src, nvinfer1::DataType type) {
    this->type = type;
    this->count = src.count;

    if (type == nvinfer1::DataType::kFLOAT) {
      auto destBuf = new float[src.count];
      this->values = destBuf;

      if (src.type == nvinfer1::DataType::kFLOAT) {
        //gLogVerbose << "Float Weights(Host) => Float Array(Host)\n";
        std::copy_n(static_cast<const float*>(src.values), src.count, destBuf);
      } else {
        assert(src.type == nvinfer1::DataType::kHALF);

        //gLogVerbose << "Half Weights(Host) => Float Array(Host)\n";
        const auto s = static_cast<const half*>(src.values);
        auto d = static_cast<float*>(const_cast<void*>(this->values));

        for (auto it = 0; it < src.count; it++) {
          d[it] = __half2float(s[it]);
        }
      }
    } else if (type == nvinfer1::DataType::kHALF) {
      auto destBuf = new half[src.count];
      this->values = destBuf;

      if (src.type == nvinfer1::DataType::kHALF) {
        //gLogVerbose << "Half Weights(Host) => Half Array(Host)\n";
        std::copy_n(static_cast<const half*>(src.values), src.count, destBuf);
      } else {
        assert(src.type == nvinfer1::DataType::kFLOAT);

        //gLogVerbose << "Float Weights(Host) => Half Array(Host)\n";
        const auto s = static_cast<const float*>(src.values);
        auto d = static_cast<half*>(const_cast<void*>(this->values));

        for (auto it = 0; it < src.count; it++) {
          d[it] = __float2half(s[it]);
        }
      }
    } else {
      throw std::runtime_error("Unsupported DataType specified for plugin.");
    }
  }

  void convertAndCopy(const char*& srcBuf, size_t count, nvinfer1::DataType type) {
    this->type = type;
    this->count = count;
    const auto nbBytes = getWeightsSize(*this, type);
    auto destBuf = new char[nbBytes];
    this->values = destBuf;

    std::copy_n(srcBuf, nbBytes, destBuf);
    srcBuf += nbBytes;
  }
};

template <typename T>
inline void copyToDevice(WeightsWithOwnership& hostWeights, size_t nbBytes, cuda_unique_ptr<T>& cudaWeights) {
  if (hostWeights.values) {
    void* cudaMem{nullptr};
    CUASSERT(cudaMalloc(&cudaMem, nbBytes));
    CUASSERT(cudaMemcpy(cudaMem, hostWeights.values, nbBytes, cudaMemcpyHostToDevice));
    cudaWeights.reset(static_cast<T*>(cudaMem));
  }
}

inline void convertAndCopyToDevice(const nvinfer1::Weights& src, float* destDev) {
  size_t wordSize = sizeof(float);
  size_t nbBytes = src.count * wordSize;
  if (src.type == nvinfer1::DataType::kFLOAT) {
    //gLogVerbose << "Float Weights(Host) => Float Array(Device)" << std::endl;
    CUASSERT(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
  } else {
    //gLogVerbose << "Half Weights(Host) => Float Array(Device)" << std::endl;
    std::vector<float> tmp(src.count);
    const half* values = reinterpret_cast<const half*>(src.values);

    for (size_t it = 0; it < tmp.size(); it++) {
      tmp[it] = __half2float(values[it]);
    }

    CUASSERT(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
  }
}

inline void convertAndCopyToDevice(const nvinfer1::Weights& src, half* destDev) {
  size_t wordSize = sizeof(half);
  size_t nbBytes = src.count * wordSize;
  if (src.type == nvinfer1::DataType::kHALF) {
    //gLogVerbose << "Half Weights(Host) => Half Array(Device)" << std::endl;
    CUASSERT(cudaMemcpy(destDev, src.values, nbBytes, cudaMemcpyHostToDevice));
  } else {
    //gLogVerbose << "Float Weights(Host) => Half Array(Device)" << std::endl;
    std::vector<half> tmp(src.count);
    const float* values = reinterpret_cast<const float*>(src.values);

    for (size_t it = 0; it < tmp.size(); it++) {
      tmp[it] = __float2half(values[it]);
    }
    CUASSERT(cudaMemcpy(destDev, &tmp[0], nbBytes, cudaMemcpyHostToDevice));
  }
}

inline nvinfer1::DataType fieldTypeToDataType(const nvinfer1::PluginFieldType ftype) {
  switch (ftype) {
    case nvinfer1::PluginFieldType::kFLOAT32: {
      //gLogVerbose << "PluginFieldType is Float32" << std::endl;
      return nvinfer1::DataType::kFLOAT;
    }
    case nvinfer1::PluginFieldType::kFLOAT16: {
      //gLogVerbose << "PluginFieldType is Float16" << std::endl;
      return nvinfer1::DataType::kHALF;
    }
    case nvinfer1::PluginFieldType::kINT32: {
      //gLogVerbose << "PluginFieldType is Int32" << std::endl;
      return nvinfer1::DataType::kINT32;
    }
    case nvinfer1::PluginFieldType::kINT8: {
      //gLogVerbose << "PluginFieldType is Int8" << std::endl;
      return nvinfer1::DataType::kINT8;
    }
    default:
      throw std::invalid_argument("No corresponding datatype for plugin field type");
  }
}

}  // namespace plugin
}  // namespace nvinfer1

#endif  // PLUGIN_COMMON_COMMON_H_
