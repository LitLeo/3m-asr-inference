#ifndef PLUGIN_SILU_KERNEL_H_
#define PLUGIN_SILU_KERNEL_H_

#include "NvInfer.h"

// #include "plugin_math.h"
#include "common.h"

namespace nvinfer1 {
namespace plugin {

int compute_silu(const float* input, const int N, float* output, cudaStream_t stream);

int compute_silu(const half* input, const int N, half* output, cudaStream_t stream);

}  // namespace plugin
}  // namespace nvinfer1

#endif  // PLUGIN_SILU_KERNEL_H_
