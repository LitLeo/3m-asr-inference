#ifndef PLUGIN_CAT_SPLIT_CACHE_KERNEL_H_
#define PLUGIN_CAT_SPLIT_CACHE_KERNEL_H_

#include "common.h"

namespace nvinfer1 {
namespace plugin {

int ComputeCatSplitCache(cudaStream_t stream, const int batch, const int cache_dim, const int input_dim,
                         const float* in_cache, const float* input, float* output, float* out_cache);

int ComputeCatSplitCache(cudaStream_t stream, const int batch, const int cache_dim, const int input_dim,
                         const half* in_cache, const half* input, half* output, half* out_cache);

}  // namespace plugin
}  // namespace nvinfer1
#endif  // PLUGIN_CAT_SPLIT_CACHE_KERNEL_H_
