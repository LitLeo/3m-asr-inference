#ifndef PLUGIN_ATT_STREAM_SOFTMAX_KERNEL_H_
#define PLUGIN_ATT_STREAM_SOFTMAX_KERNEL_H_

#include "common.h"

namespace nvinfer1 {
namespace plugin {

int ComputeAttStreamSoftmax(cudaStream_t stream, const int ld, const int B, const int N, const float rsqrt_head_size,
                            const int cache_len, const int* decode_frame_num, const int* mask_idx, const float* input,
                            float* output);

int ComputeAttStreamSoftmax(cudaStream_t stream, const int ld, const int B, const int N, const float rsqrt_head_size,
                            const int cache_len, const int* decode_frame_num, const int* mask_idx, const half* input,
                            half* output);

}  // namespace plugin
}  // namespace nvinfer1
#endif  // PLUGIN_ATT_STREAM_SOFTMAX_KERNEL_H_
