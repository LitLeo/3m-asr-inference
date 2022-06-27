#include "rel_pos_qkv_to_context_plugin.h"

#include "att_masked_softmax_plugin.h"
#include "common.h"

using namespace nvinfer1;
using namespace std;

namespace nvinfer1 {
namespace plugin {

template <typename T>
__global__ void TransposeCtxKernel(const int H, const T* input, T* output) {
  // Input:  HxSxNxB
  // Output: HxNxSxB

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;

  int N = blockDim.y;
  int S = gridDim.x;
  // B = gridDim.y

  const int NH = N * H;
  const int NHS = NH * S;
  const int in_offset = s * H + n * S * H + b * NHS;
  const int out_offset = n * H + s * NH + b * NHS;

  const int i = threadIdx.x;
  if (i < H) {
    output[out_offset + i] = input[in_offset + i];
  }
}

inline void LaunchTransCtx(cudaStream_t stream, const int S, const int B, 
    const int head_size, const int num_heads, const float* input, float* output) {
  const dim3 grid(S, B, 1);
  if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    const dim3 block(H, num_heads, 1);
    TransposeCtxKernel<float2><<<grid, block, 0, stream>>>(H, input2, output2);
    CUDA_CHECK(cudaPeekAtLastError());
  } else {
    const dim3 block(head_size, num_heads, 1);
    TransposeCtxKernel<float><<<grid, block, 0, stream>>>(head_size, input, output);
    CUDA_CHECK(cudaPeekAtLastError());
  }
}

inline void LaunchTransCtx(cudaStream_t stream, const int S, const int B, 
    const int head_size, const int num_heads, const half* input, half* output) {
  const dim3 grid(S, B, 1);
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    const dim3 block(H, num_heads, 1);
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    TransposeCtxKernel<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const dim3 block(H, num_heads, 1);
    const half2* input2 = reinterpret_cast<const half2*>(input);
    half2* output2 = reinterpret_cast<half2*>(output);
    TransposeCtxKernel<half2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else { 
    // this should be an "odd" case. probably not worth catching it in the half2 kernel.
    const dim3 block(head_size, num_heads, 1);
    TransposeCtxKernel<half><<<grid, block, 0, stream>>>(head_size, input, output);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void TransposeQKV(const int H, const T* input, T* output) {
  // Input:  HxNx3xSxB
  // Output: HxSxNxBx3
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z; // matrix id

  const int N = blockDim.y;

  const int S = gridDim.x;
  const int B = gridDim.y;
  const int NH = N * H;
  const int NHS = NH * S;
  const int in_offset = n * H + m * NH + s * 3 * NH + b * NHS * 3;
  const int out_offset = s * H + n * S * H + b * NHS + m * NHS * B;

  const int i = threadIdx.x;
  if (i < H)
    output[out_offset + i] = input[in_offset + i];
}

inline void LaunchTransQKV(cudaStream_t stream, const int S, const int B, const int head_size, const int num_heads,
    const float* input, float* output) {

  const dim3 grid(S, B, 3);
  if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    const dim3 block(H, num_heads, 1);
    TransposeQKV<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else {
    const dim3 block(head_size, num_heads, 1);
    TransposeQKV<float><<<grid, block, 0, stream>>>(head_size, input, output);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

inline void LaunchTransQKV(cudaStream_t stream, const int S, const int B, const int head_size, const int num_heads,
    const half* input, half* output){

  const dim3 grid(S, B, 3);
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    const dim3 block(H, num_heads, 1);
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    TransposeQKV<float2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const dim3 block(H, num_heads, 1);
    const half2* input2 = reinterpret_cast<const half2*>(input);
    half2* output2 = reinterpret_cast<half2*>(output);
    TransposeQKV<half2><<<grid, block, 0, stream>>>(H, input2, output2);
  } else { 
    // this should be an "odd" case. probably not worth catching it in the half2 kernel..
    const dim3 block(head_size, num_heads, 1);
    TransposeQKV<half><<<grid, block, 0, stream>>>(head_size, input, output);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
inline int ComputeRelPosQKVToCtxTpl(cublasHandle_t& cublas, const int B, const int S, const int num_heads, const int head_size,
    const float rsqrtHeadSize, const T* input, T* output, T* qkptr, T* pptr, T* tptr, cudaStream_t stream,
    const int* maskIdx = nullptr) {

  // input should be BxSx3xNxH => tptr: 3xBxNxSxH
  LaunchTransQKV(stream, S, B, head_size, num_heads, input, tptr);

  const int tsize = B * num_heads * S * head_size;
  const int imatSize = S * head_size;
  const int omatSize = S * S;
  const int numMats = B * num_heads;
  const T* qptr = tptr;
  const T* kptr = qptr + tsize;
  const T* vptr = kptr + tsize;

  cublasSetStream(cublas, stream);
  CublasConfigHelper helper(cublas);

  // Q, K, V: BxNxSxH (inputs)
  // Q * K': BxNxSxS (-> scratch1)
  // P: BxNxSxS (-> scratch2)
  // P * V: BxNxSxH (output)

  // compute Q*K' (as K'*Q)
  CUBLAS_CHECK(cublasGemmStridedBatched<T>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, S, S, head_size, 1.f, kptr, head_size, imatSize,
      qptr, head_size, imatSize, 0.f, qkptr, S, omatSize, numMats));

  // apply softmax
  if (maskIdx)
  { // if we have a mask
      compute_masked_scaled_softmax_tpl<T>(stream, S, B, num_heads, rsqrtHeadSize, maskIdx, qkptr, pptr);
  }
  else
  { // if we don't have a mask
      /*computeScaledSoftmax<T>(stream, S, B, num_heads, rsqrtHeadSize, qkptr, pptr);*/
   assert(0); 
  }

  // compute P*V (as V*P)
  CUBLAS_CHECK(cublasGemmStridedBatched<T>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, head_size, S, S, 1.f, vptr, head_size, imatSize,
      pptr, S, omatSize, 0.f, tptr, head_size, imatSize, numMats));

  // tptr is 3xBxNxSxH, so 3x output
  LaunchTransCtx(stream, S, B, head_size, num_heads, tptr, output);
    return 0;
}

template<typename T>
int ComputeRelPosQKVToCtxTpl(QKVToContextParams& params, const T* input, T* output, const int* mask_idx = nullptr) {

}

int ComputeRelPosQKVToCtx(cublasHandle_t& cublas, const int B, const int S, const int num_heads, const int head_size,
    const float rsqrtHeadSize, const float* input, float* output, float* qkptr, float* pptr, float* tptr, cudaStream_t stream,
    const int* maskIdx) {
  return compute_rel_poqkvToCtxTpl(cublas, B, S, num_heads, head_size, rsqrtHeadSize, input, output, qkptr, pptr, tptr, stream, maskIdx);
}

int ComputeRelPosQKVToCtx(cublasHandle_t& cublas, const int B, const int S, const int num_heads, const int head_size,
    const float rsqrtHeadSize, const half* input, half* output, half* qkptr, half* pptr, half* tptr, cudaStream_t stream,
    const int* maskIdx) {
  return ComputeRelPosQKVToCtxTpl(cublas, B, S, num_heads, head_size, rsqrtHeadSize, input, output, qkptr, pptr, tptr, stream, maskIdx);
}

} // plugin
} // nvinfer1

