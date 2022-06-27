#include "prior_prob_plugin.h"

namespace nvinfer1 {
namespace plugin {

/**
 * @brief   求先验概率，log_prior已经在外面求和并log 了
            c[i,j] = logf(a[i,j]+1e-20) + b[j]
 */
template <typename T>
__global__ void PriorProbKernel(const T *input, const T *log_prior, T *output, int dim) {
  const int idx = threadIdx.x;
  const int bdx = blockIdx.x;
  T *output_ptr = output + bdx * dim;
  const T *input_ptr = input + bdx * dim;

  for (int i = idx; i < dim; i += blockDim.x) {
    output_ptr[i] = logf(input_ptr[i] + 1e-20) + log_prior[i];
  }
}

int ComputePriorProb(cudaStream_t stream, int n, int dim, const float *input, 
                     const float *log_prior, float *output) {
  int blockSize = 64;
  const int gridSize = n;
  PriorProbKernel<float><<<gridSize, blockSize, 0, stream>>>(input, log_prior, output, dim);
    
  CUDA_CHECK(cudaPeekAtLastError());
  return 0;
}

/*#ifdef __SCORE_HALF__*/
/*inline int ComputeCmvn(cudaStream_t stream, int n, const half* input, half* output) {*/
    /*const int blockSize = 256;*/

    /*if (0 == (n & 1)) {*/
        /*const int n2 = n / 2;*/

        /*const int gridSize = (n2 + blockSize - 1) / blockSize;*/
        /*const half2 A2 = __floats2half2_rn(A, A);*/
        /*const half2 B2 = __floats2half2_rn(B, B);*/
        /*const half2 C2 = __floats2half2_rn(C, C);*/
        /*const half2 *input2 = reinterpret_cast<const half2*>(input);*/
        /*half2 *output2 = reinterpret_cast<half2*>(output);*/
        /*CmvnKernel<half2, blockSize><<<gridSize, blockSize, 0, stream>>>(A2, B2, C2, n2, input2, output2);*/
    /*}*/
    /*else {*/
        /*const int gridSize = (n + blockSize - 1) / blockSize;*/
        /*CmvnKernel<half, blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, C, n, input, output);*/
    /*}*/

    /*CUDA_CHECK(cudaPeekAtLastError());*/
    /*return 0;*/
/*}*/
/*#endif*/


}  // nvinfer1
}  // plugin

