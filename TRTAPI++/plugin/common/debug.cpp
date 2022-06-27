#include "debug.h"

#include <iomanip>

#include "cuda_runtime.h"

using namespace std;

template <class T>
void p_row(T *row_ptr, int row, int width) {
  ios::fmtflags f(cout.flags());  // save old format
  cout.flags(ios::left);

  if (width > 6) {
    cout << setprecision(8) << "[row_" << row << "]: " << row_ptr[0] << ", " << row_ptr[1] << ", " << row_ptr[2]
         << " ... " << row_ptr[width - 3] << ", " << row_ptr[width - 2] << ", " << row_ptr[width - 1] << endl;
  } else {
    cout << "[row_" << row << "]: ";
    for (size_t i = 0; i < width; i++) {
      cout << setprecision(8) << row_ptr[i] << ", ";
    }
    cout << endl;
  }
  // restore old format
  cout.flags(f);
}

template <class T>
void p(T *ptr, int height, int width) {
  printf("Matrix[%zu, %zu]\n", height, width);
  if (height > 6) {
    p_row(ptr, 0, width);
    p_row(ptr + width, 1, width);
    p_row(ptr + width * 2, 2, width);
    printf("..........\n");
    p_row(ptr + width * (height - 3), height - 3, width);
    p_row(ptr + width * (height - 2), height - 2, width);
    p_row(ptr + width * (height - 1), height - 1, width);
  } else {
    for (size_t i = 0; i < height; i++) {
      p_row(ptr + width * i, i, width);
    }
  }
}

#ifdef BUILD_LIBTORCH_PLUGINS
void print_tensor(torch::Tensor tensor, std::string message, bool print_value) {
  auto dense_tensor = torch::_cast_Float(tensor);
  dense_tensor = dense_tensor.cpu().clone();

  std::string size_str = "[";
  for (int i = 0; i < dense_tensor.sizes().size(); i++) size_str += std::to_string(dense_tensor.sizes()[i]) + " ";
  size_str += "]";

  torch::Tensor sum = torch::sum(dense_tensor);
  cout << message << ", size=" << size_str << ", type=" << dense_tensor.dtype() << ", sum=" << sum << endl;

  if (print_value) {
    auto cpu_tensor = dense_tensor.to(torch::kCPU);

    if (dense_tensor.sizes().size() == 2) {
      p(cpu_tensor.data_ptr<float>(), cpu_tensor.sizes()[0], cpu_tensor.sizes()[1]);
    } else if (cpu_tensor.sizes().size() == 3) {
      auto h = cpu_tensor.sizes()[0];
      if (h < 6) {
        for (int i = 0; i < cpu_tensor.sizes()[0]; i++) {
          printf("dim=%d\n", i);
          p(cpu_tensor[i].data_ptr<float>(), cpu_tensor.sizes()[1], cpu_tensor.sizes()[2]);
        }
      } else {
        printf("dim=%d\n", 0);
        p(cpu_tensor[0].data_ptr<float>(), cpu_tensor.sizes()[1], cpu_tensor.sizes()[2]);
        printf("dim=%d\n", 1);
        p(cpu_tensor[1].data_ptr<float>(), cpu_tensor.sizes()[1], cpu_tensor.sizes()[2]);
        printf("..........\n");
        printf("dim=%d\n", h - 2);
        p(cpu_tensor[h - 2].data_ptr<float>(), cpu_tensor.sizes()[1], cpu_tensor.sizes()[2]);
        printf("dim=%d\n", h - 1);
        p(cpu_tensor[h - 1].data_ptr<float>(), cpu_tensor.sizes()[1], cpu_tensor.sizes()[2]);
      }
    } else {
      auto arr = cpu_tensor.data_ptr<float>();
      printf("dense_tensor value: %f %f %f %f %f %f\n", arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
    }
  }

  printf("================================================================================\n");
}

void print_inttensor(torch::Tensor tensor, std::string message, bool print_value) {
  auto dense_tensor = tensor.cpu().clone();

  std::string size_str = "[";
  for (int i = 0; i < dense_tensor.sizes().size(); i++) size_str += std::to_string(dense_tensor.sizes()[i]) + " ";
  size_str += "]";

  torch::Tensor sum = torch::sum(dense_tensor);
  cout << message << ", size=" << size_str << ", type=" << dense_tensor.dtype() << ", sum=" << sum << endl;

  if (print_value) {
    auto cpu_tensor = dense_tensor.to(torch::kCPU);

    if (dense_tensor.sizes().size() == 2) {
      p(cpu_tensor.data_ptr<int>(), cpu_tensor.sizes()[0], cpu_tensor.sizes()[1]);
    } else if (cpu_tensor.sizes().size() == 3) {
      for (int i = 0; i < cpu_tensor.sizes()[0]; i++) {
        printf("dim=%d\n", i);
        p(cpu_tensor[i].data_ptr<int>(), cpu_tensor.sizes()[1], cpu_tensor.sizes()[2]);
      }
    } else {
      auto arr = cpu_tensor.data_ptr<int>();
      printf("dense_tensor value: %d %d %d %d %d %d\n", arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
    }
  }

  printf("================================================================================\n");
}

void print_tensor(const float *ptr, nvinfer1::Dims dims, std::string message, bool print_value) {
  std::vector<int64_t> input_dims(dims.nbDims);
  for (int i = 0; i < dims.nbDims; i++) input_dims[i] = dims.d[i];

  auto type = at::device(at::kCUDA).dtype(torch::kFloat);
  auto input = at::from_blob(const_cast<void *>((const void *)ptr), input_dims, type);

  print_tensor(input, message, print_value);
}

void print_tensor(const half *ptr, nvinfer1::Dims dims, std::string message, bool print_value) {
  std::vector<int64_t> input_dims(dims.nbDims);
  for (int i = 0; i < dims.nbDims; i++) input_dims[i] = dims.d[i];

  auto type = at::device(at::kCUDA).dtype(torch::kFloat16);
  auto input = at::from_blob(const_cast<void *>((const void *)ptr), input_dims, type);

  input = torch::_cast_Float(input);

  print_tensor(input, message, print_value);
}

#endif  // BUILD_LIBTORCH_PLUGINS

template <class T>
void print_data_tpl(const T *ptr, int len, std::string message) {
  T *h_ptr = new T[len];

  cudaMemcpy(h_ptr, ptr, sizeof(T) * len, cudaMemcpyDefault);
  cudaDeviceSynchronize();

  cout << message << ": ";
  for (int i = 0; i < len; i++) cout << h_ptr[i] << " ";
  printf("\n");

  delete [] h_ptr;
}

void print_data(const float *ptr, int len, std::string message) { print_data_tpl<float>(ptr, len, message); }

void print_data(const int *ptr, int len, std::string message) { print_data_tpl<int>(ptr, len, message); }

void print_data(const half* ptr, int len, std::string message) {
  half *h_ptr = new half[len];

  cudaMemcpy(h_ptr, ptr, sizeof(half) * len, cudaMemcpyDefault);
  cudaDeviceSynchronize();

  cout << message << ": ";
  for (int i = 0; i < len; i++) cout << __half2float(h_ptr[i]) << " ";
  printf("\n");

  delete [] h_ptr;
}

// bool test_compare(const torch::Tensor a, const torch::Tensor b) {
//     auto a_size = a.sizes();
//     auto b_size = b.sizes();

//     for (size_t i = 0; i < a_size.size(); i++) {
//         if (a_size[i] != b_size[i]) {
//             printf("i=%d, a_size %f != b_size %f \n", i, a_size[i], b_size[i]);
//             assert(0);
//         }
//     }

//     test_compare(a.data_ptr<float>(), b.data_ptr<float>(), a.numel());
// }

// bool test_compare(const Mat &a, const Mat &b, float diff) {
//     PCHECK(a.n() == b.n());
//     PCHECK(a.c() == b.c());
//     PCHECK(a.h() == b.h());
//     PCHECK(a.w() == b.w());
//     PCHECK(a.stride_w() == b.stride_w());
//     PCHECK(a.leading_w() == b.leading_w());
//     PCHECK(a.count() == b.count());
//     PCHECK(a.data_type() == b.data_type());
//     switch (a.data_type()) {
//         case DataType::kFLOAT:
//             return test_compare(a.cpu_data<float>(), b.cpu_data<float>(), a.count(), diff);

//         default:
//             return false;
//     }
//     return false;
// }

// bool test_compare(const FMatrix &a, const FMatrix &b, float diff) {
//     return test_compareT(a, b, diff);
// }
// bool test_compare(const FMatrix &a, const FMatrix &b, const FMatrix &c, float diff) {
//     return test_compareT(a, b, diff) && test_compareT(a, c, diff);
// }
// bool test_compare(const IMatrix &a, const IMatrix &b, int diff) {
//     return test_compareT(a, b, diff);
// }
// bool test_compare(const CMatrix &a, const CMatrix &b, signed char diff) {
//     return test_compareT(a, b, diff);
// }
// bool test_compare(const UCMatrix &a, const UCMatrix &b, unsigned char diff) {
//     return test_compareT(a, b, diff);
// }

