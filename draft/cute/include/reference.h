#pragma once
#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "helper.h"
#include "utils.h"
#include <cmath>
/*
cutlass golden compute
*/
template <typename Atype, typename ALayout, typename Btype, typename BLayout,
          typename Ctype, typename CLayout>
inline void
cutlass_gemmTN_ref(cutlass::HostTensor<Atype, ALayout> const &A, // row-major
                   cutlass::HostTensor<Btype, BLayout> const &B, // col-major
                   cutlass::HostTensor<Ctype, CLayout> &C,
                   Ctype alpha = static_cast<Ctype>(1),
                   Ctype beta = static_cast<Ctype>(0), int repeat = 1000) {
  int m = A.extent().row();
  int n = B.extent().column();
  int k = A.extent().column();
  float gflop = 2.0 * m * n * k / 1e9;
  using Gemm = cutlass::gemm::device::Gemm<Atype, ALayout, Btype, BLayout,
                                           Ctype, CLayout, Ctype,
                                           cutlass::arch::OpClassTensorOp>;
  // cute::print_type(Gemm{});
  Gemm op;
  typename Gemm::Arguments args({m, n, k}, {A.device_data(), k},
                                {B.device_data(), k}, {C.device_data(), n},
                                {C.device_data(), n}, {alpha, beta});
  GpuTimer timer;

  CUTLASS_CHECK(op(args));
  timer.start();
  for (int iter = 0; iter < repeat; iter++) {
    op(args);
  }
  timer.stop();
  float duration_ms = timer.elapsed_millis() / repeat;
  float tflops = gflop / duration_ms;
  printf("cutlass ref: %f tflops\n", tflops);
}
/*
cublas golden compute
*/
template <typename Atype, typename ALayout, typename Btype, typename BLayout,
          typename Ctype, typename CLayout,
          typename ScalarType = typename UnderlyingType<
              Ctype>::type> // cvt cutlass::half_t to half
inline void
cublas_gemmTN_ref(cutlass::HostTensor<Atype, ALayout> const &A, // row-major
                  cutlass::HostTensor<Btype, BLayout> const &B, // col-major
                  cutlass::HostTensor<Ctype, CLayout> &C,
                  ScalarType alpha = static_cast<ScalarType>(1),
                  ScalarType beta = static_cast<ScalarType>(0),
                  int repeat = 1000, cudaStream_t stream = nullptr) {
  int m = A.extent().row();
  int n = B.extent().column();
  int k = A.extent().column();
  using Atype_ = typename UnderlyingType<Atype>::type;
  using Btype_ = typename UnderlyingType<Btype>::type;
  using Ctype_ = typename UnderlyingType<Ctype>::type;
  float gflop = 2.0 * m * n * k / 1e9;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, stream);
  GpuTimer timer;
  // warmup
  cublasStatus_t ret =
      cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                  (Btype_ *)B.device_data(), k, (Atype_ *)A.device_data(), k,
                  &beta, (Ctype_ *)C.device_data(), n);
  timer.start(stream);
  for (int iter = 0; iter < repeat; iter++) {
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                (Btype_ *)B.device_data(), k, (Atype_ *)A.device_data(), k,
                &beta, (Ctype_ *)C.device_data(), n);
  }
  timer.stop();
  float duration_ms = timer.elapsed_millis() / repeat;
  float tflops = gflop / duration_ms;
  printf("cublas ref: %f tflops, %f ms latency\n", tflops, duration_ms);

  if (ret != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Got cublas error at : " << __LINE__ << std::endl;
  }
  cublasDestroy(handle);
}


template <typename Accumtype, 
          typename Atype, typename ALayout, typename Btype, typename BLayout,
          typename Ctype, typename CLayout> // cvt cutlass::half_t to half
inline void
cublas_gemmExTN_ref(cutlass::HostTensor<Atype, ALayout> const &A, // row-major
                    cutlass::HostTensor<Btype, BLayout> const &B, // col-major
                    cutlass::HostTensor<Ctype, CLayout> &C,
                    int repeat = 1000, cudaStream_t stream = nullptr) {
                    
  Ctype alpha = static_cast<Ctype>(1);
  Ctype beta = static_cast<Ctype>(0);
  int m = A.extent().row();
  int n = B.extent().row();
  int k = A.extent().column();

  cudaDataType compute_type = CUDA_R_32F;
  cudaDataType indata_type = CUDA_R_32F;
  cudaDataType outdata_type = CUDA_R_32F;
  if constexpr (cute::is_same_v<cutlass::half_t, Atype>)
    indata_type = CUDA_R_16F;
  else if constexpr (cute::is_same_v<cutlass::bfloat16_t, Atype>)
    indata_type = CUDA_R_16BF;

  if constexpr (cute::is_same_v<cutlass::half_t, Ctype>)
    outdata_type = CUDA_R_16F; 
  else if constexpr (cute::is_same_v<cutlass::bfloat16_t, Ctype>)
    outdata_type = CUDA_R_16BF;

  if constexpr (cute::is_same_v<cutlass::half_t, Accumtype>) {
    compute_type = CUDA_R_16F;
  }

  printf("cublas indata_type: %d, outdata_type %d, compute_type: %d.\n", indata_type, outdata_type, compute_type);

  float gflop = 2.0 * m * n * k / 1e9;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, stream);
  GpuTimer timer;
  // warmup
  // cublasStatus_t ret =
  //     cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
  //                 (Btype_ *)B.device_data(), k, (Atype_ *)A.device_data(), k,
  //                 &beta, (Ctype_ *)C.device_data(), n);

  // cublas 如果accum是float，则输出也需要是float。
  // 不支持fp16/bf16输入且accum为float的情况下，输出为fp16/bf16.
  cublasStatus_t ret = cublasGemmEx(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                B.device_data(), indata_type, k,
                A.device_data(), indata_type, k,
                &beta,
                C.device_data(), outdata_type, n,
                compute_type,               // 计算精度
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                
  timer.start(stream);
  for (int iter = 0; iter < repeat; iter++) {
    // cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
    //             (Btype_ *)B.device_data(), k, (Atype_ *)A.device_data(), k,
    //             &beta, (Ctype_ *)C.device_data(), n);
    cublasGemmEx(handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      n, m, k,
      &alpha,
      B.device_data(), indata_type, k,
      A.device_data(), indata_type, k,
      &beta,
      C.device_data(), outdata_type, n,
      compute_type,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
  timer.stop();
  float duration_ms = timer.elapsed_millis() / repeat;
  float tflops = gflop / duration_ms;
  printf("%-30s: %f tflops, %f ms latency\n", "[cublas ref]", tflops, duration_ms);

  if (ret != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Got cublas error at : " << __LINE__ << std::endl;
  }
  cublasDestroy(handle);
}

template <typename T>
void gen_rand_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}


template <class TA, class TB, class TC>
void gemm_host_ref(int M, int N, int K,
              TA const* A, int ldA,
              TB const* B, int ldB,
              TC      * C, int ldC) {
  float *nA = new float[M*K];
  float *nB = new float[N*K];
  for (size_t i = 0; i < M; i++) {
    for (size_t k = 0; k < K; k++) {
      nA[i*K+k] = static_cast<float>(A[i*K+k]);
    }
  }
  for (size_t j = 0; j < N; j++) {
    for (size_t k = 0; k < K; k++) {
      nB[j*K+k] = static_cast<float>(B[j*K+k]);
    }
  }
  printf("finish host cast.\n");
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float ctemp = 0.0f;
      for (size_t k = 0; k < K; k++) {
        ctemp += nA[i*K+k] * nB[j*K+k];
      }
      C[i*N+j] = static_cast<TC>(ctemp);
    }
  }
  printf("finish host compute.\n");
  delete[] nA;
  delete[] nB;
}

template <typename T>
__global__ static void gpu_compare_kernel(const T *x, const T *y, int n,
                                          float threshold, int *count,
                                          float *max_error) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  }

  float v0 = x[idx];
  float v1 = y[idx];
  // if (threadIdx.x == 0)
  //   printf("%f,", v0);
  float diff = fabs(v0 - v1);
  if (diff > threshold) {
    atomicAdd(count, 1);

    // for positive floating point, there int representation is in the same
    // order.
    int int_diff = *((int *)(&diff));
    atomicMax((int *)max_error, int_diff);
  }
}

template <typename T>
void gpu_compare(const T *x, const T *y, size_t n, float threshold = 1e-1) {
  int *num_count;
  float *max_error;
  cudaMalloc(&num_count, sizeof(int));
  cudaMalloc(&max_error, sizeof(float));
  cudaMemset(num_count, 0, sizeof(int));
  cudaMemset(max_error, 0, sizeof(float));

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  gpu_compare_kernel<<<grid, block>>>(x, y, n, threshold, num_count, max_error);
  int num = 0;
  float error = 0;
  cudaMemcpy(&num, num_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&error, max_error, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (num == 0) {
    printf_pass("check ok, max_error = %f\n", error);
  } else {
    float p = (100.f * num) / n;
    printf_fail("===============================\n");
    printf_fail("check fail: diff %.1f%% = %d/%d max_error = %f\n", p, num, n,
                error);
    printf_fail("===============================\n");
  }
  cudaFree(num_count);
  cudaFree(max_error);
}

template <typename T>
void cpu_cosine_similarity(T *x, T *y, size_t n, float threshold = 0.999) {

  double xy = 0.0f;
  double x_2 = 0.0f;
  double y_2 = 0.0f;
  for (size_t i = 0; i < n; i++) {
    auto xi = static_cast<float>(x[i]);
    auto yi = static_cast<float>(y[i]);
    xy += xi * yi;
    x_2 += xi * xi;
    y_2 += yi * yi;
  }
  // (A dot B) / (mod(A) * mod(B))
  float cos_similarity = xy / (std::sqrt(x_2 + 1e-5) * std::sqrt(y_2 + 1e-5));
  if (cos_similarity >= threshold && cos_similarity <= 1.0f) {
    printf_pass("check ok, cos_similarity = %f\n", cos_similarity);
  } else {
    printf_fail("check fail, cos_similarity = %f\n", cos_similarity);
  }
}

template <typename T>
float cosine_similarity_safe(T *x, T *y, size_t n, float threshold = 0.999) {
  // 1. 先做一轮扫描，找绝对值最大元素，用来缩放
  float max_a = 0.0f, max_b = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    max_a = std::max(max_a, std::fabs(x[i]));
    max_b = std::max(max_b, std::fabs(y[i]));
    // if (i%100 == 0)
    //   printf("data: %f, %f.\n", x[i], y[i]);
  }
  // 若其中一路全 0，直接返回 0 避免 0/0
  if (max_a == 0.0f || max_b == 0.0f) return 0.0f;

  // 2. 用较大者统一缩放，保证所有乘子 ≤ 1
  float scale = std::max(max_a, max_b);
  float dot = 0.0f, na = 0.0f, nb = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    // printf("(%f,%f), ", x[i], y[i]);
    float sa = x[i] / scale;
    float sb = y[i] / scale;
    dot += sa * sb;
    na  += sa * sa;
    nb  += sb * sb;
    // if (i%100 == 0)
    //   printf("data: %f, %f, %f, %f.\n", dot, na, nb, scale);
  }
  // printf("zdata: %f, %f, %f.\n", dot, na, nb);
  // 3. 计算余弦值
  float cos_similarity = dot / (std::sqrt(na) * std::sqrt(nb));
  if (cos_similarity >= threshold && cos_similarity <= 1.001f) {
    printf_pass("check ok, cos_similarity = %f\n", cos_similarity);
  } else {
    printf_fail("check fail, cos_similarity = %f\n", cos_similarity);
  }
}

template <typename Kernel>
inline float launch_with_timer(Kernel kernel, int repeat = 1000,
                               cudaStream_t stream = 0) {
  GpuTimer timer;
  timer.start(stream);
  for (int iter = 0; iter < repeat; iter++) {
    kernel();
  }
  timer.stop();
  return timer.elapsed_millis() / repeat;
}