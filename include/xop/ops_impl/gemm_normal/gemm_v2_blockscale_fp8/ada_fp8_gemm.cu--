

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "ada_blockwise_gemm.cuh"

// template <typename T>
// void check(T ptr, char const* const func, char const* const file, int const line)
// {
//     if (ptr)
//     {
//         throw TllmException(file, line,
//             fmtstr("[TensorRT-LLM][ERROR] CUDA runtime error in %s: %s", func, _cudaGetErrorEnum(ptr)).c_str());
//     }
// }

// #define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

inline int getMultiProcessorCount()
{
    int nSM{0};
    int deviceID{0};
    cudaGetDevice(&deviceID);
    cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID);
    return nSM;
}

static int kNumDeviceSMs = -1;
void gemm_dispatch_sm89(void* mat_a, void* mat_b, void* mat_d, float* scales_a, float* scales_b, uint32_t shape_m,
  uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, int num_device_sms = kNumDeviceSMs)
{
  if (num_device_sms < 0)
  {
      num_device_sms = kNumDeviceSMs = getMultiProcessorCount();
  }
  using ElementInput = cute::float_e4m3_t;
  using ElementOutput = cute::bfloat16_t;
  using ElementAccum = float;//cute::bfloat16_t;
  using ElementBlockScale = float;
  static constexpr int Stages = 4;
  using TileShape = cutlass::gemm::GemmShape<32, 128, 128>; // only support 32x128x128 for now
  using KT = ada_blockwise_gemm::AdaBlockwiseGemmTraits<ElementInput, ElementOutput, ElementAccum, ElementBlockScale,
      Stages, TileShape::kM, TileShape::kN, TileShape::kK>;
  using Gemm = ada_blockwise_gemm::AdaBlockwiseGemm<KT>;

  int gemm_m = shape_m;
  int gemm_n = shape_n;
  int gemm_k = shape_k;
  typename KT::Arguments args({gemm_m, gemm_n, gemm_k}, mat_a, mat_b, mat_d, scales_a, scales_b);

  Gemm gemm{};

  auto status = gemm.can_implement(args);
  // TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "This kernel is not supported. Last CUDA error is: %s",
  //     cutlassGetStatusString(status));

  status = gemm.initialize(args);
  // TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess,
  //     "Failed to initialize the CUTLASS kernel. Last CUDA error is: %s", cutlassGetStatusString(status));

  status = gemm.run(stream);
  // TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess,
  //     "Failed to run the CUTLASS kernel. Last CUDA error is: %s", cutlassGetStatusString(status));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void cuda_check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA_CHECK error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(val) cuda_check((val), #val, __FILE__, __LINE__)


struct GpuTimer {
  GpuTimer() {
      CUDA_CHECK(cudaEventCreate(&start_));
      CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
      CUDA_CHECK(cudaEventDestroy(start_));
      CUDA_CHECK(cudaEventDestroy(stop_));
  }
  void Start() {
      CUDA_CHECK(cudaEventRecord(start_, NULL));
  }
  void Stop() {
      CUDA_CHECK(cudaEventRecord(stop_, NULL));
  }
  float ElapsedMillis() {
      float elapsed;
      CUDA_CHECK(cudaEventSynchronize(stop_));
      CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_, stop_));
      return elapsed;
  }

  cudaEvent_t start_;
  cudaEvent_t stop_;
};

void GenFp8Matrix(const int height, const int width, cutlass::float_e4m3_t *mat) {
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> converter;
  for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
          // mat[i*width + j] = converter((float)(rand() % 20 - 10)); // int: -10 ~ 10
          mat[i*width + j] = converter((float)(1.0f));
      }
  }
}

int main(int argc, char const** argv) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  // A*Bt=C
  int M = 2048, K = 1920;
  int N = 4096;

  const int mem_size_a = sizeof(cutlass::float_e4m3_t) * M * K;
  const int mem_size_b = sizeof(cutlass::float_e4m3_t) * N * K;
  const int mem_size_c = sizeof(cutlass::bfloat16_t) * M * N;
  const int mem_size_scale_a = sizeof(float) * M * K/128;
  const int mem_size_scale_b = sizeof(float) * N/128 * K/128;

  cutlass::float_e4m3_t *h_a = (cutlass::float_e4m3_t *)malloc(mem_size_a);
  cutlass::float_e4m3_t *h_b = (cutlass::float_e4m3_t *)malloc(mem_size_b);
  cutlass::bfloat16_t *h_c = (cutlass::bfloat16_t *)malloc(mem_size_c);
  float *h_scale_a = (float *)malloc(mem_size_scale_a);
  float *h_scale_b = (float *)malloc(mem_size_scale_b);

  // Initialize 
  srand(time(NULL));
  GenFp8Matrix(M, K, h_a);
  GenFp8Matrix(N, K, h_b);
  for (int i=0; i<M; i++) {
    for (int j=0; j<K/128; j++) {
      h_scale_a[i*K/128 + j] = 1.0f;
    }
  }
  for (int i=0; i<N/128; i++) {
    for (int j=0; j<K/128; j++) {
      h_scale_b[i*K/128 + j] = 1.0f;
    }
  }
  cutlass::float_e4m3_t *d_a, *d_b;
  cutlass::bfloat16_t *d_c;
  float *d_scale_a, *d_scale_b;

  CUDA_CHECK(cudaMalloc((void **)&d_a, mem_size_a));
  CUDA_CHECK(cudaMalloc((void **)&d_b, mem_size_b));
  CUDA_CHECK(cudaMalloc((void **)&d_c, mem_size_c));
  CUDA_CHECK(cudaMalloc((void **)&d_scale_a, mem_size_scale_a));
  CUDA_CHECK(cudaMalloc((void **)&d_scale_b, mem_size_scale_b));


  GpuTimer gpu_timer;
  float total_ms = 0;
  // for (int i=0; i<1000; i++) {
    CUDA_CHECK(cudaMemcpy(d_a, h_a, mem_size_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, mem_size_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale_a, h_scale_a, mem_size_scale_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale_b, h_scale_b, mem_size_scale_b, cudaMemcpyHostToDevice));
    // scaleA[m, (k+127)//128], scaleB[(n+127)//128, (k+127)//128]
    
    gpu_timer.Start();
    cudaStream_t default_stream = cudaStreamDefault;
    gemm_dispatch_sm89(d_a, d_b, d_c, d_scale_a, d_scale_b, M, N, K, default_stream);
    gpu_timer.Stop();
    total_ms += gpu_timer.ElapsedMillis();

    CUDA_CHECK(cudaMemcpy(h_c, d_c, mem_size_c, cudaMemcpyDeviceToHost));    
  // }
  printf("total_ms: %f.\n", total_ms);

  cutlass::NumericConverter<float, cutlass::bfloat16_t> converter;
  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      printf("%f, ", converter(h_c[i*N+j]));
      // if(converter(h_c[i*K+j]) != 2048) {
      //   printf("e: %f, ", converter(h_c[i*K+j]));
      // }
    }
  }
  return 0;
}
