#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

#include "common.h"
#include "reference.h"

using namespace cute;

#define ENBALE_SIMPLE_NO_SMEM_V0
#define ENBALE_GEMM_V2

#ifdef ENBALE_SIMPLE_NO_SMEM_V0
#include "gemm_simple_no_smem.cuh"
#endif

#ifdef ENBALE_GEMM_V2
#include "gemm_multi_stage.cuh"
#endif

template <typename ElementType, typename OutElementType, typename AccumElementType>
void TestGemm(cudaStream_t stream, int warmup, int repeat, int m, int n, int k) {
  using ElementInput = ElementType; // cutlass::half_t;
  using ElementOutput = OutElementType;
  using ElementAccumulator = AccumElementType; // float;
  printf("gemm m:%d n:%d k:%d - %s, %s, <%s>\n", m, n, k, get_type_name<ElementInput>(), get_type_name<ElementOutput>(), get_type_name<ElementAccumulator>());
  // init
  cutlass::HostTensor<ElementInput, cutlass::layout::RowMajor> A_tensor(cutlass::MatrixCoord({m, k}));
  cutlass::HostTensor<ElementInput, cutlass::layout::RowMajor> B_tensor(cutlass::MatrixCoord({n, k}));
  cutlass::HostTensor<ElementOutput, cutlass::layout::RowMajor> C_tensor(cutlass::MatrixCoord({m, n}));
  cutlass::HostTensor<ElementOutput, cutlass::layout::RowMajor> C_ref_tensor(cutlass::MatrixCoord({m, n}));

  cutlass::reference::host::TensorFillRandomUniform(A_tensor.host_view(), 0, -1, 1);
  cutlass::reference::host::TensorFillRandomUniform(B_tensor.host_view(), 0, -1, 1);
  // cutlass::reference::host::TensorFill(A_tensor.host_view(), ElementInput(1));
  // cutlass::reference::host::TensorFill(B_tensor.host_view(), ElementInput(1));
  A_tensor.sync_device();
  B_tensor.sync_device();

  // cublas 如果accum是float，则输出也需要是float。
  // 不支持fp16/bf16输入且accum为float的情况下，输出为fp16/bf16.
  if constexpr (cute::is_same_v<float, ElementAccumulator> && !cute::is_same_v<float, ElementOutput>) {
    cutlass::HostTensor<float, cutlass::layout::RowMajor> C_ref_tensor_fp32(cutlass::MatrixCoord({m, n}));
    cublas_gemmExTN_ref<ElementAccumulator>(A_tensor, B_tensor, C_ref_tensor_fp32, repeat, stream);
    C_ref_tensor_fp32.sync_host();
    
    cutlass::NumericConverter<ElementOutput, float> converter;
    for (int i = 0; i < C_ref_tensor.size(); ++i) {
      C_ref_tensor.host_data()[i] = converter(C_ref_tensor_fp32.host_data()[i]);
      // printf("%f, ", C_ref_tensor_fp32.host_data()[i]);
    }
  }
  else {
    cublas_gemmExTN_ref<ElementAccumulator>(A_tensor, B_tensor, C_ref_tensor, repeat, stream);
    C_ref_tensor.sync_host();
  }
  // gemm_host(m,n,k, A_tensor.host_data(), k, B_tensor.host_data(), k, C_ref_tensor.host_data(), n);

  auto perf_epi = [&](auto kernel, std::string kernel_name) {
    for (int i = 0; i < warmup; i++) {
      kernel();
    }

    auto duration_ms = launch_with_timer(kernel, repeat, stream);
    auto flop = 2.0 * m * n * k;
    auto tflops = compute_tflops(flop, duration_ms);
    printf("%-30s: %f tflops, %f ms latency - ", kernel_name.c_str(), tflops, duration_ms);
    cudaDeviceSynchronize();
    C_tensor.sync_host();

    // gpu_compare(C_tensor.device_data(), C_ref_tensor.device_data(), C_ref_tensor.capacity());
    cosine_similarity_safe(C_tensor.host_data(), C_ref_tensor.host_data(), C_ref_tensor.capacity());
  };

#ifdef ENBALE_SIMPLE_NO_SMEM_V0
  auto run_v0 = [&](auto kernel_traits, std::string kernel_name = "kernel") {
    using KT = decltype(kernel_traits);
    dim3 block(size(typename KT::TiledMma{}));
    dim3 grid(ceil_div(m, KT::kTileM), ceil_div(n, KT::kTileN));
    auto kernel = [&] {
      gemm_no_smem::GemmSimpleKernel<KT><<<grid, block, 0, stream>>>(C_tensor.device_data(), A_tensor.device_data(), B_tensor.device_data(), m, n, k);        
    };
    perf_epi(kernel, kernel_name);
  };
  run_v0(gemm_no_smem::KernelTraits<ElementInput, ElementOutput, ElementAccumulator, decltype(make_shape(_128{}, _128{}, _32{}))>{}, 
      "[gemm_128*128*32_no_smem]");
#endif

#ifdef ENBALE_GEMM_V2
  auto run_v1 = [&](auto kernel_traits, std::string kernel_name = "kernel") {
    using KT = decltype(kernel_traits);
    dim3 block(size(typename KT::TiledMma{}));
    dim3 grid(ceil_div(m, KT::kTileM), ceil_div(n, KT::kTileN));
    auto kernel = [&] {
      int shm_size = KT::kShmSize;
      cudaFuncSetAttribute(gemm_v2::gemm_multi_stage<KT>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
      gemm_v2::gemm_multi_stage<KT><<<grid, block, shm_size, stream>>>(C_tensor.device_data(), A_tensor.device_data(), B_tensor.device_data(), m, n, k);
    };
    perf_epi(kernel, kernel_name);
  };
  run_v1(gemm_v2::KernelTraits<ElementInput, ElementOutput, ElementAccumulator, decltype(make_shape(_128{}, _128{}, _32{}))>{}, 
      "[gemm_v2]");
#endif
}

int main(int argc, const char *argv[]) {
  int m = 2048, n = 1024, k = 1024;
  int warmup = 10, repeat = 100;

  if (argc > 1) {
    m = atoi(argv[1]);
  }
  if (argc > 2) {
    n = atoi(argv[2]);
  }
  if (argc > 3) {
    k = atoi(argv[3]);
  }
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  TestGemm<cutlass::half_t, cutlass::half_t, cutlass::half_t>(stream, warmup, repeat, m, n, k);
  TestGemm<cutlass::half_t, cutlass::half_t, float>(stream, warmup, repeat, m, n, k);
  TestGemm<cutlass::bfloat16_t, cutlass::bfloat16_t, float>(stream, warmup, repeat, m, n, k);

  // TestGemm<cutlass::half_t, float, float>(stream, warmup, repeat, m, n, k);
  // TestGemm<cutlass::bfloat16_t, float, float>(stream, warmup, repeat, m, n, k);

  cudaStreamDestroy(stream);
  return 0;
}