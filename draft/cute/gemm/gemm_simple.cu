#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

#include "common.h"
#include "reference.h"

using namespace cute;

#define ENBALE_SIMPLE_NO_SMEM_V0

#ifdef ENBALE_SIMPLE_NO_SMEM_V0
#include "gemm_simple_no_smem.cuh"
#endif

int main(int argc, const char *argv[]) {
  using TYPE = cutlass::half_t; // float cutlass::half_t;

  int m = 4096, n = 2048, k = 8192;
  int warmup = 5, repeat = 100;

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
  printf("gemm m:%d n:%d k:%d\n", m, n, k);
  // init
  auto A_tensor = make_cutlass_rowmajor_tensor<TYPE>(m, k);
  auto B_tensor = make_cutlass_colmajor_tensor<TYPE>(k, n);
  auto C_tensor = make_cutlass_rowmajor_tensor<TYPE>(m, n);
  auto C_ref_tensor = make_cutlass_rowmajor_tensor<TYPE>(m, n);

  cutlass::reference::host::TensorFillRandomUniform(A_tensor.host_view(), 0, -2,
                                                    2);
  cutlass::reference::host::TensorFillRandomUniform(B_tensor.host_view(), 0, -2,
                                                    2);
  // cutlass::reference::host::TensorFill(A_tensor.host_view(), TYPE(1));
  // cutlass::reference::host::TensorFill(B_tensor.host_view(), TYPE(1));
  A_tensor.sync_device();
  B_tensor.sync_device();

  cublas_gemmExTN_ref<TYPE>(A_tensor, B_tensor, C_ref_tensor, repeat, stream);
  // gemm_host(m,n,k, A_tensor.host_data(), k, B_tensor.host_data(), k, C_ref_tensor.host_data(), n);
  C_ref_tensor.sync_host();


  auto run = [&](auto kernel_traits, std::string kernel_name = "kernel") {
    using KT = decltype(kernel_traits);
    dim3 block(size(typename KT::TiledMma{}));
    dim3 grid(ceil_div(m, KT::kTileM), ceil_div(n, KT::kTileN));
    auto kernel = [&] {
      #ifdef ENBALE_SIMPLE_NO_SMEM_V0
      gemm_no_smem::GemmSimpleKernel<KT><<<grid, block, 0, stream>>>(C_tensor.device_data(), A_tensor.device_data(), B_tensor.device_data(), m, n, k);
      #endif
    };
    for (int i = 0; i < warmup; i++) {
      kernel();
    }

    auto duration_ms = launch_with_timer(kernel, repeat, stream);
    auto flop = 2.0 * m * n * k;
    auto tflops = compute_tflops(flop, duration_ms);
    printf("%s: %f tflops, %f ms latency\n", kernel_name.c_str(), tflops, duration_ms);
    cudaDeviceSynchronize();
    C_tensor.sync_host();

    gpu_compare(C_tensor.host_data(), C_ref_tensor.host_data(),
                C_ref_tensor.capacity());
  };

#ifdef ENBALE_SIMPLE_NO_SMEM_V0
  run(gemm_no_smem::KernelTraits<TYPE, TYPE, TYPE, decltype(make_shape(_128{}, _128{}, _32{}))>{}, 
      "gemm_128*128*32_no_smem_simple");
#endif

  cudaStreamDestroy(stream);

  return 0;
}