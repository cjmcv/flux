#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

#include "common.h"
#include "reference.h"

using namespace cute;

#define ENBALE_SIMPLE_NO_SMEM_V0
// #define ENBALE_GEMM_V2

#ifdef ENBALE_SIMPLE_NO_SMEM_V0
#include "gemm_simple_no_smem.cuh"
#endif

#ifdef ENBALE_GEMM_V2
#include "gemm_multi_stage.cuh"
#endif

template <typename ABCtype, typename Accumtype>
void TestGemm(cudaStream_t stream, int warmup, int repeat, int m, int n, int k) {
  using TYPE = ABCtype; // cutlass::half_t;
  using ACCUM_TYPE = Accumtype; // float;
  printf("gemm m:%d n:%d k:%d - io %s - acc %s\n", m, n, k, get_type_name<TYPE>(), get_type_name<ACCUM_TYPE>());
  // init
  cutlass::HostTensor<TYPE, cutlass::layout::RowMajor> A_tensor(cutlass::MatrixCoord({m, k}));
  cutlass::HostTensor<TYPE, cutlass::layout::RowMajor> B_tensor(cutlass::MatrixCoord({n, k}));
  cutlass::HostTensor<TYPE, cutlass::layout::RowMajor> C_tensor(cutlass::MatrixCoord({m, n}));
  cutlass::HostTensor<TYPE, cutlass::layout::RowMajor> C_ref_tensor(cutlass::MatrixCoord({m, n}));

  // auto A_tensor = make_cutlass_rowmajor_tensor<TYPE>(m, k);
  // auto B_tensor = make_cutlass_colmajor_tensor<TYPE>(k, n);
  // auto C_tensor = make_cutlass_rowmajor_tensor<TYPE>(m, n);
  // auto C_ref_tensor = make_cutlass_rowmajor_tensor<TYPE>(m, n);
  cutlass::reference::host::TensorFillRandomUniform(A_tensor.host_view(), 0, -1, 1);
  cutlass::reference::host::TensorFillRandomUniform(B_tensor.host_view(), 0, -1, 1);
  // cutlass::reference::host::TensorFill(A_tensor.host_view(), TYPE(1));
  // cutlass::reference::host::TensorFill(B_tensor.host_view(), TYPE(1));
  A_tensor.sync_device();
  B_tensor.sync_device();
  cublas_gemmExTN_ref<ACCUM_TYPE>(A_tensor, B_tensor, C_ref_tensor, repeat, stream);
  // gemm_host(m,n,k, A_tensor.host_data(), k, B_tensor.host_data(), k, C_ref_tensor.host_data(), n);
  C_ref_tensor.sync_host();

  auto run = [&](auto kernel_traits, std::string kernel_name = "kernel") {
    using KT = decltype(kernel_traits);
    dim3 block(size(typename KT::TiledMma{}));
    dim3 grid(ceil_div(n, KT::kTileN), ceil_div(m, KT::kTileM));
    auto kernel = [&] {
      #ifdef ENBALE_SIMPLE_NO_SMEM_V0
      gemm_no_smem::GemmSimpleKernel<KT><<<grid, block, 0, stream>>>(C_tensor.device_data(), A_tensor.device_data(), B_tensor.device_data(), m, n, k);
      #endif
      #ifdef ENBALE_GEMM_V2
      // TODO: ¼Óacc16
      int shm_size = KT::kShmSize;
      cudaFuncSetAttribute(gemm_v2::gemm_multi_stage<KT>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
      gemm_v2::gemm_multi_stage<KT><<<grid, block, shm_size, stream>>>(C_tensor.device_data(), A_tensor.device_data(), B_tensor.device_data(), m, n, k);
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

    // gpu_compare(C_tensor.device_data(), C_ref_tensor.device_data(), C_ref_tensor.capacity());
    cosine_similarity_safe(C_tensor.host_data(), C_ref_tensor.host_data(), C_ref_tensor.capacity());
  };

#ifdef ENBALE_SIMPLE_NO_SMEM_V0
  run(gemm_no_smem::KernelTraits<TYPE, TYPE, ACCUM_TYPE, decltype(make_shape(_128{}, _128{}, _32{}))>{}, 
      "gemm_128*128*32_no_smem_simple");
#endif
#ifdef ENBALE_GEMM_V2
  run(gemm_v2::KernelTraits<TYPE>{}, 
      "gemm_128*128*32_no_smem_simple");
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

  // TestGemm<cutlass::half_t, float>(stream, warmup, repeat, m, n, k);
  TestGemm<float, float>(stream, warmup, repeat, m, n, k);

  cudaStreamDestroy(stream);
  return 0;
}