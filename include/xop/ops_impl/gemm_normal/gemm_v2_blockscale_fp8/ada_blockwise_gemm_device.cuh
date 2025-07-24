
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>

#include "ada_blockwise_gemm_kernel.cuh"
namespace xop {
namespace device {

using namespace cutlass;

template <typename KT>
struct AdaBlockwiseGemm {

  struct Arguments {
    cutlass::gemm::GemmCoord problem_size{};
    void const* ptr_a;
    void const* ptr_b;
    void* ptr_d;
    float const* ptr_scale_a;
    float const* ptr_scale_b;

    Arguments() {}
    Arguments(cutlass::gemm::GemmCoord problem_size_, void const* ptr_a_, void const* ptr_b_, void* ptr_d_,
            float const* ptr_scale_a_, float const* ptr_scale_b_)
            : problem_size(problem_size_)
            , ptr_a(ptr_a_)
            , ptr_b(ptr_b_)
            , ptr_d(ptr_d_)
            , ptr_scale_a(ptr_scale_a_)
            , ptr_scale_b(ptr_scale_b_) {}
  };

  using GemmKernel = kernel::AdaBlockwiseGemmKernel<KT>;

  static constexpr int kSmemSize = KT::kSmemSize;
  static constexpr int kThreadCount = KT::kThreadCount;

  /// Kernel parameters object
  Arguments params_;

  AdaBlockwiseGemm(): params_() {}

  Status can_implement(Arguments const& args) {
    // printf("kSmemSize: %d.\n", int(kSmemSize));
    if (kSmemSize > (48 << 10)) {
      cudaFuncSetAttribute(kernel::sm89_fp8_gemm_impl<GemmKernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
      auto result = cudaGetLastError();
        
      if (result != cudaSuccess) {
        printf("sm89 gemm kernel cannot launch: %s.\n", cudaGetErrorString(result));
        return Status::kInvalid;    
      }
    }

    if (args.problem_size.n() % KT::kTileN != 0) {
      CUTLASS_TRACE_HOST("  n:" << args.problem_size.n() << " % kTileN:" << KT::kTileN << " != 0");
      return Status::kInvalid;
    }

    if (args.problem_size.k() % KT::kTileK != 0) {
      CUTLASS_TRACE_HOST("  k:" << args.problem_size.k() << " % kTileK:" << KT::kTileK << " != 0");
      return Status::kInvalid;
    }

    return Status::kSuccess;
  }

  Status initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    auto ptr_a = reinterpret_cast<typename GemmKernel::ElementInput const*>(args.ptr_a);
    auto ptr_b = reinterpret_cast<typename GemmKernel::ElementInput const*>(args.ptr_b);
    auto ptr_d = reinterpret_cast<typename GemmKernel::ElementOutput*>(args.ptr_d);
    auto ptr_scale_a = reinterpret_cast<typename GemmKernel::ElementBlockScale const*>(args.ptr_scale_a);
    auto ptr_scale_b = reinterpret_cast<typename GemmKernel::ElementBlockScale const*>(args.ptr_scale_b);

    Arguments params(args.problem_size, ptr_a, ptr_b, ptr_d, ptr_scale_a, ptr_scale_b);
    params_ = params;
    return Status::kSuccess;
  }

  Status run(cudaStream_t stream = nullptr) {
    int shape_m = params_.problem_size.m();
    int shape_n = params_.problem_size.n();
    int shape_k = params_.problem_size.k();
    int grid_m = (shape_m + KT::kTileM - 1) / KT::kTileM;
    int grid_n = (shape_n + KT::kTileN - 1) / KT::kTileN;
    int grid_k = 1;
    dim3 grid = dim3(grid_m, grid_n, grid_k);
    dim3 block = dim3(kThreadCount, 1, 1);
    kernel::sm89_fp8_gemm_impl<GemmKernel>
        <<<grid, block, kSmemSize, stream>>>(shape_m, shape_n, shape_k, params_.ptr_a, params_.ptr_b, params_.ptr_d, params_.ptr_scale_a, params_.ptr_scale_b);

    return Status::kSuccess;
  }
};

} // namespace device
} // namespace xop
