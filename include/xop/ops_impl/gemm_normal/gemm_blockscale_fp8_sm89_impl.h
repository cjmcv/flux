#pragma once
#include "xop/ops_impl/global_resource.h"
#include "xop/ops_impl/common_cutlass.h"

#include "gemm_blockscale_fp8_sm89/ada_blockwise_gemm_device.cuh"

namespace xop {
// 
template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag, 
          class TileShape, class PermShape, int NumStages>
class GemmBlockScaleFp8Sm89Impl : public GemmBase {

  using ElementBlockScale = float;
  using KT = traits::AdaBlockwiseGemmTraits<ElementA, ElementC, ElementAccumulator, ElementBlockScale,
                                            TileShape, PermShape, NumStages>;
  using Gemm = device::AdaBlockwiseGemm<KT>;

public:
  typename Gemm::Arguments args_from_options(RtBlockScaleFp8ArgumentsV3 *rt_args) {
    cutlass::gemm::GemmCoord problem_size = {rt_args->m, rt_args->n, rt_args->k};
    return typename Gemm::Arguments (
      problem_size, 
      rt_args->ptr_A, 
      rt_args->ptr_B, 
      rt_args->ptr_D, 
      (float const* )rt_args->d_blockscale_A, 
      (float const* )rt_args->d_blockscale_B);
  }
  
  void initialize(RtArgumentsBase *args, void *fusion_args = nullptr, void *stream = nullptr) {
    RtBlockScaleFp8ArgumentsV3 *rt_args = static_cast<RtBlockScaleFp8ArgumentsV3*>(args);
    gemm_dev_ = Gemm();

    auto arguments = args_from_options(rt_args);
    CUTLASS_CHECK(gemm_dev_.can_implement(arguments));
    CUTLASS_CHECK(gemm_dev_.initialize(arguments));
  }

  void run(void *stream = nullptr) {
    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUTLASS_CHECK(gemm_dev_.run(cu_stream));
  }

private:
  Gemm gemm_dev_;
};

} // namespace xop