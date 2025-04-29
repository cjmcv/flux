#pragma once
#include "gemm_base.h"
#include "params_util.h"

namespace xop {

template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, class LayoutB, class LayoutC, 
          class ArchTag, int SplitKFactor,
          class ThreadblockShape, class WarpShape>
class GemmPureV2SimtDevice : public GemmBase {

  using EpilogueOpSimt = cutlass::epilogue::thread::LinearCombination<
      ElementC,               // Element type for C and D matrix operands
      1,                      // Memory access granularity of C and D matrix in units of elements
      ElementAccumulator,     // Element type from internal accumaccumulation
      ElementAccumulator>;    // Data type used to compute linear combination

  using DeviceGemmSimt = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt, //OperatorClass,
    ArchTag, // ArchTag,cutlass::arch::Sm89
    ThreadblockShape, //cutlass::gemm::GemmShape<64, 64, 4>,
    WarpShape, //cutlass::gemm::GemmShape<32, 16, 4>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueOpSimt>;

public:
  typename DeviceGemmSimt::Arguments args_from_options(const RtParams &rt_params) {
    cutlass::gemm::GemmCoord problem_size = {rt_params.m, rt_params.n, rt_params.k};
    return typename DeviceGemmSimt::Arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
      problem_size,                     // problem_size
      SplitKFactor,                             // batch count / splitk slices
      {                                         // epilogue parameters
        ElementAccumulator(rt_params.alpha),
        ElementAccumulator(rt_params.beta)
      },
      rt_params.ptr_A,                   // ptr_A
      rt_params.ptr_B,                   // ptr_B
      rt_params.ptr_C,                   // ptr_C
      rt_params.ptr_D,                   // ptr_D
      problem_size.mk().product(),      // batch_stride_A
      problem_size.nk().product(),      // batch_stride_B
      problem_size.mn().product(),      // batch_stride_C
      problem_size.mn().product(),      // batch_stride_D
      rt_params.stride_a,              // stride_a
      rt_params.stride_b,              // stride_b
      rt_params.stride_c,              // stride_c
      rt_params.stride_d);             // stride_d
  }

  void initialize(RtParams &rt_params) {
    gemm_dev_ = DeviceGemmSimt();

    ImplHelper<LayoutA, LayoutB, LayoutC> helper(rt_params.m, rt_params.n, rt_params.k);
    rt_params.stride_a = helper.get_stride_a();
    rt_params.stride_b = helper.get_stride_b();
    rt_params.stride_c = helper.get_stride_c();
    rt_params.stride_d = helper.get_stride_c();

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    auto arguments = args_from_options(rt_params);
    size_t workspace_size = DeviceGemmSimt::get_workspace_size(arguments);
  
    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  
    // Check the problem size is supported or not
    CUTLASS_CHECK(gemm_dev_.can_implement(arguments));
  
    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm_dev_.initialize(arguments, workspace.get()));
  }

  void run() {
    CUTLASS_CHECK(gemm_dev_());
  }

private:
  DeviceGemmSimt gemm_dev_;
};

} // namespace xop