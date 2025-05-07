#pragma once
#include "ctlop/ops_impl/global_resource.h"
#include "ctlop/ops_impl/args_util.h"

namespace ctlop {

template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, class LayoutB, class LayoutC, 
          class ArchTag, 
          class ThreadblockShape, class WarpShape, 
          class ThreadBlockSwizzle, int NumStages, int SplitKFactor>
class GemmPureV2SimtImpl : public GemmBase {

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
    EpilogueOpSimt,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;

public:
  typename DeviceGemmSimt::Arguments args_from_options(RtArgumentsV2 *rt_args) {
    cutlass::gemm::GemmCoord problem_size = {rt_args->m, rt_args->n, rt_args->k};
    return typename DeviceGemmSimt::Arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
      problem_size,                     // problem_size
      SplitKFactor,                             // batch count / splitk slices
      {                                         // epilogue parameters
        ElementAccumulator(rt_args->alpha),
        ElementAccumulator(rt_args->beta)
      },
      rt_args->ptr_A,                   // ptr_A
      rt_args->ptr_B,                   // ptr_B
      rt_args->ptr_C,                   // ptr_C
      rt_args->ptr_D,                   // ptr_D
      problem_size.mk().product(),      // batch_stride_A
      problem_size.nk().product(),      // batch_stride_B
      problem_size.mn().product(),      // batch_stride_C
      problem_size.mn().product(),      // batch_stride_D
      rt_args->stride_a,              // stride_a
      rt_args->stride_b,              // stride_b
      rt_args->stride_c,              // stride_c
      rt_args->stride_d);             // stride_d
  }

  void initialize(RtArguments *args, void *stream = nullptr) {
    RtArgumentsV2 *rt_args = dynamic_cast<RtArgumentsV2*>(args);
    gemm_dev_ = DeviceGemmSimt();

    ImplHelper<LayoutA, LayoutB, LayoutC> helper(rt_args->m, rt_args->n, rt_args->k);
    rt_args->stride_a = helper.get_stride_a();
    rt_args->stride_b = helper.get_stride_b();
    rt_args->stride_c = helper.get_stride_c();
    rt_args->stride_d = helper.get_stride_c();

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    auto arguments = args_from_options(rt_args);
    size_t workspace_size = DeviceGemmSimt::get_workspace_size(arguments);
  
    // Allocate workspace memory
    void *workspace_ptr = GlobalBuffer::instance().ResizeBufferIfNeeded(workspace_size);
  
    // Check the problem size is supported or not
    CUTLASS_CHECK(gemm_dev_.can_implement(arguments));
  
    // Initialize CUTLASS kernel with arguments and workspace pointer
    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUTLASS_CHECK(gemm_dev_.initialize(arguments, workspace_ptr, cu_stream));
  }

  void run(void *stream = nullptr) {
    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUTLASS_CHECK(gemm_dev_.run(cu_stream));
  }

private:
  DeviceGemmSimt gemm_dev_;
};

} // namespace ctlop