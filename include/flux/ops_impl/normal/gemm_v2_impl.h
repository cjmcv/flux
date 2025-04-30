#pragma once
#include "gemm_base.h"
#include "params_util.h"

namespace xop {

template <class ElementA, class ElementB, class ElementC, class ElementAccumulator,
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag, 
          class ThreadblockShape, class WarpShape, class InstructionShape,
          class ThreadBlockSwizzle, int NumStages, int SplitKFactor, int AvailSms>
class GemmPureV2Impl : public GemmBase  {

  // Epilogue output operator
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC,               // Element type for C and D matrix operands
      128 / cutlass::sizeof_bits<ElementC>::value, // Memory access granularity of C and D matrix in units of elements
      ElementAccumulator,     // Element type from internal accumaccumulation
      ElementAccumulator>;    // Data type used to compute linear combination

  // Classic data-parallel device GEMM implementation type
  using DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      ThreadBlockSwizzle, // cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<> / cutlass::gemm::threadblock::ThreadblockSwizzleStreamK
      NumStages,
      128 / cutlass::sizeof_bits<ElementA>::value,  // AlignmentA, Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
      128 / cutlass::sizeof_bits<ElementB>::value>; // AlignmentB

public:
  void initialize(RtParams &rt_params) {
    gemm_dev_ = DeviceGemmBasic();
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    ImplHelper<LayoutA, LayoutB, LayoutC> helper(rt_params.m, rt_params.n, rt_params.k);
    rt_params.stride_a = helper.get_stride_a();
    rt_params.stride_b = helper.get_stride_b();
    rt_params.stride_c = helper.get_stride_c();
    rt_params.stride_d = helper.get_stride_c();
    auto arguments = args_from_options(rt_params);
    size_t workspace_size = DeviceGemmBasic::get_workspace_size(arguments);
  
    // Allocate workspace memory
    if (workspace_.size() < workspace_size)
      workspace_.reallocate(workspace_size);
  
    // Check the problem size is supported or not
    CUTLASS_CHECK(gemm_dev_.can_implement(arguments));
  
    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm_dev_.initialize(arguments, workspace_.get()));
  }

  void run() {
    CUTLASS_CHECK(gemm_dev_());
  }

private:
  // avail_sms: Number of device SMs to use is unlimited
  //         1: Set loadbalancing width to 1 SM (no load balancing)
  //        -1: Reset loadbalancing width to unspecified SMs (i.e., the number of device SMs)
  typename DeviceGemmBasic::Arguments args_from_options(const RtParams &rt_params) {
    cutlass::gemm::GemmCoord problem_size = {rt_params.m, rt_params.n, rt_params.k};
    if constexpr (cute::is_same_v<ThreadBlockSwizzle, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>) {  
      return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        problem_size,                     // problem_size
        SplitKFactor,                   // batch count / splitk slices
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
    else {
      return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        problem_size,                     // problem_size
        SplitKFactor,                   // batch count / splitk slices
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
        rt_params.stride_d,              // stride_d
        AvailSms);                                // avail_sms
    }
  }

private:
  DeviceGemmBasic gemm_dev_;
  cutlass::device_memory::allocation<uint8_t> workspace_;
};

} // namespace xop