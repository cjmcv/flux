#pragma once
#include "ctlop/ops_impl/global_resource.h"
#include "ctlop/ops_impl/args_util.h"

namespace ctlop {

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
  void initialize(RtArguments &args, void *stream = nullptr) {
    RtArgumentsV2& rt_args = dynamic_cast<RtArgumentsV2&>(args);

    gemm_dev_ = DeviceGemmBasic();
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    ImplHelper<LayoutA, LayoutB, LayoutC> helper(rt_args.m, rt_args.n, rt_args.k);
    rt_args.stride_a = helper.get_stride_a();
    rt_args.stride_b = helper.get_stride_b();
    rt_args.stride_c = helper.get_stride_c();
    rt_args.stride_d = helper.get_stride_c();
    auto arguments = args_from_options(rt_args);
    size_t workspace_size = DeviceGemmBasic::get_workspace_size(arguments);

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
  // avail_sms: Number of device SMs to use is unlimited
  //         1: Set loadbalancing width to 1 SM (no load balancing)
  //        -1: Reset loadbalancing width to unspecified SMs (i.e., the number of device SMs)
  typename DeviceGemmBasic::Arguments args_from_options(const RtArgumentsV2 &rt_args) {
    cutlass::gemm::GemmCoord problem_size = {rt_args.m, rt_args.n, rt_args.k};
    if constexpr (cute::is_same_v<ThreadBlockSwizzle, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>) {  
      return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        problem_size,                     // problem_size
        SplitKFactor,                   // batch count / splitk slices
        {                                         // epilogue parameters
          ElementAccumulator(rt_args.alpha),
          ElementAccumulator(rt_args.beta)
        },
        rt_args.ptr_A,                   // ptr_A
        rt_args.ptr_B,                   // ptr_B
        rt_args.ptr_C,                   // ptr_C
        rt_args.ptr_D,                   // ptr_D
        problem_size.mk().product(),      // batch_stride_A
        problem_size.nk().product(),      // batch_stride_B
        problem_size.mn().product(),      // batch_stride_C
        problem_size.mn().product(),      // batch_stride_D
        rt_args.stride_a,              // stride_a
        rt_args.stride_b,              // stride_b
        rt_args.stride_c,              // stride_c
        rt_args.stride_d);             // stride_d    
    }
    else {
      return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        problem_size,                     // problem_size
        SplitKFactor,                   // batch count / splitk slices
        {                                         // epilogue parameters
          ElementAccumulator(rt_args.alpha),
          ElementAccumulator(rt_args.beta)
        },
        rt_args.ptr_A,                   // ptr_A
        rt_args.ptr_B,                   // ptr_B
        rt_args.ptr_C,                   // ptr_C
        rt_args.ptr_D,                   // ptr_D
        problem_size.mk().product(),      // batch_stride_A
        problem_size.nk().product(),      // batch_stride_B
        problem_size.mn().product(),      // batch_stride_C
        problem_size.mn().product(),      // batch_stride_D
        rt_args.stride_a,              // stride_a
        rt_args.stride_b,              // stride_b
        rt_args.stride_c,              // stride_c
        rt_args.stride_d,              // stride_d
        AvailSms);                                // avail_sms
    }
  }

private:
  DeviceGemmBasic gemm_dev_;
};

} // namespace ctlop


///////////////////////////////////////////////////////////////////////////
// // template
// template <class ElementA, class ElementB, class ElementC。。。>
// class GemmPureV2Impl : public GemmBase  {

//   using EpilogueOp = cutlass::epilogue::thread::LinearCombination<。。。>;
//   using DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<。。。>;

// public:
//   void initialize(RtArguments &rt_args, void *stream = nullptr) {
//     gemm_dev_ = DeviceGemmBasic();
//     // Using the arguments, query for extra workspace required for matrix multiplication computation
//     ImplHelper<LayoutA, LayoutB, LayoutC> helper(rt_args.m, rt_args.n, rt_args.k);
//     。。。
//     auto arguments = args_from_options(rt_args);
//     size_t workspace_size = DeviceGemmBasic::get_workspace_size(arguments);
  
//     void *workspace_ptr = GlobalBuffer::instance().ResizeBufferIfNeeded(workspace_size);
//     CUTLASS_CHECK(gemm_dev_.can_implement(arguments));
  
//     auto cu_stream = static_cast<cudaStream_t>(stream);
//     CUTLASS_CHECK(gemm_dev_.initialize(arguments, workspace_ptr, cu_stream));
//   }

//   void run(void *stream = nullptr) {
//     auto cu_stream = static_cast<cudaStream_t>(stream);
//     CUTLASS_CHECK(gemm_dev_.run(cu_stream));
//   }

// private:
//   typename DeviceGemmBasic::Arguments args_from_options(const RtArguments &rt_args) {
//     return typename DeviceGemmBasic::Arguments(
//       ...
//     )}
//   }

// private:
//   DeviceGemmBasic gemm_dev_;
// };