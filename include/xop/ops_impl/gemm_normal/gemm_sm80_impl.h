#pragma once
#include "xop/ops_impl/global_resource.h"
#include "xop/ops_impl/common_cutlass.h"
// #define ENABLE_EVT

#ifdef ENABLE_EVT // 搜索空?与不使用evt有区?，需要再?整
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#endif

namespace xop {

template <class ElementA, class ElementB, class ElementC, class ElementAccumulator,
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag, 
          class ThreadblockShape, class WarpShape, class InstructionShape,
          class ThreadBlockSwizzle, int NumStages, int SplitKFactor, int AvailSms>
class GemmSm80Impl : public GemmBase  {

#ifdef ENABLE_EVT
  using ElementCompute = ElementAccumulator;
  using ElementOutput = ElementC;

  static constexpr int EVTEpilogueStages = 1;  
  static constexpr int AlignmentC       = 128 / cutlass::sizeof_bits<ElementC>::value;
  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, 
    WarpShape, 
    ElementC, 
    AlignmentC, 
    EVTEpilogueStages
  >;

  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, ElementC,
      cute::Stride<cute::_0, cute::_1, int32_t>  // StrideMNL
  >;

  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;

  using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<
      Compute0, // 2
      Accum,    // 0
      Bias>;    // 1

  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t> // StrideMNL
  >;

  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<
      D,
      EVTCompute0>; // EVTCompute2

  using EVTKernelStreamK =
      typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA, LayoutA, cutlass::ComplexTransform::kNone, 128 / cutlass::sizeof_bits<ElementA>::value,
      ElementB, LayoutB, cutlass::ComplexTransform::kNone, 128 / cutlass::sizeof_bits<ElementB>::value,
      ElementC, LayoutC, AlignmentC,
      ElementAccumulator,
      ElementCompute,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EVTD,
      ThreadBlockSwizzle,
      NumStages,
      cutlass::arch::OpMultiplyAdd,
      EVTEpilogueStages
  >::GemmKernel;

  using DeviceGemmBasic = cutlass::gemm::device::GemmUniversalAdapter<EVTKernelStreamK>;

#else
  // Epilogue output operator
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC,               // Element type for C and D matrix operands
      128 / cutlass::sizeof_bits<ElementC>::value, // Memory access granularity of C and D matrix in units of elements
      ElementAccumulator,     // Element type from internal accumaccumulation
      ElementAccumulator,     // Data type used to compute linear combination
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;    // bias 

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
#endif

public:
  void initialize(RtArgumentsBase *args, void *fusion_args = nullptr, void *stream = nullptr) {
    RtArgumentsV2 *rt_args = static_cast<RtArgumentsV2*>(args);

    gemm_dev_ = DeviceGemmBasic();
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    ImplHelper<LayoutA, LayoutB, LayoutC> helper(rt_args->m, rt_args->n, rt_args->k);
    rt_args->stride_a = helper.get_stride_a();
    rt_args->stride_b = helper.get_stride_b();
    rt_args->stride_c = rt_args->C_s == -1 ? helper.get_stride_c() : 0;
    // printf("rt_args->stride_c: %d.\n", rt_args->stride_c);
    rt_args->stride_d = helper.get_stride_c();
    auto arguments = args_from_options(rt_args);
    size_t workspace_size = DeviceGemmBasic::get_workspace_size(arguments);

    // Allocate workspace memory
    auto cu_stream = static_cast<cudaStream_t>(stream);
    void *workspace_ptr = GlobalBuffer::instance().GetDeviceBuffer(kDevBufferPoolWorkspace, workspace_size);

    // Check the problem size is supported or not
    CUTLASS_CHECK(gemm_dev_.can_implement(arguments));
  
    // Initialize CUTLASS kernel with arguments and workspace pointer
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
  typename DeviceGemmBasic::Arguments args_from_options(const RtArgumentsV2 *rt_args) {
    cutlass::gemm::GemmCoord problem_size = {rt_args->m, rt_args->n, rt_args->k};
#ifdef ENABLE_EVT
    typename EVTD::Arguments callback_args{
      {
        {}, // Accum
        {(ElementC *)rt_args->ptr_C, ElementC(0), {cute::_0{}, cute::_1{}, int32_t(problem_size.n())}},                 // Bias
        {}  // Compute0
      },        // EVTCompute2
      {(ElementC *)rt_args->ptr_D, {problem_size.n(), cute::_1{}, problem_size.mn().product()}},                   // D
    };
    void *ptr_C = nullptr;
    void *ptr_D = nullptr;
    int batch_stride_C = 0;
    int batch_stride_D = 0;
    int stride_c = 0;
    int stride_d = 0;
#else
    auto callback_args = typename EpilogueOp::Params{
      ElementAccumulator(rt_args->alpha),
      ElementAccumulator(rt_args->beta)
    };
    void *ptr_C = rt_args->ptr_C;
    void *ptr_D = rt_args->ptr_D;
    int batch_stride_C = rt_args->stride_c == 0 ? rt_args->n : problem_size.mn().product();
    int batch_stride_D = problem_size.mn().product();
    int stride_c = rt_args->stride_c;
    int stride_d = rt_args->stride_d;
#endif
    
    if constexpr (cute::is_same_v<ThreadBlockSwizzle, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>) {  
      return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        problem_size,                     // problem_size
        SplitKFactor,                     // batch count / splitk slices
        callback_args,                    // epilogue parameters
        rt_args->ptr_A,                   // ptr_A
        rt_args->ptr_B,                   // ptr_B
        ptr_C,                            // ptr_C
        ptr_D,                            // ptr_D
        problem_size.mk().product(),      // batch_stride_A
        problem_size.nk().product(),      // batch_stride_B
        batch_stride_C,                   // batch_stride_C
        batch_stride_D,                   // batch_stride_D
        rt_args->stride_a,              // stride_a
        rt_args->stride_b,              // stride_b
        stride_c,              // stride_c
        stride_d);             // stride_d    
    }
    else {
      return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        problem_size,                     // problem_size
        SplitKFactor,                   // batch count / splitk slices
        callback_args,                    // epilogue parameters
        rt_args->ptr_A,                   // ptr_A
        rt_args->ptr_B,                   // ptr_B
        ptr_C,                            // ptr_C
        ptr_D,                            // ptr_D
        problem_size.mk().product(),      // batch_stride_A
        problem_size.nk().product(),      // batch_stride_B
        batch_stride_C,                   // batch_stride_C
        batch_stride_D,                   // batch_stride_D
        rt_args->stride_a,              // stride_a
        rt_args->stride_b,              // stride_b
        stride_c,              // stride_c
        stride_d,              // stride_d
        AvailSms);                                // avail_sms
    }
  }

private:
  DeviceGemmBasic gemm_dev_;
};

} // namespace xop


///////////////////////////////////////////////////////////////////////////
// // template
// template <class ElementA, class ElementB, class ElementC。。。>
// class GemmSm80Impl : public GemmBase  {

//   using EpilogueOp = cutlass::epilogue::thread::LinearCombination<。。。>;
//   using DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<。。。>;

// public:
//   void initialize(RtArgumentsBase *args, void *stream = nullptr) {
//     RtArgumentsV2 *rt_args = dynamic_cast<RtArgumentsV2*>(args);
//     gemm_dev_ = DeviceGemmBasic();
//     // Using the arguments, query for extra workspace required for matrix multiplication computation
//     ImplHelper<LayoutA, LayoutB, LayoutC> helper(rt_args->m, rt_args->n, rt_args->k);
//     。。。
//     auto arguments = args_from_options(rt_args);
//     size_t workspace_size = DeviceGemmBasic::get_workspace_size(arguments);

//     auto cu_stream = static_cast<cudaStream_t>(stream);
//     void *workspace_ptr = GlobalBuffer::instance().GetDeviceBuffer(kDevBufferPoolWorkspace, workspace_size);
//     CUTLASS_CHECK(gemm_dev_.can_implement(arguments));
//     CUTLASS_CHECK(gemm_dev_.initialize(arguments, workspace_ptr, cu_stream));
//   }

//   void run(void *stream = nullptr) {
//     auto cu_stream = static_cast<cudaStream_t>(stream);
//     CUTLASS_CHECK(gemm_dev_.run(cu_stream));
//   }

// private:
//   typename DeviceGemmBasic::Arguments args_from_options(const RtArgumentsBase *rt_args) {
//     return typename DeviceGemmBasic::Arguments(
//       ...
//     )}
//   }

// private:
//   DeviceGemmBasic gemm_dev_;
// };