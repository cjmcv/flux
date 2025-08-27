#pragma once
#include "xop/ops_impl/global_resource.h"
#include "xop/ops_impl/common_cutlass.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
// #include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "gemm_ar_v2/default_gemm_universal_with_visitor_rs.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "gemm_ar_v2/visitor_store_rs.hpp"
// #include "gemm_ar_v2/gemm_universal_rs.h"

////////////////////////////////
// #include "xop/../../src/ops/allreduce_normal/custom_all_reduce.h"
#include "xop/../../src/ops/allreduce_normal/custom_all_reduce.cuh"
///////////////////////////////

namespace xop {

template <class ElementA, class ElementB, class ElementC, class ElementAccumulator,
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag, 
          class ThreadblockShape, class WarpShape, class InstructionShape,
          class ThreadBlockSwizzle, int NumStages, int SplitKFactor, int AvailSms>
class GemmArV2Impl : public GemmBase  {
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

  // using C1 = cutlass::epilogue::threadblock::VisitorAuxLoad<
  //     OutputTileThreadMap, ElementC, 
  //     cute::Stride<int64_t, cute::_1, int64_t> // StrideMNL
  // >;

  // using C2 = cutlass::epilogue::threadblock::VisitorAuxLoad<
  //     OutputTileThreadMap, ElementC, 
  //     cute::Stride<int64_t, cute::_1, int64_t> // StrideMNL
  // >;

  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;

  using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<
      Compute0, // 2
      Accum,    // 0
      Bias>;    // 1
    
  // using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
  //     cutlass::plus, ElementCompute, ElementCompute,
  //     cutlass::FloatRoundStyle::round_to_nearest
  // >;

  // using EVTCompute1 = cutlass::epilogue::threadblock::Sm80EVT<
  //     Compute1,
  //     EVTCompute0,
  //     C1>;

  // using Compute2 = cutlass::epilogue::threadblock::VisitorCompute<
  //     cutlass::plus, ElementOutput, ElementCompute,
  //     cutlass::FloatRoundStyle::round_to_nearest
  // >;

  // using EVTCompute2 = cutlass::epilogue::threadblock::Sm80EVT<
  //     Compute2,
  //     EVTCompute1,
  //     C2>;

  using D = cutlass::epilogue::threadblock::VisitorAuxStoreRs<
      OutputTileThreadMap, ElementOutput, cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<int64_t, cute::_1, int64_t> // StrideMNL
  >;

  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<
      D,
      EVTCompute0>; // EVTCompute2

  using EVTKernelStreamK =
      typename cutlass::gemm::kernel::DefaultGemmWithVisitorRs<
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

  // // Epilogue output operator
  // using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
  //     ElementC,               // Element type for C and D matrix operands
  //     128 / cutlass::sizeof_bits<ElementC>::value, // Memory access granularity of C and D matrix in units of elements
  //     ElementAccumulator,     // Element type from internal accumaccumulation
  //     ElementAccumulator,     // Data type used to compute linear combination
  //     cutlass::epilogue::thread::ScaleType::NoBetaScaling>;    // bias 

  // // Classic data-parallel device GEMM implementation type
  // using DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<
  //     ElementA, LayoutA,
  //     ElementB, LayoutB,
  //     ElementC, LayoutC,
  //     ElementAccumulator,
  //     cutlass::arch::OpClassTensorOp,
  //     ArchTag,
  //     ThreadblockShape,
  //     WarpShape,
  //     InstructionShape,
  //     EpilogueOp,
  //     ThreadBlockSwizzle, // cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<> / cutlass::gemm::threadblock::ThreadblockSwizzleStreamK
  //     NumStages,
  //     128 / cutlass::sizeof_bits<ElementA>::value,  // AlignmentA, Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
  //     128 / cutlass::sizeof_bits<ElementB>::value>; // AlignmentB

public:
  void initialize(RtArguments *args, void *fusion_args = nullptr, void *stream = nullptr) {
    RtArgumentsV2 *rt_args = dynamic_cast<RtArgumentsV2*>(args);
    RtCommArguments *t_args = (RtCommArguments*)(fusion_args);
    comm_args_.handle = t_args->handle;
    comm_args_.reg_buffer = t_args->reg_buffer;
    comm_args_.reg_buffer_sz_bytes = t_args->reg_buffer_sz_bytes;
    comm_args_.gemm_out = t_args->gemm_out;

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
    void *workspace_ptr = GlobalBuffer::instance().ResizeDeviceBufferIfNeeded(workspace_size);

    // Check the problem size is supported or not
    CUTLASS_CHECK(gemm_dev_.can_implement(arguments));
  
    // Initialize CUTLASS kernel with arguments and workspace pointer
    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUTLASS_CHECK(gemm_dev_.initialize(arguments, workspace_ptr, cu_stream));

    ////
    output_ = rt_args->ptr_D;
    output_len_ = rt_args->m * rt_args->n;
  }

  void run(void *stream = nullptr) {
    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUTLASS_CHECK(gemm_dev_.run(cu_stream));

    // all_reduce(comm_args_.handle, comm_args_.gemm_out, rt_args->ptr_D, comm_args_.reg_buffer, comm_args_.reg_buffer_sz_bytes);

    {
      auto fa = reinterpret_cast<vllm::CustomAllreduce*>(comm_args_.handle);
      auto input = comm_args_.gemm_out;
      // auto output = rt_args->ptr_D;
      // const at::cuda::OptionalCUDAGuard device_guard(device_of(inp)); // 切换到inp所在的目标device
      // auto stream = c10::cuda::getCurrentCUDAStream().stream();
    
      // TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
      // TORCH_CHECK_EQ(inp.numel(), out.numel());
      // TORCH_CHECK(_is_weak_contiguous(out));
      // TORCH_CHECK(_is_weak_contiguous(inp));
      // auto input_size = inp.numel() * inp.element_size();

      auto input_size = output_len_ * sizeof(ElementOutput);
      auto reg_buffer = reinterpret_cast<void*>(comm_args_.reg_buffer);
      if (reg_buffer) {
        // TORCH_CHECK_LE(input_size, comm_args_.reg_buffer_sz_bytes); !! todo
        CUDA_CHECK(cudaMemcpyAsync(reg_buffer, input, input_size, cudaMemcpyDeviceToDevice, cu_stream));
      } else {
        reg_buffer = input;
      }

      if constexpr (cute::is_same_v<ElementOutput, float>) {
        fa->allreduce<float>(
          cu_stream, reinterpret_cast<float*>(reg_buffer), reinterpret_cast<float*>(output_), output_len_);
      }
      else if constexpr (cute::is_same_v<ElementOutput, cutlass::half_t>) {
        fa->allreduce<half>(
          cu_stream, reinterpret_cast<half*>(reg_buffer), reinterpret_cast<half*>(output_), output_len_);
      }
      else if constexpr (cute::is_same_v<ElementOutput, cutlass::bfloat16_t>) {
        fa->allreduce<nv_bfloat16>(
          cu_stream, reinterpret_cast<nv_bfloat16*>(reg_buffer), reinterpret_cast<nv_bfloat16*>(output_), output_len_);
      }
      else {
        throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
      }
    }
  }

private:
  // avail_sms: Number of device SMs to use is unlimited
  //         1: Set loadbalancing width to 1 SM (no load balancing)
  //        -1: Reset loadbalancing width to unspecified SMs (i.e., the number of device SMs)
  typename DeviceGemmBasic::Arguments args_from_options(const RtArgumentsV2 *rt_args) {
    cutlass::gemm::GemmCoord problem_size = {rt_args->m, rt_args->n, rt_args->k};
    int batch_stride_C = rt_args->stride_c == 0 ? rt_args->n : problem_size.mn().product();
    printf("hello you");
    typename EVTD::Arguments callback_args{
      {
        {}, // Accum
        {(ElementC *)rt_args->ptr_C, ElementC(0), {cute::_0{}, cute::_1{}, int32_t(problem_size.n())}},            // Bias
        {}  // Compute0
      },        // EVTCompute2
      {(ElementC *)comm_args_.gemm_out, {problem_size.n(), cute::_1{}, problem_size.mn().product()}},                   // D
    };   
    // typename EVTD::Arguments callback_args{
    //   {
    //     {
    //       {
    //         {}, // Accum
    //         {tensor_Vector.device_data(), ElementC(0), {_0{}, _1{}, int32_t(options.problem_size.n())}},                 // Bias
    //         {}  // Compute0
    //       },    // EVTCompute0
    //       {tensor_c1.device_data(), ElementC(0), {options.problem_size.n(), _1{}, options.problem_size.mn().product()}}, // C1
    //       {}    // Compute1
    //     },      // EVTCompute1
    //     {tensor_c2.device_data(), ElementC(0), {options.problem_size.n(), _1{}, options.problem_size.mn().product()}},   // C2
    //     {}      // Compute2
    //   },        // EVTCompute2
    //   {tensor_d.device_data(), {options.problem_size.n(), _1{}, options.problem_size.mn().product()}},                   // D
    // };   
    
    if constexpr (cute::is_same_v<ThreadBlockSwizzle, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>) {  
      return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        problem_size,                     // problem_size
        SplitKFactor,                   // batch count / splitk slices
        callback_args, 
        rt_args->ptr_A,                   // ptr_A
        rt_args->ptr_B,                   // ptr_B
        nullptr,                   // ptr_C
        nullptr,                   // ptr_D
        problem_size.mk().product(),      // batch_stride_A
        problem_size.nk().product(),      // batch_stride_B
        0,                   // batch_stride_C
        0,      // batch_stride_D
        rt_args->stride_a,              // stride_a
        rt_args->stride_b,              // stride_b
        0,              // stride_c
        0);             // stride_d    
    }
    else {
      return typename DeviceGemmBasic::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        problem_size,                     // problem_size
        SplitKFactor,                   // batch count / splitk slices
        callback_args, 
        rt_args->ptr_A,                   // ptr_A
        rt_args->ptr_B,                   // ptr_B
        nullptr,                   // ptr_C
        nullptr,                   // ptr_D
        problem_size.mk().product(),      // batch_stride_A
        problem_size.nk().product(),      // batch_stride_B
        0,                   // batch_stride_C
        0,      // batch_stride_D
        rt_args->stride_a,              // stride_a
        rt_args->stride_b,              // stride_b
        0,              // stride_c
        0,              // stride_d
        AvailSms);                                // avail_sms
    }
  }

private:
  DeviceGemmBasic gemm_dev_;

  RtCommArguments comm_args_;
  void *output_;
  int output_len_;
};

} // namespace xop