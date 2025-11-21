#pragma once
#include "xop/ops_impl/global_resource.h"
#include "xop/ops_impl/common_cutlass.h"

#include "cutlass/gemm/kernel/gemv.h"
#include "cutlass/gemm/device/gemv.h"

namespace xop {

template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, 
          class ArchTag, 
          int ElementsPerAccess, int ThreadCount, int ThreadsPerRow>
class GemvSimtImpl : public GemmBase {

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    1,
    ElementAccumulator,
    ElementAccumulator>;

  using Gemv = cutlass::gemm::device::Gemv<
    cutlass::gemm::kernel::Gemv<
      ElementA,               // Element A
      LayoutA,                // Layout A
      ElementB,               // Element B
      ElementC,               // Element C
      ElementAccumulator,     // Element accumulator
      EpilogueOp,             // Output operator
      ElementsPerAccess      // Element access granularity
    >
  >;

  using TensorRefA = cutlass::TensorRef<ElementA, LayoutA>;

public:
  typename Gemv::Arguments args_from_options(RtArgumentsV2 *rt_args) {
    cutlass::MatrixCoord problem_size = {rt_args->m, rt_args->k};

    // // gemm: A[m=1,k] * B[n,k] = C[m,n]
    // // gemv: weight A[m,k] * input B[1,k] = C[1,m] row major
    // // so: B[n,k] => A[m,k]
    // void *t = rt_args->ptr_A;
    // rt_args->ptr_A = rt_args->ptr_B;
    // rt_args->ptr_B = t;
    // // if constexpr (cute::is_same_v<LayoutA, cutlass::layout::RowMajor>) {
    // //   rt_args->m = rt_args->n;
    // //   rt_args->n = 1;
    // // }
    // // else {
    // //   rt_args->m = rt_args->k;
    // //   rt_args->k = rt_args->n;
    // //   rt_args->n = 1;
    // // }
    // rt_args->m = rt_args->n;
    // rt_args->n = 1;

    LayoutA A_layout(rt_args->k);
    TensorRefA ref_a = cutlass::TensorRef((ElementA *)rt_args->ptr_A, A_layout);

    printf("mk: %d, %d, %d, %d\n", rt_args->m, rt_args->k, rt_args->l, ref_a.stride(0));
    return typename Gemv::Arguments {
      problem_size,
      rt_args->l,                       // batch_count,
      {rt_args->alpha, rt_args->beta},
      ref_a,
      (ElementB *)rt_args->ptr_B,
      (ElementC *)rt_args->ptr_C,
      (ElementC *)rt_args->ptr_D,
      rt_args->m * rt_args->k,         // batch_stride_A
      rt_args->k,                      // batch_stride_B
      rt_args->m,                      // batch_stride_C
      rt_args->m                       // batch_stride_D   
    };
  }

  void initialize(RtArgumentsBase *args, void *fusion_args = nullptr, void *stream = nullptr) {
    RtArgumentsV2 *rt_args = static_cast<RtArgumentsV2*>(args);
    gemv_ = Gemv();

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    auto arguments = args_from_options(rt_args);
    size_t workspace_size = Gemv::get_workspace_size(arguments);
  
    // Allocate workspace memory
    auto cu_stream = static_cast<cudaStream_t>(stream);
    void *workspace_ptr = GlobalBuffer::instance().GetDeviceBuffer(kDevBufferPoolWorkspace, workspace_size);
  
    // Check the problem size is supported or not
    CUTLASS_CHECK(gemv_.can_implement(arguments));
  
    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemv_.initialize(arguments, workspace_ptr, cu_stream));
  }

  void run(void *stream = nullptr) {
    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUTLASS_CHECK(gemv_.run(cu_stream));
  }

private:
  Gemv gemv_;
};

} // namespace xop