#pragma once
#include "xop/ops_impl/global_resource.h"
#include "xop/ops_impl/args_util.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "gemm_v2_blockscale_fp8/ada_blockwise_gemm.cuh"


namespace xop {

template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag>
class GemmV2BlockScaleFp8Impl : public GemmBase {

  // using ElementA = cute::float_e4m3_t;
  // using ElementC = cute::bfloat16_t;
  // using ElementAccumulator = float;//cute::bfloat16_t;
  using ElementBlockScale = float;
  static constexpr int Stages = 4;
  using TileShape = cutlass::gemm::GemmShape<32, 128, 128>; // only support 32x128x128 for now
  using KT = ada_blockwise_gemm::AdaBlockwiseGemmTraits<ElementA, ElementC, ElementAccumulator, ElementBlockScale,
      Stages, TileShape::kM, TileShape::kN, TileShape::kK>;
  using Gemm = ada_blockwise_gemm::AdaBlockwiseGemm<KT>;

  // using EpilogueOpSimt = cutlass::epilogue::thread::LinearCombination<
  //     ElementC,               // Element type for C and D matrix operands
  //     1,                      // Memory access granularity of C and D matrix in units of elements
  //     ElementAccumulator,     // Element type from internal accumaccumulation
  //     ElementAccumulator>;    // Data type used to compute linear combination

  // using DeviceGemmSimt = cutlass::gemm::device::GemmUniversal<
  //   ElementA, LayoutA,
  //   ElementB, LayoutB,
  //   ElementC, LayoutC,
  //   ElementAccumulator,
  //   cutlass::arch::OpClassSimt, //OperatorClass,
  //   ArchTag, // ArchTag,cutlass::arch::Sm89
  //   ThreadblockShape, //cutlass::gemm::GemmShape<64, 64, 4>,
  //   WarpShape, //cutlass::gemm::GemmShape<32, 16, 4>,
  //   cutlass::gemm::GemmShape<1, 1, 1>,
  //   EpilogueOpSimt,
  //   cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  //   2>;

public:
  // typename DeviceGemmSimt::Arguments args_from_options(RtArgumentsV2 *rt_args) {
  //   cutlass::gemm::GemmCoord problem_size = {rt_args->m, rt_args->n, rt_args->k};
  //   return typename DeviceGemmSimt::Arguments(
  //     cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
  //     problem_size,                     // problem_size
  //     SplitKFactor,                             // batch count / splitk slices
  //     {                                         // epilogue parameters
  //       ElementAccumulator(rt_args->alpha),
  //       ElementAccumulator(rt_args->beta)
  //     },
  //     rt_args->ptr_A,                   // ptr_A
  //     rt_args->ptr_B,                   // ptr_B
  //     rt_args->ptr_C,                   // ptr_C
  //     rt_args->ptr_D,                   // ptr_D
  //     problem_size.mk().product(),      // batch_stride_A
  //     problem_size.nk().product(),      // batch_stride_B
  //     problem_size.mn().product(),      // batch_stride_C
  //     problem_size.mn().product(),      // batch_stride_D
  //     rt_args->stride_a,              // stride_a
  //     rt_args->stride_b,              // stride_b
  //     rt_args->stride_c,              // stride_c
  //     rt_args->stride_d);             // stride_d
  // }

  inline int getMultiProcessorCount() {
      int nSM{0};
      int deviceID{0};
      cudaGetDevice(&deviceID);
      cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID);
      return nSM;
  }
  

  void initialize(RtArguments *args, void *stream = nullptr) {
    // RtArgumentsV2 *rt_args = dynamic_cast<RtArgumentsV2*>(args);
    RtBlockScaleFp8ArgumentsV3 *rt_args = dynamic_cast<RtBlockScaleFp8ArgumentsV3*>(args);
    // gemm_dev_ = DeviceGemmSimt();
    // gemm_dev_ = Gemm();
    
    static int num_device_sms = -1;
    if (num_device_sms < 0) {
        num_device_sms = getMultiProcessorCount();
    }

    gemm_dev_ = Gemm();

    typename KT::Arguments arguments({rt_args->m, rt_args->n, rt_args->k}, rt_args->ptr_A, rt_args->ptr_B, rt_args->ptr_D, (float const* )rt_args->d_blockscale_A, (float const* )rt_args->d_blockscale_B);

    // // Using the arguments, query for extra workspace required for matrix multiplication computation
    // // auto arguments = args_from_options(rt_args);
    // size_t workspace_size = Gemm::get_workspace_size(arguments);
  
    // // Allocate workspace memory
    // void *workspace_ptr = GlobalBuffer::instance().ResizeDeviceBufferIfNeeded(workspace_size);
  
    // Check the problem size is supported or not
    CUTLASS_CHECK(gemm_dev_.can_implement(arguments));
  
    // Initialize CUTLASS kernel with arguments and workspace pointer
    // auto cu_stream = static_cast<cudaStream_t>(stream);
    CUTLASS_CHECK(gemm_dev_.initialize(arguments)); // , workspace_ptr, cu_stream
  }

  void run(void *stream = nullptr) {
    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUTLASS_CHECK(gemm_dev_.run(cu_stream));
  }

private:
  Gemm gemm_dev_;
};

} // namespace xop