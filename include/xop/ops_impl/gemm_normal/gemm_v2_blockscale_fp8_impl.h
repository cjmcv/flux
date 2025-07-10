#pragma once
#include "xop/ops_impl/global_resource.h"
#include "xop/ops_impl/args_util.h"

#include "gemm_v2_blockscale_fp8/ada_blockwise_gemm_device.cuh"

namespace xop {

template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag>
class GemmV2BlockScaleFp8Impl : public GemmBase {

  using ElementBlockScale = float;
  static constexpr int Stages = 4;
  using TileShape = cutlass::gemm::GemmShape<32, 128, 128>; // only support 32x128x128 for now
  using KT = ada_blockwise_gemm::AdaBlockwiseGemmTraits<ElementA, ElementC, ElementAccumulator, ElementBlockScale,
      Stages, TileShape::kM, TileShape::kN, TileShape::kK>;
  using Gemm = ada_blockwise_gemm::AdaBlockwiseGemm<KT>;

public:
  typename KT::Arguments args_from_options(RtBlockScaleFp8ArgumentsV3 *rt_args) {
    cutlass::gemm::GemmCoord problem_size = {rt_args->m, rt_args->n, rt_args->k};
    return typename KT::Arguments (
      problem_size, 
      rt_args->ptr_A, 
      rt_args->ptr_B, 
      rt_args->ptr_D, 
      (float const* )rt_args->d_blockscale_A, 
      (float const* )rt_args->d_blockscale_B);
  }

  inline int getMultiProcessorCount() {
      int nSM{0};
      int deviceID{0};
      cudaGetDevice(&deviceID);
      cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID);
      return nSM;
  }
  

  void initialize(RtArguments *args, void *stream = nullptr) {
    RtBlockScaleFp8ArgumentsV3 *rt_args = dynamic_cast<RtBlockScaleFp8ArgumentsV3*>(args);    
    static int num_device_sms = -1;
    if (num_device_sms < 0) {
        num_device_sms = getMultiProcessorCount();
    }

    gemm_dev_ = Gemm();

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    auto arguments = args_from_options(rt_args);
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