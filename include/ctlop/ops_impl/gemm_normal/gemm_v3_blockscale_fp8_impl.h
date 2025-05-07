#pragma once
#include "ctlop/ops_impl/global_resource.h"
#include "ctlop/ops_impl/args_util.h"

namespace ctlop {

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;
template <class ElementA, class ElementB, class ElementC,
          class LayoutA, class LayoutB, class LayoutC,
          class ClusterShape, 
          RasterOrderOptions RasterOrder, int Swizzle>
class GemmBlockScaleFp8Impl : public GemmBase {
public:
  ////
  using         ElementD    = ElementC;
  using         LayoutD     = LayoutC;
  // Auxiliary matrix configuration and other fusion types
  using         ElementAux   = ElementC;
  using         LayoutAux    = LayoutC;
  using         ElementAmax  = float;
  using         ElementBias  = float;

  // Core kernel configurations
  using ElementAccumulator  = float;                                          // Element type for internal accumulation
  using ElementBlockScale   = float;                                          // Element type for blockscaling during accumulation
  using ElementCompute      = float;                                          // Element type for epilogue computation
  using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
  using TileShape           = cute::Shape<cute::_128, cute::_128, cute::_128>;                           // Threadblock-level tile size
  ////

  using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum<>;
  using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecializedCooperative;
  
  using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;
  using FusionOperation     = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltActAmaxAux<
      LayoutAux, cutlass::epilogue::thread::ReLU, ElementD, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementC>;
  
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,  // Shape of the threadblocks in a cluster
      EpilogueTileType,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 128 / cutlass::sizeof_bits<ElementC>::value,
      ElementD, LayoutD, 128 / cutlass::sizeof_bits<ElementD>::value,
      EpilogueSchedule,
      FusionOperation
    >::CollectiveOp;
  
  using CollectiveMainloopWithBlockWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, 128 / cutlass::sizeof_bits<ElementA>::value,
      ElementB, LayoutB, 128 / cutlass::sizeof_bits<ElementB>::value,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
      >,
      KernelSchedule
    >::CollectiveOp;
  
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>, // Indicates ProblemShape
      CollectiveMainloopWithBlockWiseScaling,
      CollectiveEpilogue
  >;
  
  // CORE
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  //
  using ElementScalar     = typename Gemm::EpilogueOutputOp::ElementScalar;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using StrideAux = StrideD;

public:
  void initialize(RtArguments &args, void *stream = nullptr) {
    RtBlockScaleFp8ArgumentsV3& rt_args = dynamic_cast<RtBlockScaleFp8ArgumentsV3&>(args);
    static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
      "ElementAccumulator and ElementBlockScale should be same datatype");

    // Instantiate CUTLASS kernel depending on templates
    gemm_dev_ = Gemm();

    // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
    auto arguments = args_from_options(rt_args);

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    void *workspace_ptr = GlobalBuffer::instance().ResizeBufferIfNeeded(workspace_size);

    // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm_dev_.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm_dev_.initialize(arguments, workspace_ptr));
  }

  void run(void *stream = nullptr) {
    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUTLASS_CHECK(gemm_dev_.run(cu_stream));
  }

private:
  typename Gemm::Arguments args_from_options(const RtBlockScaleFp8ArgumentsV3 &rt_args)
  {
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(rt_args.m, rt_args.k, rt_args.l));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(rt_args.n, rt_args.k, rt_args.l));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(rt_args.m, rt_args.n, rt_args.l));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(rt_args.m, rt_args.n, rt_args.l));
    StrideAux stride_aux = stride_D;

    // Note : This value has to match the KernelSchedule::ScalePromotionInterval
    // Else kernel will fail can_implement() check
    // Deprecation Notice : We plan to remove this params member in an upcoming release
    // Users can safely delete this line from their code, since the default is already 4
    unsigned int mma_promotion_interval = 4;

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {rt_args.m, rt_args.n, rt_args.k, rt_args.l},
      {(ElementA *)rt_args.ptr_A,
      stride_A,
      (ElementB *)rt_args.ptr_B,
      stride_B,
      mma_promotion_interval,
      (ElementBlockScale *)rt_args.d_blockscale_A, // blockscale_tensor_A.device_data(),
      (ElementBlockScale *)rt_args.d_blockscale_B, // blockscale_tensor_B.device_data()
      },
      {
        {}, // epilogue.thread
        (ElementC *)rt_args.ptr_C, stride_C,
        (ElementD *)rt_args.ptr_D, stride_D
      }
    };

    auto &fusion_args = arguments.epilogue.thread;
    fusion_args.alpha = rt_args.alpha;
    fusion_args.beta = rt_args.beta;
    fusion_args.alpha_ptr = nullptr; // (ElementScalar *)rt_args.d_scalar_alpha; // scalar_alpha.device_data();
    fusion_args.beta_ptr = nullptr; //(ElementScalar *)rt_args.d_scalar_beta; // scalar_beta.device_data();
    fusion_args.scale_a = rt_args.scale_a;
    fusion_args.scale_b = rt_args.scale_b;
    fusion_args.scale_c = rt_args.scale_c;
    fusion_args.scale_a_ptr = nullptr; //(ElementScalar *)rt_args.d_scale_A; // scale_A.device_data();
    fusion_args.scale_b_ptr = nullptr; //(ElementScalar *)rt_args.d_scale_B; // scale_B.device_data();
    fusion_args.scale_c_ptr = nullptr; //(ElementScalar *)rt_args.d_scale_C; // scale_C.device_data();

    // ignored if tensor types are not fp8
    fusion_args.scale_d = rt_args.scale_d;
    fusion_args.scale_aux = rt_args.scale_aux;
    fusion_args.scale_d_ptr = nullptr; //(ElementScalar *)rt_args.d_scale_D; //scale_D.device_data();
    fusion_args.scale_aux_ptr = nullptr; //(ElementScalar *)rt_args.d_scale_aux; // scale_aux.device_data();

    // leaving/setting these as nullptr disables the fusion at runtime
    fusion_args.bias_ptr = nullptr;

    if (rt_args.save_aux) {
      fusion_args.aux_ptr = (ElementAux *)rt_args.d_tensor_aux; // tensor_aux.device_data();
      fusion_args.dAux = stride_aux;
      if (rt_args.save_amax) {
        fusion_args.amax_aux_ptr = (ElementScalar *)rt_args.d_abs_max_aux; // abs_max_aux.device_data();
      }
    }

    if (rt_args.save_amax) {
      fusion_args.amax_D_ptr = (ElementScalar *)rt_args.d_abs_max_D; // abs_max_D.device_data();
    }

    arguments.scheduler.raster_order = RasterOrder;
    // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
    arguments.scheduler.max_swizzle_size = Swizzle;

    return arguments;
  }

private:
  Gemm gemm_dev_;
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