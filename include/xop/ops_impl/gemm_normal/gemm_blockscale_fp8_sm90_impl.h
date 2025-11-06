#pragma once
#include "xop/ops_impl/global_resource.h"
#include "xop/ops_impl/common_cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
namespace xop {

#define BLOCKSCALE_FP8_FUSED_COMPLEX 0
using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;
template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag, class TileScheduler, class TileShape, class ClusterShape, 
          RasterOrderOptions RasterOrder, int Swizzle>
class GemmBlockScaleFp8Sm90Impl : public GemmBase {
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
  using ElementBlockScale   = float;                                          // Element type for blockscaling during accumulation
  using ElementCompute      = float;                                          // Element type for epilogue computation
  using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag                          // Threadblock-level tile size
  ////

  using ScaleConfig   = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;

  using LayoutSFA     = decltype(ScaleConfig::deduce_layoutSFA());    // Layout type for SFA matrix operand
  using LayoutSFB     = decltype(ScaleConfig::deduce_layoutSFB());    // Layout type for SFB matrix operand
  using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
  using EpilogueScheduleType     = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType     = cutlass::epilogue::collective::EpilogueTileAuto;

#ifdef BLOCKSCALE_FP8_FUSED_COMPLEX
  using FusionOperation     = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltActAmaxAux<
      LayoutAux, cutlass::epilogue::thread::Identity, ElementD, ElementCompute, ElementAux, ElementAmax, ElementBias, ElementC>; // cutlass::epilogue::thread::ReLU
#else  
  using FusionOperation     = cutlass::epilogue::fusion::LinearCombination<ElementC, ElementCompute>;
#endif

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,  // Shape of the threadblocks in a cluster
      EpilogueTileType,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, 128 / cutlass::sizeof_bits<ElementC>::value,
      ElementD, LayoutD, 128 / cutlass::sizeof_bits<ElementD>::value,
      EpilogueScheduleType,
      FusionOperation
    >::CollectiveOp;
  
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, 128 / cutlass::sizeof_bits<ElementA>::value,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, 128 / cutlass::sizeof_bits<ElementB>::value,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    MainloopScheduleType
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler
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
  void initialize(RtArgumentsBase *args, void *fusion_args = nullptr, void *stream = nullptr) {
    RtBlockScaleArguments *rt_args = static_cast<RtBlockScaleArguments*>(args);

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
    void *workspace_ptr = GlobalBuffer::instance().ResizeDeviceBufferIfNeeded(workspace_size);

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
  typename Gemm::Arguments args_from_options(const RtBlockScaleArguments *rt_args)
  {
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(rt_args->m, rt_args->k, rt_args->l));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(rt_args->n, rt_args->k, rt_args->l));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(rt_args->m, rt_args->n, rt_args->l));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(rt_args->m, rt_args->n, rt_args->l));
    StrideAux stride_aux = stride_D;
    // The m in layout_SFA is already padded by function gemm_blockscale_fp8_scale_a_preprocess.
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape((rt_args->m+3)/4*4, rt_args->n, rt_args->k, rt_args->l));
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(rt_args->m, rt_args->n, rt_args->k, rt_args->l));
    
    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {rt_args->m, rt_args->n, rt_args->k, rt_args->l},
      {(ElementA *)rt_args->ptr_A,
      stride_A,
      (ElementB *)rt_args->ptr_B,
      stride_B,
      (ElementBlockScale *)rt_args->ptr_blockscale_A, // blockscale_tensor_A.device_data(),
      layout_SFA,
      (ElementBlockScale *)rt_args->ptr_blockscale_B, // blockscale_tensor_B.device_data()
      layout_SFB,
      },
      {
        {}, // epilogue.thread
        (ElementC *)rt_args->ptr_C, stride_C,
        (ElementD *)rt_args->ptr_D, stride_D
      }
    };

    #ifdef BLOCKSCALE_FP8_FUSED_COMPLEX
    auto &fusion_args = arguments.epilogue.thread;
    fusion_args.alpha = rt_args->alpha;
    fusion_args.beta = rt_args->beta;
    fusion_args.alpha_ptr = nullptr; // (ElementScalar *)rt_args->d_scalar_alpha; // scalar_alpha.device_data();
    fusion_args.beta_ptr = nullptr; //(ElementScalar *)rt_args->d_scalar_beta; // scalar_beta.device_data();
    fusion_args.scale_a = rt_args->scale_a;
    fusion_args.scale_b = rt_args->scale_b;
    fusion_args.scale_c = rt_args->scale_c;
    fusion_args.scale_a_ptr = nullptr; //(ElementScalar *)rt_args->d_scale_A; // scale_A.device_data();
    fusion_args.scale_b_ptr = nullptr; //(ElementScalar *)rt_args->d_scale_B; // scale_B.device_data();
    fusion_args.scale_c_ptr = nullptr; //(ElementScalar *)rt_args->d_scale_C; // scale_C.device_data();

    // ignored if tensor types are not fp8
    fusion_args.scale_d = rt_args->scale_d;
    fusion_args.scale_aux = rt_args->scale_aux;
    fusion_args.scale_d_ptr = nullptr; //(ElementScalar *)rt_args->d_scale_D; //scale_D.device_data();
    fusion_args.scale_aux_ptr = nullptr; //(ElementScalar *)rt_args->d_scale_aux; // scale_aux.device_data();

    // leaving/setting these as nullptr disables the fusion at runtime
    fusion_args.bias_ptr = nullptr;

    if (rt_args->save_aux) {
      fusion_args.aux_ptr = (ElementAux *)rt_args->d_tensor_aux; // tensor_aux.device_data();
      fusion_args.dAux = stride_aux;
      if (rt_args->save_amax) {
        fusion_args.amax_aux_ptr = (ElementScalar *)rt_args->d_abs_max_aux; // abs_max_aux.device_data();
      }
    }

    if (rt_args->save_amax) {
      fusion_args.amax_D_ptr = (ElementScalar *)rt_args->d_abs_max_D; // abs_max_D.device_data();
    }
    #endif

    arguments.scheduler.raster_order = RasterOrder;
    // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
    arguments.scheduler.max_swizzle_size = Swizzle;

    return arguments;
  }

private:
  Gemm gemm_dev_;
};

} // namespace xop