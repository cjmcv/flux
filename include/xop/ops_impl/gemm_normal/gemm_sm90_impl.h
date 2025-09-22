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

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;
template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag, class TileScheduler, class TileShape, class ClusterShape, 
          RasterOrderOptions RasterOrder, int Swizzle,
          class MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto,
          // Type of epilogue schedule to generate
          class EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto,
          // Number of pipeline stages to use
          class StageCountType = cutlass::gemm::collective::StageCountAuto,
          // Type of tile scheduler to use
          class TileSchedulerType = cutlass::gemm::PersistentScheduler>

class GemmSm90Impl : public GemmBase {
public:
  ////
  using ElementD            = ElementC;
  using LayoutD             = LayoutC;
  using ElementCompute      = float;                                          // Element type for epilogue computation
  using ElementScalar       = float;

  using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;
  static constexpr bool UseCustomEVT = false;

  // 16B alignment lets us use TMA
  static constexpr int AlignmentA = 16 / sizeof(ElementA);
  static constexpr int AlignmentB = 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  static_assert(not UseCustomEVT ||
    (cute::is_same_v<EpilogueScheduleType, cutlass::epilogue::TmaWarpSpecialized> ||
      cute::is_same_v<EpilogueScheduleType, cutlass::epilogue::TmaWarpSpecializedCooperative>),
    "Epilogue visitor trees are currently only supported by the TMA warp-specialized epilogue");
  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  // EVTs can be constructed by composing the fundamental load/store/compute visitor operations defined in include/cutlass/epilogue/fusion
  // For more complex examples of EVT construction please refer to include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp
  using CustomEVT =  // alpha * acc + beta * C
    cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::homogeneous_multiply_add, ElementD, ElementCompute, RoundStyle>, // beta * C + (alpha * acc)
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>, // beta
      cutlass::epilogue::fusion::Sm90SrcFetch<ElementC>, // C
      cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
        cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>, // alpha
        cutlass::epilogue::fusion::Sm90AccFetch // acc
      >
    >;

  // A predefined set of fusion operations (implemented with EVT) are supported by the TMA warp-specialized epilogue.
  // Users can select one of these operations by passing one of the tags defined in include/cutlass/epilogue/fusion/operations.hpp
  // to the CollectiveBuilder. This frees the user from having to compute additional parameters such as stage counts and copy atoms/layouts.
  // These tags also provide additional metadata that can be queried at compile time.
  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_1,cute::_1,cute::_1>,
      EpilogueTileType,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType,
      cute::conditional_t<UseCustomEVT, CustomEVT, DefaultOperation>
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      cute::Shape<cute::_128,cute::_128,cute::_64>, cute::Shape<cute::_2,cute::_1,cute::_1>,
      cute::conditional_t<cute::is_same_v<StageCountType, cutlass::gemm::collective::StageCountAuto>,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          StageCountType>,
      MainloopScheduleType
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutTagA = cutlass::gemm::detail::StrideToLayoutTagA_t<StrideA>;
  using LayoutTagB = cutlass::gemm::detail::StrideToLayoutTagB_t<StrideB>;
  using LayoutTagC = cutlass::gemm::detail::StrideToLayoutTagC_t<StrideC>;
  using LayoutTagD = cutlass::gemm::detail::StrideToLayoutTagC_t<StrideD>;

public:
  void initialize(RtArguments *args, void *fusion_args = nullptr, void *stream = nullptr) {
    RtBlockScaleFp8ArgumentsV3 *rt_args = dynamic_cast<RtBlockScaleFp8ArgumentsV3*>(args);

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
  typename Gemm::Arguments args_from_options(const RtBlockScaleFp8ArgumentsV3 *rt_args) {
    ProblemShapeType problem_size = ProblemShapeType{rt_args->m, rt_args->n, rt_args->k, rt_args->l};
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(rt_args->m, rt_args->k, rt_args->l));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(rt_args->n, rt_args->k, rt_args->l));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(rt_args->m, rt_args->n, rt_args->l));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(rt_args->m, rt_args->n, rt_args->l));
    
    // Change device_id to another value if you are running on a machine with multiple GPUs and wish
    // to use a GPU other than that with device ID 0.
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {(ElementA *)rt_args->ptr_A, stride_A, (ElementB *)rt_args->ptr_B, stride_B},
      {{}, // epilogue.thread
      (ElementC *)rt_args->ptr_C, stride_C, (ElementD *)rt_args->ptr_D, stride_D},
      hw_info
    };

    // Custom EVT fusions will have nested unnamed args, the structure of which
    // can be deduced from the type definition of the EVT.
    // Each node's arguments has the recursive structure of
    // {first_child_args, ..., last_child_args, op_args},
    // For more complex examples of EVT initialization please refer to
    // include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp
    if constexpr (UseCustomEVT) {
      arguments.epilogue.thread =
        {    // ternary op : beta * C + (alpha * acc)
          {{rt_args->beta}}, // leaf op+args : beta
          {},               // leaf op+args : C
          {                 // binary op : alpha * acc
            {{rt_args->alpha}}, // leaf op+args : alpha
            {},                // leaf op+args : acc
            {}              // binary args : multiplies
          },                // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
    // Pre-defined fusions will have flat, named args for user-friendlyness
    else {
      arguments.epilogue.thread.alpha = rt_args->alpha;
      arguments.epilogue.thread.beta = rt_args->beta;
    }

    return arguments;
  }

private:
  Gemm gemm_dev_;
};

} // namespace xop