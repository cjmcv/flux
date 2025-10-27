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

template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag, class TileShape, class ClusterShape, 
          class MainloopScheduleType, class EpilogueScheduleType, class TileScheduler>

class GemmW4A16Sm90Impl : public GemmBase {
public:
  using MmaType = cutlass::bfloat16_t;
  using QuantType = cutlass::int4b_t;
  constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

  // A matrix configuration
  using         ElementA    = MmaType;                                        // Element type for A matrix operand
  using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
  constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         ElementB    = QuantType;                                      // Element type for B matrix operand
  using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
  constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // This example manually swaps and transposes, so keep transpose of input layouts
  using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
  using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

  // Define the CuTe layout for reoredered quantized tensor B
  // LayoutAtomQuant places values that will be read by the same thread in contiguous locations in global memory.
  // It specifies the reordering within a single warp's fragment
  //using ValueShuffle = Layout<_1>;                          // no value reordering
  using ValueShuffle = Layout<Shape<_2,_4>, Stride<_4,_1>>; // order [0,2,4,6,1,3,5,7]
  int constexpr NumShuffleAtoms = 1;
  using MmaAtomShape = Layout<Shape<_1,Int<NumShuffleAtoms>>>;
  using LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<MmaType, MmaAtomShape, ValueShuffle>());
  using LayoutB_Reordered = decltype(cute::tile_to_shape(LayoutAtomQuant{}, Layout<Shape<int,int,int>, StrideB>{}));

  using ElementScale = MmaType;
  using ElementZero = ElementScale;
  using LayoutScale = cutlass::layout::RowMajor;

  // C/D matrix configuration
  using         ElementC    = cutlass::bfloat16_t;                                // Element type for C and D matrix operands
  using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
  constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

  // D matrix configuration
  using         ElementD    = ElementC;
  using         LayoutD     = LayoutC;
  constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Core kernel configurations
  using ElementAccumulator  = float;                                          // Element type for internal accumulation
  using ElementCompute      = float;                                          // Element type for epilogue computation
  using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
  using TileShape           = Shape<_128,_128,cute::Int<TileShapeK>>;         // Threadblock-level tile size
  using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster
  using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  // Kernel to launch based on the default setting in the Collective Builder 
  using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      EpilogueTileType,
      ElementAccumulator, ElementAccumulator,
      // Transpose layout of D here since we use explicit swap + transpose
      // the void type for C tells the builder to allocate 0 smem for the C matrix.
      // We can enable this if beta == 0 by changing ElementC to void below.
      ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type, AlignmentC,
      ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
      EpilogueSchedule // This is the only epi supporting the required swap + transpose.
    >::CollectiveOp;

  // ScaleOnlyShuffled
  using CollectiveMainloopScaleOnlyShuffled = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      cute::tuple<ElementB, ElementScale>, LayoutB_Reordered, AlignmentB,
      ElementA, LayoutA_Transpose, AlignmentA,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
      >,
      KernelSchedule
    >::CollectiveOp;

  using GemmKernelScaleOnlyShuffled = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, // Indicates ProblemShape
      CollectiveMainloopScaleOnlyShuffled,
      CollectiveEpilogue
    >;

  using GemmScaleOnlyShuffled = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnlyShuffled>;
  ////////////////

  using StrideC = typename GemmKernelScaleOnly::StrideC;
  using StrideD = typename GemmKernelScaleOnly::StrideD;

public:
  void initialize(RtArgumentsBase *args, void *fusion_args = nullptr, void *stream = nullptr) {
    RtBlockScaleArguments *rt_args = static_cast<RtBlockScaleArguments*>(args);

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
  typename Gemm::Arguments args_from_options(const RtBlockScaleArguments *rt_args) {
    ProblemShapeType problem_size = ProblemShapeType{rt_args->m, rt_args->n, rt_args->k, rt_args->l};
    auto shape_B = cute::make_shape(rt_args->n, rt_args->k, rt_args->l);
    int const scale_k = cutlass::ceil_div(rt_args->k, rt_args->g);
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(rt_args->m, rt_args->k, rt_args->l));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    // Reverse stride here due to swap and transpose
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(rt_args->n, rt_args->m, rt_args->l));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(rt_args->n, rt_args->m, rt_args->l));
    
    auto layout_B = make_layout(shape_B, stride_B);
    if (true) { // shuffle
      // Repeat the reorder layout atom to tile the whole tensor shape 
      layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
      cutlass::reorder_tensor(block_B.get(), layout_B, layout_B_reordered);
    }

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