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
#include "cutlass/util/mixed_dtype_utils.hpp" // cutlass::compute_memory_reordering_atom
#include "cutlass/util/packed_stride.hpp"

namespace xop {

template <class ElementA, class ElementC, 
          class ArchTag, class TileShapetemp0, class ClusterShape, 
          class KernelSchedule, class EpilogueSchedule>

class GemmW4A16Sm90Impl : public GemmBase {
public:
  using MmaType = ElementA;
  using QuantType = cutlass::int4b_t;
  static constexpr int TileShapeK = 128 * 8 / cutlass::sizeof_bits<MmaType>::value;

  // A matrix configuration
  // using         ElementA    = MmaType;                                        // Element type for A matrix operand
  using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
  static constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         ElementB    = QuantType;                                      // Element type for B matrix operand
  using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
  static constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // This example manually swaps and transposes, so keep transpose of input layouts
  using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
  using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

  // Define the CuTe layout for reoredered quantized tensor B
  // LayoutAtomQuant places values that will be read by the same thread in contiguous locations in global memory.
  // It specifies the reordering within a single warp's fragment
  //using ValueShuffle = Layout<_1>;                          // no value reordering
  using ValueShuffle = cutlass::Layout<cutlass::Shape<cute::_2,cute::_4>, cutlass::Stride<cute::_4,cute::_1>>; // order [0,2,4,6,1,3,5,7]
  static constexpr int NumShuffleAtoms = 1;
  using MmaAtomShape = cutlass::Layout<cutlass::Shape<cute::_1,cute::Int<NumShuffleAtoms>>>;
  using LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<MmaType, MmaAtomShape, ValueShuffle>());
  using LayoutB_Reordered = decltype(cute::tile_to_shape(LayoutAtomQuant{}, cutlass::Layout<cutlass::Shape<int,int,int>, StrideB>{}));

  using ElementScale = MmaType;
  using ElementZero = ElementScale;
  using LayoutScale = cutlass::layout::RowMajor;

  // C/D matrix configuration
  // using         ElementC    = cutlass::bfloat16_t;                                // Element type for C and D matrix operands
  using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
  static constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

  // D matrix configuration
  using         ElementD    = ElementC;
  using         LayoutD     = LayoutC;
  static constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Core kernel configurations
  using ElementAccumulator  = float;                                          // Element type for internal accumulation
  using ElementCompute      = float;                                          // Element type for epilogue computation
  // using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
  using TileShape           = cutlass::Shape<cute::_128,cute::_128,cute::Int<TileShapeK>>;         // Threadblock-level tile size
  // using ClusterShape        = cutlass::Shape<cute::_1,cute::_1,cute::_1>;                                // Shape of the threadblocks in a cluster
  // using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  // Kernel to launch based on the default setting in the Collective Builder 
  // using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
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
      cute::Shape<int,int,int,int>, // Indicates ProblemShape
      CollectiveMainloopScaleOnlyShuffled,
      CollectiveEpilogue
    >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnlyShuffled>; // GemmScaleOnlyShuffled
  
  using StrideS = typename CollectiveMainloopScaleOnlyShuffled::StrideScale;
  using StrideC = typename GemmKernelScaleOnlyShuffled::StrideC;
  using StrideD = typename GemmKernelScaleOnlyShuffled::StrideD;
  ////////////////
  
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
    const int m = rt_args->m;
    const int n = rt_args->n;
    const int k = rt_args->k;
    const int l = rt_args->l;
    const int g = rt_args->g;

    // ProblemShapeType problem_size = ProblemShapeType{m, n, k, l};
    auto shape_B = cute::make_shape(n, k, l);
    int const scale_k = cutlass::ceil_div(k, g);
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    StrideS stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, scale_k, l));
    // Reverse stride here due to swap and transpose
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, l));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, l));
    
    auto layout_B = make_layout(shape_B, stride_B);

    LayoutB_Reordered layout_B_reordered;
    if (true) { // shuffle
      // Repeat the reorder layout atom to tile the whole tensor shape 
      layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
      cutlass::reorder_tensor((ElementB *)rt_args->ptr_B, layout_B, layout_B_reordered);
    }

    using Args = typename Gemm::Arguments;
    auto&& dB = [&]() {
      return layout_B_reordered; // offline swizzling is enabled.
      // if constexpr (cute::is_same_v<Gemm, GemmScaleOnlyShuffled> ||
      //               cute::is_same_v<Gemm, GemmScaleWithZeroPointShuffled>) {
      //   // offline swizzling is enabled.
      //   return layout_B_reordered;
      // }
      // else {
      //   return stride_B;
      // }
    }();

    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {n, m, k, l},
      {(ElementB *)rt_args->ptr_B, dB, (ElementA *)rt_args->ptr_A, stride_A, (ElementScale *)rt_args->ptr_blockscale_B, stride_S, g},
      {{rt_args->alpha, rt_args->beta}, (ElementC *)rt_args->ptr_C, stride_C, (ElementD *)rt_args->ptr_D, stride_D}
    };

    return arguments;
  }

private:
  Gemm gemm_dev_;
};

} // namespace xop