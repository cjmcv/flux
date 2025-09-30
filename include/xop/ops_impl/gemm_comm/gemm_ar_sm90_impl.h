#pragma once
#include "xop/ops_impl/global_resource.h"
#include "xop/ops_impl/common_cutlass.h"
#include "xop/ops_impl/debug_util.h"

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

#include "gemm_ar_sm90/sm90_gemm_tma_warpspecialized_cooperative_ar.hpp"
#include "gemm_ar_sm90/sm90_gemm_tma_warpspecialized_pingpong_ar.hpp"
#include "gemm_ar_sm90/sm90_visitor_store_tma_warpspecialized_ar.hpp"
#include "gemm_ar_sm90/sm90_allreduce_utils.hpp"

#include "xop/../../src/ops/allreduce_normal/custom_all_reduce.cuh"

namespace xop {

struct AllReduceSm90Arguments {
  int world_size;
  int rank;
  int packed_array_num;
  
  void *reg_buffer;
  vllm::RankData* rank_data;
  vllm::RankSignals rank_signals;
  vllm::Signal *self_signal;
  
  void *output;
  uint8_t *aux_local_buffer;
  int aux_buffer_streamk_reduce_mark_step;   // Used to mark the tiles that require reduction in Stream-K
  int aux_buffer_reduce_arrival_step; // Used to count the number of arrivals of reduce epi
  size_t aux_local_size;
  virtual ~AllReduceSm90Arguments() {}
};

template <class ElementA, class ElementB, class ElementC, class ElementAccumulator, 
          class LayoutA, class LayoutB, class LayoutC,
          class ArchTag, class TileShape, class ClusterShape, 
          class MainloopScheduleType, class EpilogueScheduleType, class TileScheduler>

class GemmArSm90Impl : public GemmBase {
public:
  ////
  using ElementD            = ElementC;
  using LayoutD             = LayoutC;
  using ElementCompute      = float;                                          // Element type for epilogue computation
  using ElementScalar       = float;

  using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;
  static constexpr bool UseCustomEVT = true;

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
  using CustomComputeEVT =  // (alpha * acc) + beta * C
    cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::homogeneous_multiply_add, ElementD, ElementCompute, RoundStyle>, // beta * C + (alpha * acc)
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>, // beta
      cutlass::epilogue::fusion::Sm90SrcFetch<ElementC>, // C
      cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
        cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>, // alpha
        cutlass::epilogue::fusion::Sm90AccFetch // acc
      >
    >;
  using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
    TileShape, EpilogueTileType, ElementC, ElementD, EpilogueScheduleType
  >;
  using AuxStoreDescriptor = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
    EpilogueDescriptor, cutlass::layout::RowMajor, ElementD/*ElementAux*/
  >;
  using AuxStore = cutlass::epilogue::fusion::Sm90AuxStoreReduceScatter<
                                AuxStoreDescriptor::Stages, TileShape, typename EpilogueDescriptor::EpilogueTile,
                                typename AuxStoreDescriptor::Element, RoundStyle,
                                typename AuxStoreDescriptor::Stride, typename AuxStoreDescriptor::SmemLayoutAtom,
                                typename AuxStoreDescriptor::CopyOpR2S, CommKindEnum::IntraNode>;

    // using AuxStoreType = Sm90AuxStoreReduceScatter<
    // DispatchPolicy::StagesD,
    // decltype(params.tile_shape()),
    // decltype(params.epilogue_tile_mn()),
    // ElementD,
    // RoundStyle,
    // decltype(params.stride_d()),
    // decltype(params.smem_layout_atom_d()),
    // decltype(params.copy_op_r2s()),
    // rs_meta.comm_kind()>;
  using CustomEVT = cutlass::epilogue::fusion::Sm90EVT<AuxStore, CustomComputeEVT>;

  // A predefined set of fusion operations (implemented with EVT) are supported by the TMA warp-specialized epilogue.
  // Users can select one of these operations by passing one of the tags defined in include/cutlass/epilogue/fusion/operations.hpp
  // to the CollectiveBuilder. This frees the user from having to compute additional parameters such as stage counts and copy atoms/layouts.
  // These tags also provide additional metadata that can be queried at compile time.
  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;

  // CollectiveEpilogue的ClusterShape是111，CollectiveMainloop的是211
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
      TileShape, cute::Shape<cute::_1, cute::_1, cute::_1>, // ClusterShape,
      EpilogueTileType,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType,
      cute::conditional_t<UseCustomEVT, CustomEVT, DefaultOperation>
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
      >,
      MainloopScheduleType
    >::CollectiveOp;

  using ReduceScatterDma = Sm90ReduceScatterDma<
        1, // StagesDma,
        TileShape,
        typename EpilogueDescriptor::EpilogueTile,
        typename AuxStoreDescriptor::SmemLayoutAtom,
        ElementD,
        typename AuxStoreDescriptor::Stride,
        CommKindEnum::IntraNode,
        false>; // rs_meta.fuse_reduction()()

  using GemmKernel = cutlass::gemm::kernel::GemmUniversalRsSm90<
      cute::Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler,
      ReduceScatterDma
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
    RtArgumentsV2 *rt_args = dynamic_cast<RtArgumentsV2*>(args);

    //////////////////////////////////////////
    is_serial_ = false;
    m_ = rt_args->m;
    n_ = rt_args->n;
    output_len_ = rt_args->m * rt_args->n;
    ar_args_.output = rt_args->ptr_D;    
    ar_args_.aux_local_size = output_len_ * sizeof(ElementD);
    ar_args_.aux_local_buffer = GlobalBuffer::instance().ResizeDeviceBuffer2IfNeeded(ar_args_.aux_local_size);
    auto cu_stream = static_cast<cudaStream_t>(stream);
    fetch_comm_args(fusion_args, cu_stream);
    //////////////////////////////////////////

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

#ifdef ENABLE_ALLREDUCE
    if (is_serial_ == true) {
      int max_blocks = 32;
      constexpr int threads = 1024;
      int blocks = max_blocks; // std::min(max_blocks, n_ / ThreadblockShape::kN);
      vllm::cross_device_reduce_1stage<to_cuda_type_t<ElementD>, 2><<<blocks, threads, 0, cu_stream>>>(ar_args_.rank_data, ar_args_.rank_signals, ar_args_.self_signal, reinterpret_cast<to_cuda_type_t<ElementD>*>(ar_args_.output), ar_args_.rank, ar_args_.packed_array_num);      
    }
#endif
  }

private:
  typename Gemm::Arguments args_from_options(const RtArgumentsV2 *rt_args) {
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

#ifdef ENABLE_ALLREDUCE
    ElementD *gemm_out = (ElementD *)ar_args_.reg_buffer;
#else
    ElementD *gemm_out = (ElementD *)rt_args->ptr_D;
#endif
    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, // mode
      problem_size, // problem_shape
      {(ElementA *)rt_args->ptr_A, stride_A, (ElementB *)rt_args->ptr_B, stride_B}, // mainloop
      {{}, // epilogue.thread
       (ElementC *)rt_args->ptr_C, stride_C, gemm_out, stride_D}, // epilogue
      hw_info // hw_info
    };

    cudaMemcpy(&host_rank_data_, ar_args_.rank_data, sizeof(vllm::RankData), cudaMemcpyDeviceToHost);
    // cudaMemcpy(host_signals_, ar_args_.rank_signals.signals, 8*sizeof(vllm::Signal*), cudaMemcpyDeviceToHost);

    // cudaMemcpy(rank_data_, ar_args_.rank_data->ptrs, 8 * sizeof(ElementD*), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&host_rank_signals_, &ar_args_.rank_signals, sizeof(vllm::RankSignals), cudaMemcpyDeviceToHost);
    for (int i=0; i<kMaxLocalWorldSize; i++) {
      rank_data_[i] = (ElementD *)host_rank_data_.ptrs[i];

      // ar_args_.rank_signals.signals[i] 是device指针，将整个Signal拷贝到内存，再基于内存索引其成员变量ptr。下面仍拷贝不成功is_device_pointer(barrier_ptrs_[0])仍然为0！！！
      cudaMemcpy(host_signals_[i], ar_args_.rank_signals.signals[i], sizeof(vllm::Signal), cudaMemcpyDeviceToHost); // 
      // barrier_ptrs_[i] = (int *)host_signals_[i]->_flag;
      barrier_ptrs_[i] = (int *)ar_args_.rank_signals.signals[i]->_flag; // 这样反而可以？？？
    }

    printf("is_device_pointer: %d, %d, %d\n", is_device_pointer(barrier_ptrs_[0]), is_device_pointer(rank_data_[0]), is_device_pointer(ar_args_.rank_signals.signals[0]));

    arguments.rs_dma = typename GemmKernel::ReduceScatterDmaArguments{
      .output_scatter_ptrs = (ElementD **)rank_data_,
      .stride = stride_D,
      .rank = ar_args_.rank,
      .world_size = ar_args_.world_size,
      .nnodes = 1,
      .local_reduce_buffer = (void*)ar_args_.aux_local_buffer,
      .barrier_ptrs = (int **)barrier_ptrs_};

    // struct Arguments {
    //   Element **output_scatter_ptrs;
    //   StrideMNL stride;
    //   int rank = 0;
    //   int world_size = 0;
    //   int nnodes = 1;
    //   void *local_reduce_buffer = nullptr;
    //   int **barrier_ptrs;
    // };
    // Custom EVT fusions will have nested unnamed args, the structure of which
    // can be deduced from the type definition of the EVT.
    // Each node's arguments has the recursive structure of
    // {first_child_args, ..., last_child_args, op_args},
    // For more complex examples of EVT initialization please refer to
    // include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp
    if constexpr (UseCustomEVT) {
      arguments.epilogue.thread =
        {
          {    // ternary op : beta * C + (alpha * acc)
            {{rt_args->beta}}, // leaf op+args : beta
            {},               // leaf op+args : C  bias
            {                 // binary op : alpha * acc
              {{rt_args->alpha}}, // leaf op+args : alpha
              {},                // leaf op+args : acc
              {}              // binary args : multiplies
            },                // end binary op
            {} // ternary args : multiply_add
          },   
          {.barrier_ptr_aux = (int *)barrier_ptrs_[ar_args_.rank]}  // unary args : aux store D
        }; // end ternary op
    }
    // Pre-defined fusions will have flat, named args for user-friendlyness
    else {
      arguments.epilogue.thread.alpha = rt_args->alpha;
      arguments.epilogue.thread.beta = rt_args->beta;
    }

    return arguments;
  }

  void fetch_comm_args(void *fusion_args, cudaStream_t stream) {
#ifdef ENABLE_ALLREDUCE
    RtCommArguments *rt_args = (RtCommArguments*)(fusion_args);

    if constexpr (!(cute::is_same_v<ElementD, float> ||
      cute::is_same_v<ElementD, cutlass::half_t> ||
      cute::is_same_v<ElementD, cutlass::bfloat16_t>)) {
      throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
    }

    auto reg_buffer = reinterpret_cast<void*>(rt_args->reg_buffer);
    ar_args_.reg_buffer = reg_buffer;

    auto fa = reinterpret_cast<vllm::CustomAllreduce*>(rt_args->handle);
    fa->get_ptrs<to_cuda_type_t<ElementD>>(
      stream, reinterpret_cast<to_cuda_type_t<ElementD>*>(ar_args_.reg_buffer), output_len_,
      &ar_args_.world_size, &ar_args_.rank, &ar_args_.packed_array_num,
      &ar_args_.rank_data, &ar_args_.rank_signals, &ar_args_.self_signal);

#else
    // For testing
    is_serial_ = false;
    RtCommArguments *rt_args = (RtCommArguments*)(fusion_args);
    ar_args_.reg_buffer = reinterpret_cast<void*>(rt_args->reg_buffer);
    ar_args_.world_size = 2;
    ar_args_.rank = 1;
    // printf("finish malloc.\n");
#endif
  }

private:
  Gemm gemm_dev_;

  bool is_serial_;
  AllReduceSm90Arguments ar_args_;
  int m_;
  int n_;
  int output_len_;



  vllm::RankData host_rank_data_;
  vllm::Signal* host_signals_[8];
  
  ElementD *rank_data_[kMaxLocalWorldSize];
  int *barrier_ptrs_[kMaxLocalWorldSize];
};

} // namespace xop