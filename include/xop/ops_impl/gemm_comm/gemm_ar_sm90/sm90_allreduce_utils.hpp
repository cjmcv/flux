
#pragma once
#include "cute/arch/cluster_sm90.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"
#include "cutlass/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "xop/xop.h"
#include "xop/ops_impl/debug_util.h"
// #include "flux/cuda/cuda_common.h"
// #include "memory_utils.hpp"
#include "custom_barrier.hpp"
#ifdef FLUX_SHM_USE_NVSHMEM
#include "host/nvshmemx_api.h"
#endif

//////////////////////////////////////////////
// reference: cutlass/arch/memory.h
namespace cutlass {
namespace arch {
// use red.global to reduce on local GPU
template <
    /// Fragment type to store data
    typename AccessType,
    /// The bytes of storing
    int StoreBytes,
    /// Element type for reduction
    typename ElementType>
struct local_red;

template <typename AccessType>
struct local_red<AccessType, 16, half_t> {
  CUTLASS_DEVICE
  local_red(AccessType const &D, void *ptr, bool pred_guard) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    using Registers = uint16_t[8];
    Registers const &data = reinterpret_cast<Registers const &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %1, 0;\n"
        "  @p red.global.add.noftz.v8.f16 [%0], {%2, %3, %4, %5, %6, %7, %8, %9};\n"
        "}\n"
        :
        : "l"(ptr),
          "r"((int)pred_guard),
          "h"(data[0]),
          "h"(data[1]),
          "h"(data[2]),
          "h"(data[3]),
          "h"(data[4]),
          "h"(data[5]),
          "h"(data[6]),
          "h"(data[7]));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

template <typename AccessType>
struct local_red<AccessType, 16, bfloat16_t> {
  CUTLASS_DEVICE
  local_red(AccessType const &D, void *ptr, bool pred_guard) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    using Registers = uint16_t[8];
    Registers const &data = reinterpret_cast<Registers const &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %1, 0;\n"
        "  @p red.global.add.noftz.v8.bf16 [%0], {%2, %3, %4, %5, %6, %7, %8, %9};\n"
        "}\n"
        :
        : "l"(ptr),
          "r"((int)pred_guard),
          "h"(data[0]),
          "h"(data[1]),
          "h"(data[2]),
          "h"(data[3]),
          "h"(data[4]),
          "h"(data[5]),
          "h"(data[6]),
          "h"(data[7]));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

}  // namespace arch
}  // namespace cutlass

//////////////////////////////////////////////
// reference: g2s - include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp
// fetch 使用tma将数据从gmem搬运到smem
// reduce 在smem上进行计算，后赋值到gmem
namespace xop {

using namespace cute;

using _AcrossNode = cute::C<CommKindEnum::AcrossNode>;

template <
    int Stages,
    class TileShape_,
    class EpilogueTile_,
    class SmemLayoutAtom_,
    class Element_,
    class StrideMNL_,
    CommKindEnum CommKind_,
    bool FuseReduction_>
struct Sm90ReduceScatterDma {
 public:
  // Type aliases
  using TileShape = TileShape_;
  using EpilogueTile = EpilogueTile_;
  using SmemLayoutAtom = SmemLayoutAtom_;
  using Element = Element_;
  using StrideMNL = StrideMNL_;
  static constexpr CommKindEnum CommKind = CommKind_;
  static constexpr bool FuseReduction = FuseReduction_;
  static_assert(not(!FuseReduction and CommKind == _AcrossNode{}));
  static constexpr int kAlignment = 128 / sizeof_bits_v<Element>;

  // Shared Mem
  constexpr static bool is_m_major =
      cutlass::epilogue::collective::detail::is_m_major<StrideMNL>();
  static_assert(not is_m_major, "only support n-major now");
  // Find the max contiguous layout usable by TMA (if EpilogueTile is a non-compact tiler)
  using SmemShapeTma = decltype(make_shape(
      max_common_vector(make_layout(get<0>(EpilogueTile{})), make_layout(get<0>(EpilogueTile{}))),
      max_common_vector(make_layout(get<1>(EpilogueTile{})), make_layout(get<1>(EpilogueTile{})))));
  using SmemLayoutTma = decltype(tile_to_shape(
      SmemLayoutAtom{}, SmemShapeTma{},
      cute::conditional_t<is_m_major, Step<_2, _1>, Step<_1, _2>>{}));
  using SmemLayout = decltype(tile_to_shape(
      SmemLayoutTma{},
      make_shape(size<0>(shape(EpilogueTile{})), size<1>(shape(EpilogueTile{})), Int<Stages>{}),
      cute::conditional_t<is_m_major, Step<_2, _1, _3>, Step<_1, _2, _3>>{}));

  // used for tma load copy
  using FetchPipeline = cutlass::PipelineTransactionAsync<Stages>;
  using PipelineState = cutlass::PipelineState<Stages>;
  using PipelineParams = typename FetchPipeline::Params;
  static constexpr int TmaTransactionBytes =
      size<0>(EpilogueTile{}) * size<1>(EpilogueTile{}) * sizeof(Element);

  static constexpr int ThreadCount = 32;

  struct Arguments {
    Element **output_scatter_ptrs;
    StrideMNL stride;
    int rank = 0;
    int world_size = 0;
    int nnodes = 1;
    void *local_reduce_buffer = nullptr;
    int **barrier_ptrs;
    // vllm::RankSignals barrier_ptrs;
  };

  struct Params {
    using TMA_Fetch = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(static_cast<Element const *>(nullptr), repeat_like(StrideMNL{}, int32_t(0)), StrideMNL{}),
        SmemLayoutTma{}));

    tuple<int, int> problem_shape;
    int rank;
    int world_size;
    int local_rank;
    int local_world_size;
    int node_idx;
    int nnodes;
    int tile_m_perrank;

    StrideMNL stride;

    Element *local_ptr[kMaxLocalWorldSize];
    TMA_Fetch tma_load_fetch[kMaxLocalWorldSize];
    Element *local_reduce_buffer;
    int *local_barrier_ptr[kMaxLocalWorldSize];
    Layout<Shape<int, int>> tile_layout;
  };

  struct TensorStorage {
    alignas(cutlass::detail::alignment_for_swizzle(SmemLayout{}))
        array_aligned<Element, size(SmemLayout{})> tensor;
  };

  using PipelineStorage = typename FetchPipeline::SharedStorage;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const &problem_shape, Arguments const &args) {
    Params params;
    auto [M, N, K, L] = problem_shape;

    params.rank = args.rank;
    params.world_size = args.world_size;
    params.nnodes = args.nnodes;

    XOP_CHECK(params.nnodes <= kMaxWorldSize / kMaxLocalWorldSize);
    XOP_CHECK(params.world_size % params.nnodes == 0);
    params.local_world_size = params.world_size / params.nnodes;
    XOP_CHECK(params.nnodes <= params.local_world_size) << " not supported yet.";

    params.local_rank = params.rank % params.local_world_size;
    params.node_idx = params.rank / params.local_world_size;
    XOP_CHECK(params.local_world_size <= kMaxLocalWorldSize);

    auto [tile_M, tile_N, tile_K] = TileShape{};

    XOP_CHECK(M % tile_M == 0) << "M=" << M << " tile_M=" << tile_M;
    XOP_CHECK(N % tile_N == 0) << "N=" << N << " tile_N=" << tile_N;
    XOP_CHECK(M % (tile_M * params.world_size) == 0)
        << "M=" << M << " tile_M=" << tile_M << " world_size=" << params.world_size;

    XOP_CHECK(args.barrier_ptrs != nullptr);
    XOP_CHECK(args.local_reduce_buffer != nullptr);

    // 如4卡，tile_M=128, M=1024, 则每卡负责 1024/(128*4) = 2个m方向的tile
    params.tile_m_perrank = M / (tile_M * params.world_size);
    params.stride = args.stride;

    params.local_reduce_buffer = static_cast<Element *>(args.local_reduce_buffer);
    for (int local_rank = 0; local_rank < params.local_world_size; ++local_rank) {
      int global_rank = params.node_idx * params.local_world_size + local_rank;
      Element *ptr = static_cast<Element *>(args.output_scatter_ptrs[global_rank]);
      int *barrier_ptr = reinterpret_cast<int **>(args.barrier_ptrs)[global_rank];
      // int *barrier_ptr = (int *)args.barrier_ptrs.signals[global_rank]->_flag;
      XOP_CHECK(barrier_ptr != nullptr);
      params.local_ptr[local_rank] = ptr;
      params.local_barrier_ptr[local_rank] = barrier_ptr;

      auto tensor_fetch = make_tensor(ptr, make_layout(make_shape(M, N, L), args.stride));
      auto tma = make_tma_copy(SM90_TMA_LOAD{}, tensor_fetch, SmemLayoutTma{});
      params.tma_load_fetch[local_rank] = cute::move(tma);
    }

    int m_tiles = ceil_div(M, size<0>(TileShape{}));
    int n_tiles = ceil_div(N, size<1>(TileShape{}));
    params.tile_layout = make_layout(make_shape(m_tiles, n_tiles));
    return params;
  }

  const Params *params_ptr;
  Element *smem_tensor;

  CUTLASS_HOST_DEVICE
  Sm90ReduceScatterDma() {}

  CUTLASS_HOST_DEVICE
  Sm90ReduceScatterDma(Params const &params, TensorStorage const &shared_tensor)
      : params_ptr(&params), smem_tensor(const_cast<Element *>(shared_tensor.tensor.data())) {}

  template <class ProblemShapeMNKL, class TileCoordMNKL>
  CUTLASS_DEVICE auto
  fetch(
      FetchPipeline fetch_pipeline,
      PipelineState fetch_write_state,
      ProblemShapeMNKL const &problem_shape,
      TileCoordMNKL const &tile_coord) {
    using namespace cute;
    auto [M, N, K, L] = problem_shape;
    auto [m, n, k, l] = tile_coord;

    if (m >= size<0>(params_ptr->tile_layout.shape()) or
        n >= size<1>(params_ptr->tile_layout.shape())) {
      // early exit if out of bound
      return fetch_write_state;
    }

    // m是tile的m方向编号，如总共有M=1024行，tile_M=128，则 m 取值范围是 0~7，在4卡下tile_m_perrank=2。
    // 所以m=0/1时，src_rank=0；m=2/3时，src_rank=1；m=4/5时，src_rank=2；m=6/7时，src_rank=3；
    // local_src_rank中的local是指本节点，为了与跨机区分。
    // local_rank是当前卡的rank，如为2，则：
    // local_src_rank=0，m=0/1, 后半段为(2-0)*2=4, m_fetch 指向4/5
    // local_src_rank=1，m=2/3，后半段为(2-1)*2=2, m_fetch 指向4/5
    // local_src_rank=2，m=4/5，后半段为(2-2)*2=0, m_fetch 指向4/5
    // local_src_rank=3，m=6/7，后半段为(2-3)*2=-2, m_fetch 指向4/5
    // 所以对于2号卡，负责从0到3号卡的4/5块的数据收集。
    // 同理，0号卡负责0/1，1号卡负责2/3，3号卡负责6/7.
    //
    // 当前函数会有m从0-7.
    //   以local_rank=2号卡为例，当m=0/1时，从local_src_rank=0中取出其4/5. 
    //                          当m=2/3时，从local_src_rank=1中取出其4/5...
    //   以local_rank=3号卡为例，当m=0/1时，从local_src_rank=0中取出其6/7.
    //                          当m=2/3时，从local_src_rank=1中取出其6/7...
    //   从gmem(远程tma tensor)拷贝到smem(当前rank)，smem视图维度是(EPI_TILE_M,EPI_TILE_N,PIPE)，PIPE即stage，这里取1即可。
    //   gmem对应视图是(EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)，需要分 EPI_M*EPI_N 次copy。
    //   因为smem只有一份，每次copy后都需要由fetch_pipeline.producer_commit通知给reduce线程，reduce完了后会由fetch_pipeline.producer_acquire获悉，开始下一次copy。
    //   与reduce的fetch_pipeline.consumer_wait / fetch_pipeline.consumer_release 对应。
    // 
    // 同步点：
    //    1) 大块同步, Barrier::wait_eq_reset(params_ptr->local_barrier_ptr[local_src_rank], thread_idx, fetch_tile_idx * 2, 1);
    //     与 sm90_visitor_store_tma_warpspecialized_ar.hpp 中的 Barrier::wait_eq_reset(params_ptr->barrier_ptr, thread_idx, tile_idx * 2, 0, 1); 对应
    //     即远程rank完成store对应tile后，即可开始对该远程rank做fetch。
    //    2) 小块同步，基于fetch_pipeline，producer_acquire 对应 consumer_wait 与 producer_commit 对应 consumer_release。

    int thread_idx = cutlass::canonical_lane_idx();
    int src_rank = m / params_ptr->tile_m_perrank;
    int local_src_rank = src_rank % params_ptr->local_world_size;
    int m_fetch = m + (params_ptr->local_rank - local_src_rank) * params_ptr->tile_m_perrank;

    /////////////////// Fetch Tensors ////////////////////
    Tensor mFetch     = params_ptr->tma_load_fetch[local_src_rank].get_tma_tensor(make_shape(M, N, L));  // (M,N,L)
    Tensor gFetch     = local_tile(mFetch, take<0, 2>(TileShape{}), make_coord(m_fetch, n, l));  // (TILE_M,TILE_N)
    Tensor gFetch_epi = flat_divide(gFetch, EpilogueTile{});  // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor sFetch_epi = make_tensor(make_smem_ptr(smem_tensor), SmemLayout{});  // (EPI_TILE_M,EPI_TILE_N,PIPE)

    ThrCopy thrblk_g2s_fetch = params_ptr->tma_load_fetch[local_src_rank].get_slice(_0{});
    Tensor bGS_gFetch = thrblk_g2s_fetch.partition_S(gFetch_epi);
    Tensor bGS_sFetch = thrblk_g2s_fetch.partition_D(sFetch_epi);

    /////////////////// Process Loop ////////////////////
    // Predication for TMA load (one thread issues TMA load)
    bool issue_tma_load = cute::elect_one_sync();

    // wait for the tile to fetch ready before processing
    int fetch_tile_idx = params_ptr->tile_layout(m_fetch, n);

    using BarrierSync = cutlass::detail::NamedBarrierSync<ThreadCount, (int)FluxNamedBarriers::ReduceScatterFetch>;
    using Barrier     = cutlass::detail::GenericSystemBarrier<BarrierSync>;

    Barrier::wait_eq_reset(params_ptr->local_barrier_ptr[local_src_rank], thread_idx, fetch_tile_idx * 2, 1);

    // todo: 检查这个fetch_pipeline producer是否会跟reduce的consumer_wait交错进行
    //       检查当不做 FuseReduction 时，size<2>(gFetch_epi)与 reduce的 size<2>(gReduce_epi) 是否一致？
    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gFetch_epi); ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < size<2>(gFetch_epi); ++epi_m) {
        constexpr uint16_t mcast_mask = 0;
        uint64_t *tma_barrier = fetch_pipeline.producer_get_barrier(fetch_write_state);
        fetch_pipeline.producer_acquire(fetch_write_state);

        if (issue_tma_load) {
          copy(params_ptr->tma_load_fetch[local_src_rank].with(*tma_barrier, mcast_mask),
               bGS_gFetch(_, _, _, epi_m, epi_n), bGS_sFetch(_, _, _, fetch_write_state.index()));
          fetch_pipeline.producer_expect_transaction(fetch_write_state);
        }
        fetch_pipeline.producer_commit(fetch_write_state);
        ++fetch_write_state;
      }
    }
    return fetch_write_state;
  }

  CUTLASS_DEVICE auto
  fetch_tail(FetchPipeline fetch_pipeline, PipelineState fetch_write_state) {
    bool issue_tma_load = cute::elect_one_sync();
    if (issue_tma_load) {
      fetch_pipeline.producer_tail(fetch_write_state);
    }
  }

  
  template <class ProblemShapeMNKL, class TileCoordMNKL>
  CUTLASS_DEVICE auto
  reduce(
      FetchPipeline fetch_pipeline,
      PipelineState fetch_read_state,
      ProblemShapeMNKL const &problem_shape,
      TileCoordMNKL const &tile_coord) {
    auto [M, N, K, L] = problem_shape;
    auto [m, n, k, l] = tile_coord;

    if (m >= size<0>(params_ptr->tile_layout.shape()) or
        n >= size<1>(params_ptr->tile_layout.shape())) {
      // early exit if out of bound
      return fetch_read_state;
    }

    int thread_idx = cutlass::canonical_lane_idx();

    int dst_rank = m / params_ptr->tile_m_perrank;
    int local_dst_rank = dst_rank % params_ptr->local_world_size;
    int dst_node_idx = dst_rank / params_ptr->local_world_size;
    // the logical m coord of the reduction tile in the output buffer
    int m_reduce_in_output = m + (params_ptr->local_rank - local_dst_rank) * params_ptr->tile_m_perrank;
    
    // 即fetch的例子：
    // 当前函数会有m从0-7.
    //   以local_rank=2号卡为例，当m=0/1时，从local_src_rank=0中取出其4/5. 
    //                          当m=2/3时，从local_src_rank=1中取出其4/5...
    //   以local_rank=3号卡为例，当m=0/1时，从local_src_rank=0中取出其6/7.
    //                          当m=2/3时，从local_src_rank=1中取出其6/7...
    // 那么这里的 原m_reduce 范围就是0-7，4卡 => tile_m_perrank=2

    /////////////////// Reduce Tensors ////////////////////
    auto get_mReduce = [&]() {
      auto [M, N] = take<0, 2>(problem_shape);
      if constexpr (FuseReduction) {
        int M_reduce = params_ptr->tile_m_perrank * params_ptr->nnodes * params_ptr->nnodes * get<0>(TileShape{});
        if constexpr (CommKind == _AcrossNode{}) {
          auto tile_layout = make_ordered_layout(take<0, 2>(TileShape{}), make_step(_1{}, _0{}));
          auto mReduce = make_tensor(params_ptr->local_ptr[params_ptr->rank], tile_to_shape(tile_layout, make_shape(M_reduce, N)));
          return mReduce;
        } else {
          auto mReduce = make_tensor(params_ptr->local_ptr[params_ptr->rank], make_ordered_layout(make_shape(M, N), make_step(_1{}, _0{}))); // make_shape(M_reduce, N)
          return mReduce;
        }
      } else {
        auto mReduce = make_tensor(params_ptr->local_ptr[params_ptr->rank], make_ordered_layout(make_shape(M, N), make_step(_1{}, _0{}))); // 
        return mReduce;
      }
    };

    // 无论做不做reduce，gReduce的大小都是一样的。这里以m为单元进行派发数据，m为0-7, 则会有8个tile的m进行这里。
    // 如果做reduce，每个m会针对指向上面收缩后的范围(m_reduce % params_ptr->tile_m_perrank)，从smem拿数据规约到这里。
    // 如果不做reduce，每个m会直接指向原本自己所属的目的地，从smem取出数据。
    // 0-7的m里对应的smem的内容是 从其他rank需要规约的数据。所以如果不fused reduce，则需要将目的buffer中，将自己的m=2/3，4/5，6/7, 都额外规约到自己的0/1上。
    auto mReduce = get_mReduce();  // (M_reduce,N,L)
    Tensor gReduce = local_tile(mReduce, take<0, 2>(TileShape{}), make_coord(m_reduce_in_output, n));  // (TILE_M,TILE_N)
    Tensor gReduce_epi = flat_divide(gReduce, EpilogueTile{});  // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor sReduce_epi = make_tensor(make_smem_ptr(smem_tensor), SmemLayout{});  // (EPI_TILE_M,EPI_TILE_N,PIPE)

    // tiled copy for fetch from global memory from other rank to registers
    // each thread of the TiledMMA (256 threads for cooperative and 128 threads for pingpong
    // kernel) process contiguous Alignment values
    constexpr int ThreadLayoutN = size<1>(EpilogueTile{}) / kAlignment;
    constexpr int ThreadLayoutM = ThreadCount / ThreadLayoutN;

    auto tiled_copy = make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        make_layout(make_shape(Int<ThreadLayoutM>{}, Int<ThreadLayoutN>{}),
                    make_stride(Int<ThreadLayoutN>{}, _1{})),
        make_layout(make_shape(_1{}, Int<kAlignment>{}), make_stride(_0{}, _1{})));

    auto thread_copy = tiled_copy.get_slice(thread_idx);
    Tensor tsReduce = thread_copy.partition_S(sReduce_epi);  // ((Atom,AtomNum),ATOM_M,ATOM_N,PIPE)
    Tensor tgReduce = thread_copy.partition_D(gReduce_epi);  // ((Atom,AtomNum),ATOM_M,ATOM_N,EPI_M,EPI_N)

    using BarrierSync = cutlass::detail::NamedBarrierSync<ThreadCount, (int)FluxNamedBarriers::ReduceScatterReduce>;
    using Barrier = cutlass::detail::CustomizedGenericBarrier<BarrierSync>;

    using BarrierSysSync = cutlass::detail::NamedBarrierSync<ThreadCount, (int)FluxNamedBarriers::AllReduceAllgather>;
    using BarrierSys = cutlass::detail::GenericSystemBarrier<BarrierSysSync>;

    int reduce_tile_idx = params_ptr->tile_layout(m_reduce_in_output, n);
    int *lock_ptr = params_ptr->local_barrier_ptr[params_ptr->local_rank];
    int flag_idx = reduce_tile_idx * 2 + 1;

    bool is_local_tile_reduce = local_dst_rank == params_ptr->local_rank;

    if constexpr (FuseReduction) {
      if (not is_local_tile_reduce) {
        // if this tile is fetched from other rank, wait for the local rank to reduce first
        // 与下面的int reduce_count = Barrier::arrive_inc_get(lock_ptr, thread_idx, flag_idx, 1);对应
        // 如果当前tile任务是从其他rank获取数据，那么需要等待当前rank的数据就绪，即需要用过一次arrive_inc_get。
        // 因为下面的tgReduce_epi第一次是直接从smem拷贝过去的(免去清零操作？)，读取其他rank则在tgReduce_epi进行累加。
        Barrier::wait_lt(lock_ptr, thread_idx, flag_idx, 1);
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gReduce_epi); ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < size<2>(gReduce_epi); ++epi_m) {
        auto barrier_token = fetch_pipeline.consumer_try_wait(fetch_read_state);
        fetch_pipeline.consumer_wait(fetch_read_state, barrier_token);
        // do copy from smem to reg and reduce to gmem
        Tensor tsReduce_epi = tsReduce(_, _, _, fetch_read_state.index());
        Tensor tgReduce_epi = tgReduce(_, _, _, epi_m, epi_n);

        CUTLASS_PRAGMA_UNROLL
        for (int copy_m = 0; copy_m < size<1>(tgReduce_epi); ++copy_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int copy_n = 0; copy_n < size<2>(tgReduce_epi); ++copy_n) {
            Tensor trReduce = make_tensor<Element>(size<0>(tgReduce));
            // fetch from local_src_rank
            copy(tiled_copy, tsReduce_epi(_, copy_m, copy_n), trReduce);
            // write to reduce_buffer
            if constexpr (FuseReduction) {
              if (is_local_tile_reduce) {
                // trReduce是自己的，直接拷贝
                copy(tiled_copy, trReduce, tgReduce_epi(_, copy_m, copy_n));
              } else {
                // trReduce是其他rank的，需要规约
                using VecType = uint_byte_t<sizeof(trReduce)>;
                cutlass::arch::local_red<VecType, sizeof(Element) * kAlignment, Element>(
                    recast<VecType>(trReduce)(_0{}),
                    (void *)tgReduce_epi(_, copy_m, copy_n).data(),
                    true);
              }
            } else {
              // 不做reduce，就直接拷贝。
              copy(tiled_copy, trReduce, tgReduce_epi(_, copy_m, copy_n));
            }
          }
        }

        fetch_pipeline.consumer_release(fetch_read_state);
        ++fetch_read_state;
      }
    }

    // 确保所有rank都到位，每到位一个则arrive_inc_get+1，由wait_eq_reset集齐统一退出
    if constexpr (FuseReduction) {
      int reduce_count = Barrier::arrive_inc_get(lock_ptr, thread_idx, flag_idx, 1);
      if (reduce_count == params_ptr->local_world_size) {
        // 仅有一组能到达这里，其他组获取的reduce_count无法进入到这里if里面
        BarrierSys::wait_eq_reset(lock_ptr, thread_idx, flag_idx, params_ptr->local_world_size, 99);
        if constexpr (CommKind == _AcrossNode{}) {
          if (dst_node_idx != params_ptr->node_idx) {
            int remote_rank = dst_node_idx * params_ptr->local_world_size + params_ptr->local_rank;
#ifdef FLUX_SHM_USE_NVSHMEM
            nvshmemx_putmem_nbi_warp(
                gReduce.data(), gReduce.data(), gReduce.size() * sizeof(Element), remote_rank);
#endif
          }
        }
      }
    }

    // // allgather
    if constexpr (1) { 
      // 不能使用tile级别的拷贝，因为进入这里的线程是warp为单位的，并没有完整block的所有线程。所以拷贝要沿用前面的warp级别拷贝
      // auto thr_layout = make_layout(make_shape(size<0>(TileShape{}), size<1>(TileShape{}) / kAlignment));
      // auto val_layout = make_layout(make_shape(_1{}, Int<kAlignment>{}), make_stride(_0{}, _1{}));
      // auto tiled_copy = make_tiled_copy(
      //   Copy_Atom<DefaultCopy, Element>{},
      //   thr_layout,
      //   val_layout);

      // auto mReduce = make_tensor(params_ptr->local_ptr[params_ptr->rank], make_ordered_layout(make_shape(M, N), make_step(_1{}, _0{})));
      // Tensor gReduce = local_tile(mReduce, take<0, 2>(TileShape{}), make_coord(m, n));
      auto mGather = make_tensor(params_ptr->local_reduce_buffer, make_ordered_layout(make_shape(M, N), make_step(_1{}, _0{})));
      Tensor gGather = local_tile(mGather, take<0, 2>(TileShape{}), make_coord(m, n));  // (TILE_M,TILE_N)
      Tensor gGather_epi = flat_divide(gGather, EpilogueTile{});
      Tensor tgGather = thread_copy.partition_D(gGather_epi);       // ((Atom,AtomNum),ATOM_M,ATOM_N,EPI_M,EPI_N)

      if (m == m_reduce_in_output) { 
        BarrierSys::wait_eq(lock_ptr, thread_idx, flag_idx, 99);
        // Tensor tgReduce = thread_copy.partition_S(gReduce_epi);  // ((Atom,AtomNum),ATOM_M,ATOM_N,PIPE)
        copy(tiled_copy, tgReduce, tgGather);
        // xop::print_tensor_shape("src_thr_shape", src_thr);
        // xop::print_tensor("src_thr", src_thr, false);
        // xop::print_tensor("dst_thr", dst_thr, false);
      }
      else {
        // m从0-7，local_rank=0 => m=0/1; 1=>2/3; 2=>4/5; 3=>6/7
        int src_rank = m / params_ptr->tile_m_perrank;

        int *src_lock_ptr = params_ptr->local_barrier_ptr[src_rank];
        BarrierSys::wait_eq(src_lock_ptr, thread_idx, flag_idx, 99);

        auto mSrc = make_tensor(params_ptr->local_ptr[src_rank], make_ordered_layout(make_shape(M, N), make_step(_1{}, _0{})));
        Tensor gSrc = local_tile(mSrc, take<0, 2>(TileShape{}), make_coord(m, n));  // (TILE_M,TILE_N)
        Tensor gSrc_epi = flat_divide(gSrc, EpilogueTile{});

        Tensor tgSrc = thread_copy.partition_S(gSrc_epi);
        copy(tiled_copy, tgSrc, tgGather);
      }
    }
    
    return fetch_read_state;
  }

  template <class ProblemShapeMNKL, class TileCoordMNKL>
  CUTLASS_DEVICE auto
  reduce2(
      FetchPipeline fetch_pipeline,
      PipelineState fetch_read_state,
      ProblemShapeMNKL const &problem_shape,
      TileCoordMNKL const &tile_coord) {
    auto [M, N, K, L] = problem_shape;
    auto [m, n, k, l] = tile_coord;

    if (m >= size<0>(params_ptr->tile_layout.shape()) or
        n >= size<1>(params_ptr->tile_layout.shape())) {
      // early exit if out of bound
      return fetch_read_state;
    }

    int thread_idx = cutlass::canonical_lane_idx();

    int dst_rank = m / params_ptr->tile_m_perrank;
    int local_dst_rank = dst_rank % params_ptr->local_world_size;
    int dst_node_idx = dst_rank / params_ptr->local_world_size;
    // the logical m coord of the reduction tile in the output buffer
    int m_reduce_in_output = m + (params_ptr->local_rank - local_dst_rank) * params_ptr->tile_m_perrank;
    
    // 即fetch的例子：
    // 当前函数会有m从0-7.
    //   以local_rank=2号卡为例，当m=0/1时，从local_src_rank=0中取出其4/5. 
    //                          当m=2/3时，从local_src_rank=1中取出其4/5...
    //   以local_rank=3号卡为例，当m=0/1时，从local_src_rank=0中取出其6/7.
    //                          当m=2/3时，从local_src_rank=1中取出其6/7...
    // 那么这里的 原m_reduce 范围就是0-7，4卡 => tile_m_perrank=2
    // 如果 FuseReduction，单节点下：
    //   m_reduce = 0/1 => 0/1 + 2 * 0 => 0/1
    //              2/3 => 0/1 + 2 * 0 => 0/1
    //              4/5 => 0/1 + 2 * 0 => 0/1
    //              6/7 => 0/1 + 2 * 0 => 0/1
    //   均指向前0/1，为结果填充区的范围。
    // 双节点下，如双节点双卡(nnodes=2, node_idx=0)：？？确认
    //   m_reduce = 0/1 (dst_rank=0, dst_node_idx=0) => 0/1 + 2 * 0 => 0/1
    //              2/3 (dst_rank=1, dst_node_idx=0) => 0/1 + 2 * 0 => 0/1
    //              4/5 (dst_rank=2, dst_node_idx=1) => 0/1 + 2 * (1*2+0) => 4/5
    //                               如 node_idx=1   => 0/1 + 2 * (1*2+1) => 6/7
    //              6/7 (dst_rank=3, dst_node_idx=1) => 0/1 + 2 * (1*2+0) => 4/5
    //                               如 node_idx=1   => 0/1 + 2 * (1*2+1) => 6/7

    // the actual m coord in reduce_buffer
    int m_reduce = get<0>(tile_coord);
    if constexpr (FuseReduction) {
      m_reduce = (m_reduce % params_ptr->tile_m_perrank) +
             params_ptr->tile_m_perrank * (dst_node_idx * params_ptr->nnodes + params_ptr->node_idx);
    }

    /////////////////// Reduce Tensors ////////////////////
    auto get_mReduce = [&]() {
      auto [M, N] = take<0, 2>(problem_shape);
      if constexpr (FuseReduction) {
        int M_reduce = params_ptr->tile_m_perrank * params_ptr->nnodes * params_ptr->nnodes * get<0>(TileShape{});
        if constexpr (CommKind == _AcrossNode{}) {
          auto tile_layout = make_ordered_layout(take<0, 2>(TileShape{}), make_step(_1{}, _0{}));
          auto mReduce = make_tensor(params_ptr->local_reduce_buffer, tile_to_shape(tile_layout, make_shape(M_reduce, N)));
          return mReduce;
        } else {
          auto mReduce = make_tensor(params_ptr->local_reduce_buffer, make_ordered_layout(make_shape(M_reduce, N), make_step(_1{}, _0{})));
          return mReduce;
        }
      } else {
        auto mReduce = make_tensor(params_ptr->local_reduce_buffer, make_ordered_layout(make_shape(M, N), make_step(_1{}, _0{})));
        return mReduce;
      }
    };

    // 无论做不做reduce，gReduce的大小都是一样的。这里以m为单元进行派发数据，m为0-7, 则会有8个tile的m进行这里。
    // 如果做reduce，每个m会针对指向上面收缩后的范围(m_reduce % params_ptr->tile_m_perrank)，从smem拿数据规约到这里。
    // 如果不做reduce，每个m会直接指向原本自己所属的目的地，从smem取出数据。
    // 0-7的m里对应的smem的内容是 从其他rank需要规约的数据。所以如果不fused reduce，则需要将目的buffer中，将自己的m=2/3，4/5，6/7, 都额外规约到自己的0/1上。
    auto mReduce = get_mReduce();  // (M_reduce,N,L)
    Tensor gReduce = local_tile(mReduce, take<0, 2>(TileShape{}), make_coord(m_reduce, n));  // (TILE_M,TILE_N)
    Tensor gReduce_epi = flat_divide(gReduce, EpilogueTile{});  // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor sReduce_epi = make_tensor(make_smem_ptr(smem_tensor), SmemLayout{});  // (EPI_TILE_M,EPI_TILE_N,PIPE)

    // tiled copy for fetch from global memory from other rank to registers
    // each thread of the TiledMMA (256 threads for cooperative and 128 threads for pingpong
    // kernel) process contiguous Alignment values
    constexpr int ThreadLayoutN = size<1>(EpilogueTile{}) / kAlignment;
    constexpr int ThreadLayoutM = ThreadCount / ThreadLayoutN;

    auto tiled_copy = make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        make_layout(make_shape(Int<ThreadLayoutM>{}, Int<ThreadLayoutN>{}),
                    make_stride(Int<ThreadLayoutN>{}, _1{})),
        make_layout(make_shape(_1{}, Int<kAlignment>{}), make_stride(_0{}, _1{})));

    auto thread_copy = tiled_copy.get_slice(thread_idx);
    Tensor tsReduce = thread_copy.partition_S(sReduce_epi);  // ((Atom,AtomNum),ATOM_M,ATOM_N,PIPE)
    Tensor tgReduce = thread_copy.partition_D(gReduce_epi);  // ((Atom,AtomNum),ATOM_M,ATOM_N,EPI_M,EPI_N)

    using BarrierSync = cutlass::detail::NamedBarrierSync<ThreadCount, (int)FluxNamedBarriers::ReduceScatterReduce>;
    using Barrier = cutlass::detail::CustomizedGenericBarrier<BarrierSync>;

    int reduce_tile_idx = params_ptr->tile_layout(m_reduce_in_output, n);
    int *lock_ptr = params_ptr->local_barrier_ptr[params_ptr->local_rank];
    int flag_idx = reduce_tile_idx * 2 + 1;

    bool is_local_tile_reduce = local_dst_rank == params_ptr->local_rank;

    if constexpr (FuseReduction) {
      if (not is_local_tile_reduce) {
        // if this tile is fetched from other rank, wait for the local rank to reduce first
        // 与下面的int reduce_count = Barrier::arrive_inc_get(lock_ptr, thread_idx, flag_idx, 1);对应
        // 如果当前tile任务是从其他rank获取数据，那么需要等待当前rank的数据就绪，即需要用过一次arrive_inc_get。
        // 因为下面的tgReduce_epi第一次是直接从smem拷贝过去的(免去清零操作？)，读取其他rank则在tgReduce_epi进行累加。
        Barrier::wait_lt(lock_ptr, thread_idx, flag_idx, 1);
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gReduce_epi); ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < size<2>(gReduce_epi); ++epi_m) {
        auto barrier_token = fetch_pipeline.consumer_try_wait(fetch_read_state);
        fetch_pipeline.consumer_wait(fetch_read_state, barrier_token);
        // do copy from smem to reg and reduce to gmem
        Tensor tsReduce_epi = tsReduce(_, _, _, fetch_read_state.index());
        Tensor tgReduce_epi = tgReduce(_, _, _, epi_m, epi_n);

        CUTLASS_PRAGMA_UNROLL
        for (int copy_m = 0; copy_m < size<1>(tgReduce_epi); ++copy_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int copy_n = 0; copy_n < size<2>(tgReduce_epi); ++copy_n) {
            Tensor trReduce = make_tensor<Element>(size<0>(tgReduce));
            // fetch from local_src_rank
            copy(tiled_copy, tsReduce_epi(_, copy_m, copy_n), trReduce);
            // write to reduce_buffer
            if constexpr (FuseReduction) {
              if (is_local_tile_reduce) {
                // trReduce是自己的，直接拷贝
                copy(tiled_copy, trReduce, tgReduce_epi(_, copy_m, copy_n));
              } else {
                // trReduce是其他rank的，需要规约
                using VecType = uint_byte_t<sizeof(trReduce)>;
                cutlass::arch::local_red<VecType, sizeof(Element) * kAlignment, Element>(
                    recast<VecType>(trReduce)(_0{}),
                    (void *)tgReduce_epi(_, copy_m, copy_n).data(),
                    true);
              }
            } else {
              // 不做reduce，就直接拷贝。
              copy(tiled_copy, trReduce, tgReduce_epi(_, copy_m, copy_n));
            }
          }
        }

        fetch_pipeline.consumer_release(fetch_read_state);
        ++fetch_read_state;
      }
    }

    // 确保所有rank都到位，每到位一个则arrive_inc_get+1，由wait_eq_reset集齐统一退出
    if constexpr (FuseReduction) {
      int reduce_count = Barrier::arrive_inc_get(lock_ptr, thread_idx, flag_idx, 1);
      if (reduce_count == params_ptr->local_world_size) {
        Barrier::wait_eq_reset(lock_ptr, thread_idx, flag_idx, params_ptr->local_world_size, 0);
        if constexpr (CommKind == _AcrossNode{}) {
          if (dst_node_idx != params_ptr->node_idx) {
            int remote_rank = dst_node_idx * params_ptr->local_world_size + params_ptr->local_rank;
#ifdef FLUX_SHM_USE_NVSHMEM
            nvshmemx_putmem_nbi_warp(
                gReduce.data(), gReduce.data(), gReduce.size() * sizeof(Element), remote_rank);
#endif
          }
        }
      }
    }
    return fetch_read_state;
  }
};

}  // namespace bytedance::flux
