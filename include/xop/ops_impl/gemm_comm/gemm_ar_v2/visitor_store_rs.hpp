/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
  \brief Visitor tree store operations for the CUTLASS 2x epilogue
*/

#pragma once

#include <cuda/atomic>
#include "cutlass/epilogue/threadblock/fusion/visitor_2x.hpp"
#include "xop/../../src/ops/allreduce_normal/custom_all_reduce.cuh"
#include "xop/ops_impl/debug_util.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

// #define ENABLE_ALLREDUCE

template <typename T = int>
using atomic_ref_sys = cuda::atomic_ref<T, cuda::thread_scope_system>;
template <typename T = int>
using atomic_ref_dev = cuda::atomic_ref<T, cuda::thread_scope_device>;

static DINLINE void xop_st_flag_volatile(uint32_t* flag_addr, uint32_t flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE uint32_t xop_ld_flag_volatile(uint32_t* flag_addr) {
  uint32_t flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
               : "=r"(flag)
               : "l"(flag_addr));
  return flag;
}


template <int ngpus>
DINLINE void xop_barrier_at_start(const vllm::RankSignals& sg, vllm::Signal* self_sg, int rank, int bias=0) {
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  // printf("self_sg: %p - (%p, %p), rank %d: block: %d, flag: %d\n", self_sg, sg.signals[0], sg.signals[1], rank, blockIdx.x, flag);
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x+bias]->start[blockIdx.x][rank+bias];
    auto self_counter_ptr = &self_sg->start[blockIdx.x][threadIdx.x+bias];
    // printf("a<%d> sg.signals: [%d][%d], [%d][%d] => [%p][%p], [%p][%p]\n", rank, sg.signals[0]->start[blockIdx.x][0], sg.signals[1]->start[blockIdx.x][0], sg.signals[0]->start[blockIdx.x][1], sg.signals[1]->start[blockIdx.x][1],
    //                                                                             &sg.signals[0]->start[blockIdx.x][0], &sg.signals[1]->start[blockIdx.x][0], &sg.signals[0]->start[blockIdx.x][1], &sg.signals[1]->start[blockIdx.x][1]);
    // printf("a<%d> self_sg:    [%d][%d] => [%p][%p]\n", rank, self_sg->start[blockIdx.x][0], self_sg->start[blockIdx.x][1], &self_sg->start[blockIdx.x][0], &self_sg->start[blockIdx.x][1]);
    // printf("flag: %d. (%d[tid%d][bid%d][rank%d], %d[bid%d][tid%d])", flag, *peer_counter_ptr, threadIdx.x, blockIdx.x, rank, *self_counter_ptr, blockIdx.x, threadIdx.x);
    // Write the expected counter value to peer and wait for correct value
    // from peer.
    // printf("a(%d vs %d) %d\n", *peer_counter_ptr, flag, rank);
    vllm::st_flag_volatile(peer_counter_ptr, flag);
    // printf("b(%d vs %d)\n", *peer_counter_ptr, flag);
    // printf("b<%d> sg.signals: [%d][%d], [%d][%d] => [%p][%p], [%p][%p]\n", rank, sg.signals[0]->start[blockIdx.x][0], sg.signals[1]->start[blockIdx.x][0], sg.signals[0]->start[blockIdx.x][1], sg.signals[1]->start[blockIdx.x][1],
    //   &sg.signals[0]->start[blockIdx.x][0], &sg.signals[1]->start[blockIdx.x][0], &sg.signals[0]->start[blockIdx.x][1], &sg.signals[1]->start[blockIdx.x][1]);
    // printf("b<%d> self_sg:    [%d][%d] => [%p][%p]\n", rank, self_sg->start[blockIdx.x][0], self_sg->start[blockIdx.x][1], &self_sg->start[blockIdx.x][0], &self_sg->start[blockIdx.x][1]);

    while (vllm::ld_flag_volatile(self_counter_ptr) != flag);
    // while (ld_flag_volatile(self_counter_ptr) != flag) {
    //   printf(".");
    // }
  }
  __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

// This function is meant to be used as the second or the final
// synchronization barrier in the all reduce kernel. If it's the final
// synchronization barrier, we don't need to make any visibility guarantees
// for prior memory accesses.
template <int ngpus, bool final_sync = false>
DINLINE void xop_barrier_at_end(const vllm::RankSignals& sg, vllm::Signal* self_sg, int rank, int bias=0) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x+bias]->end[blockIdx.x][rank+bias];
    auto self_counter_ptr = &self_sg->end[blockIdx.x][threadIdx.x+bias];
    // Write the expected counter value to peer and wait for correct value from
    // peer.
    if constexpr (!final_sync) {
      vllm::st_flag_release(peer_counter_ptr, flag);
      while (vllm::ld_flag_acquire(self_counter_ptr) != flag);
    } else {
      vllm::st_flag_volatile(peer_counter_ptr, flag);
      while (vllm::ld_flag_volatile(self_counter_ptr) != flag);
    }
  }
  if constexpr (!final_sync) __syncthreads();

  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}


template <typename T, int ngpus>
__global__ void cross_device_reduce_1stage_tmp(vllm::RankData* _dp, vllm::RankSignals sg, vllm::Signal* self_sg,
                                             T* __restrict__ result, int rank, int size) {
  using P = typename vllm::packed_t<T>::P;
  using A = typename vllm::packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;

  vllm::barrier_at_start<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
      idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = vllm::packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }    
  
  vllm::barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

namespace cutlass::epilogue::threadblock {

using namespace cute;
using namespace detail;
using X = Underscore;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Store Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ThreadMap,
  class Element,
  FloatRoundStyle RoundStyle,
  class StrideMNL
>
struct VisitorAuxStoreRs{

  struct Arguments {
    Element* ptr_aux = nullptr;
    StrideMNL dAux = {};

    int world_size;
    int rank;

    void *reg_buffer;
    vllm::RankData* rank_data;
    vllm::RankSignals rank_signals;
    vllm::Signal *self_signal;

    void *output;

    bool is_serial;    
    bool is_streamk;
    int split_k_factor;

    uint8_t *aux_local_buffer;
    int streamk_reduce_mark_step;
    int reduce_arrival_step;
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  struct SharedStorage {};

  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorAuxStoreRs() { }

  CUTLASS_HOST_DEVICE
  VisitorAuxStoreRs(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  template <class GTensor, class G2Tensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gAux,
      G2Tensor&& tC_gRankData,
      RTensor&& tC_rAux,
      CTensor&& tC_cAux,
      gemm::GemmCoord threadblock_tile_offset,
      int thread_idx,
      ProblemShape problem_shape,
      Params const* params_ptr,
      int tile_idx
    ):
      tC_gAux(cute::forward<GTensor>(tC_gAux)),
      tC_gRankData(cute::forward<G2Tensor>(tC_gRankData)),
      tC_rAux(cute::forward<RTensor>(tC_rAux)),
      tC_cAux(cute::forward<CTensor>(tC_cAux)),
      threadblock_tile_offset(threadblock_tile_offset),
      thread_idx(thread_idx),
      problem_shape(problem_shape),
      params_ptr(params_ptr),
      tile_idx(tile_idx) { }

    GTensor tC_gAux;
    G2Tensor tC_gRankData;
    RTensor tC_rAux;
    CTensor tC_cAux;
    gemm::GemmCoord threadblock_tile_offset;
    int thread_idx;
    Params const* params_ptr;
    ProblemShape problem_shape;
    int tile_idx;
    
    CUTLASS_DEVICE void
    begin_step(int step_idx) {
      clear(tC_rAux);
    }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc,
          Array<ElementInput, FragmentSize> const& frg_input) {
      using ConvertInput = NumericArrayConverter<Element, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};

      Tensor tC_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rAux));
      tC_rAux_frg(frg_idx) = convert_input(frg_input);

      return frg_input;
    }

    CUTLASS_DEVICE void
    end_step(int step_idx) {
      // if (step_idx > 1) { return; }
      // if (blockIdx.x != 0 || blockIdx.y != 0) { return; }
      auto src_v = filter(tC_rAux);
      auto coord_v = filter(tC_cAux(_,_,_,step_idx));
      auto dst_v = filter(tC_gAux(_,_,_,step_idx));
      
      // printf("rank<%d>: %d - (%d, %d)(%d, %d) - %d, %d, %d - %d, %p, %d.\n", params_ptr->rank, thread_idx, gridDim.x, gridDim.y, blockIdx.x, blockIdx.y, (int)threadblock_tile_offset.m(), (int)threadblock_tile_offset.n(), (int)threadblock_tile_offset.k(), step_idx, (void*)&dst_v(0), elem_less(coord_v(0), problem_shape));

      // auto make_tCg_view = [&](const void* base_ptr, int step_idx, gemm::GemmCoord threadblock_tile_offset) {
      //   Tensor m = make_tensor(make_gmem_ptr((Element*)base_ptr), problem_shape, params_ptr->dAux);                 // (M,N,L)
      //   Tensor g = recast<VecType>(group_modes<3,6>(ThreadMap::partition(m, thread_idx, threadblock_tile_offset)));
      //   return filter(g(_,_,_,step_idx));
      // };
      // auto out_v = make_tCg_view(params_ptr->output, step_idx, threadblock_tile_offset);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        bool guard = elem_less(coord_v(i), problem_shape);
        // cutlass::arch::global_store<VecType, sizeof(VecType)>(src_v(i), (void*)&out_v(i), guard);
        cutlass::arch::global_store<VecType, sizeof(VecType)>(src_v(i), (void*)&dst_v(i), guard); // Store it in the local rank_data.
      }
      // if (threadIdx.x == 0) {
      //   printf("rank step<%d> (%d, %d), (%d, %d).\n", params_ptr->rank, blockIdx.x, blockIdx.y, threadblock_tile_offset.m(), threadblock_tile_offset.n());
      // }
    }

    CUTLASS_DEVICE void
    end_epilogue() {
      if (params_ptr->is_serial) return;

      // printf("(%d, %d)\n", get<1>(problem_shape), blockDim.x);
      uint32_t rank = params_ptr->rank;
      // uint32_t target_rank = (rank+1) % 2;

      int *flag_c = (int *)params_ptr->aux_local_buffer;
      int *flag_v = (int *)(params_ptr->aux_local_buffer + sizeof(int));
      int *flag_s = (int *)(params_ptr->aux_local_buffer + params_ptr->streamk_reduce_mark_step);
      int *flag_s_cnt = (int *)(params_ptr->aux_local_buffer + params_ptr->reduce_arrival_step);
#ifdef ENABLE_ALLREDUCE
      int *self_flag_e = (int*)params_ptr->rank_signals.signals[rank]->end;
      int *target_flag_e = (int*)params_ptr->rank_signals.signals[target_rank]->end;
#else
      int *self_flag_e = (int*)params_ptr->reg_buffer;
      int *target_flag_e = self_flag_e;
#endif
      // blockIdx.x => m, blockIdx.y => n;
      // Add __syncthreads before writing the flag to ensure that all data above this block has been processed. 
      // Otherwise, even if the flag is set, the data may not be valid.
      __syncthreads();
      if (threadIdx.x == 0) {
        // printf("rank0<%d> tile_idx<%d> - (%d, %d), (%d, %d, %d).\n", rank, tile_idx, blockIdx.x, blockIdx.y, threadblock_tile_offset.m(), threadblock_tile_offset.n(), threadblock_tile_offset.k());
        // A tile marked as requiring reduction shall end only after it has entered consecutively 8 times.
        // Use the old data of atomicAdd to ensure that it is unique.
        if (params_ptr->is_streamk && flag_s[tile_idx] != 0) {
          int cnt = atomicAdd(&flag_s_cnt[tile_idx], 1);
          if (cnt != 7)
            return;
        }
        if (params_ptr->split_k_factor != 1 && params_ptr->is_streamk == false) {
          int cnt = atomicAdd(&flag_s_cnt[tile_idx], 1);
          // printf("split: %d, %d, %d.\n", params_ptr->split_k_factor, flag_s_cnt[tile_idx], cnt);
          if (cnt != params_ptr->split_k_factor-1)
            return;
        }
        // printf("rank1<%d> tile_idx<%d> - (%d, %d), (%d, %d, %d).\n", rank, tile_idx, blockIdx.x, blockIdx.y, threadblock_tile_offset.m(), threadblock_tile_offset.n(), threadblock_tile_offset.k());
        int flag = self_flag_e[tile_idx] + 1;
        atomic_ref_sys<int> self_ref_e(self_flag_e[tile_idx]);
        self_ref_e.store(flag, cuda::memory_order_release);
        // __threadfence_system();

        atomic_ref_sys<int> target_ref_e(target_flag_e[tile_idx]);
        while (target_ref_e.load(cuda::memory_order_acquire) != flag) {}
        // So far, all data of tile_idx is ready.

        // 应使用旧数据idx，如使用新数据*flag_c，在取ref(flag_v[*flag_c])时，可能其他线程也刚好完成了原子加，使填数据时下标跳了两次，导致部分下标空缺。
        int idx = atomicAdd(flag_c, 1); 
        atomic_ref_sys<int> ref(flag_v[idx]);
        ref.store(tile_idx+1, cuda::memory_order_release);   
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mAux = make_tensor(
      make_gmem_ptr(params_ptr->ptr_aux),
      problem_shape,
      params_ptr->dAux);   // (M,N,L)
    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER
    Tensor tC_gAux = recast<VecType>(group_modes<3,6>(ThreadMap::partition(mAux, thread_idx, threadblock_tile_offset)));
    Tensor tC_rAux = make_tensor_like(take<0,3>(tC_gAux));

    // For test
    Tensor mReg = make_tensor(make_gmem_ptr((Element*)params_ptr->reg_buffer), problem_shape, params_ptr->dAux);
    Tensor tC_gRankData = recast<VecType>(group_modes<3,6>(ThreadMap::partition(mReg, thread_idx, threadblock_tile_offset)));

    // Generate the pred tensor
    Tensor cAux = make_identity_tensor(mAux.shape());
    Tensor tC_cAux = outer_partition(
      group_modes<3,6>(ThreadMap::partition(cAux, thread_idx, threadblock_tile_offset)),
      Shape<Int<VecLength>>{},
      (_0{})
    );

    //
    int tiled_n = (get<1>(problem_shape) + blockDim.x - 1) / blockDim.x;
    int tile_idx = threadblock_tile_offset.m() * tiled_n + threadblock_tile_offset.n();
    //
    return Callbacks<
      decltype(tC_gAux), decltype(tC_gRankData), decltype(tC_rAux),
      decltype(tC_cAux), ProblemShape>(
      cute::move(tC_gAux),
      cute::move(tC_gRankData),
      cute::move(tC_rAux),
      cute::move(tC_cAux),
      threadblock_tile_offset,
      thread_idx,
      problem_shape,
      params_ptr,
      tile_idx
    );
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////
