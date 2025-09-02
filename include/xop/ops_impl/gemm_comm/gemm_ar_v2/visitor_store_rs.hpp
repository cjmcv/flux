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

#include "cutlass/epilogue/threadblock/fusion/visitor_2x.hpp"
#include "xop/../../src/ops/allreduce_normal/custom_all_reduce.cuh"
#include "xop/ops_impl/debug_util.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

#define ENABLE_ALLREDUCE

template <typename T, int ngpus>
__device__ void cross_device_reduce_1stage_2(vllm::RankData* _dp, vllm::RankSignals sg, vllm::Signal* self_sg,
                                             T* __restrict__ result, int rank, int size) {
  using P = typename vllm::packed_t<T>::P;
  using A = typename vllm::packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;

  int max_block_num = 48;
  if (gridDim.x < max_block_num)
    max_block_num = gridDim.x;
  if (blockIdx.x >= max_block_num) 
    return;

  vllm::barrier_at_start<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
      idx += max_block_num * blockDim.x) {
    ((P*)result)[idx] = vllm::packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }    
  
  vllm::barrier_at_end<ngpus, true>(sg, self_sg, rank);
}

template <int ngpus>
DINLINE void xop_barrier_at_start(const vllm::RankSignals& sg, vllm::Signal* self_sg, int rank, int block_id, int bias=0) {
  uint32_t flag = self_sg->_flag[block_id] + 1;
  // printf("self_sg: %p - (%p, %p), rank %d: block: %d, flag: %d\n", self_sg, sg.signals[0], sg.signals[1], rank, block_id, flag);
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x+bias]->start[block_id][rank+bias];
    auto self_counter_ptr = &self_sg->start[block_id][threadIdx.x+bias];
    // printf("a<%d> sg.signals: [%d][%d], [%d][%d] => [%p][%p], [%p][%p]\n", rank, sg.signals[0]->start[block_id][0], sg.signals[1]->start[block_id][0], sg.signals[0]->start[block_id][1], sg.signals[1]->start[block_id][1],
    //                                                                             &sg.signals[0]->start[block_id][0], &sg.signals[1]->start[block_id][0], &sg.signals[0]->start[block_id][1], &sg.signals[1]->start[block_id][1]);
    // printf("a<%d> self_sg:    [%d][%d] => [%p][%p]\n", rank, self_sg->start[block_id][0], self_sg->start[block_id][1], &self_sg->start[block_id][0], &self_sg->start[block_id][1]);
    // printf("flag: %d. (%d[tid%d][bid%d][rank%d], %d[bid%d][tid%d])", flag, *peer_counter_ptr, threadIdx.x, block_id, rank, *self_counter_ptr, block_id, threadIdx.x);
    // Write the expected counter value to peer and wait for correct value
    // from peer.
    // printf("a(%d vs %d) %d\n", *peer_counter_ptr, flag, rank);
    vllm::st_flag_volatile(peer_counter_ptr, flag);
    // printf("b(%d vs %d)\n", *peer_counter_ptr, flag);
    // printf("b<%d> sg.signals: [%d][%d], [%d][%d] => [%p][%p], [%p][%p]\n", rank, sg.signals[0]->start[block_id][0], sg.signals[1]->start[block_id][0], sg.signals[0]->start[block_id][1], sg.signals[1]->start[block_id][1],
    //   &sg.signals[0]->start[block_id][0], &sg.signals[1]->start[block_id][0], &sg.signals[0]->start[block_id][1], &sg.signals[1]->start[block_id][1]);
    // printf("b<%d> self_sg:    [%d][%d] => [%p][%p]\n", rank, self_sg->start[block_id][0], self_sg->start[block_id][1], &self_sg->start[block_id][0], &self_sg->start[block_id][1]);

    while (vllm::ld_flag_volatile(self_counter_ptr) != flag);
    // while (ld_flag_volatile(self_counter_ptr) != flag) {
    //   printf(".");
    // }
  }
  __syncthreads();
  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[block_id] = flag;
}

// This function is meant to be used as the second or the final
// synchronization barrier in the all reduce kernel. If it's the final
// synchronization barrier, we don't need to make any visibility guarantees
// for prior memory accesses.
template <int ngpus, bool final_sync = false>
DINLINE void xop_barrier_at_end(const vllm::RankSignals& sg, vllm::Signal* self_sg, int rank, int block_id, int bias=0) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[block_id] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x+bias]->end[block_id][rank+bias];
    auto self_counter_ptr = &self_sg->end[block_id][threadIdx.x+bias];
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
  if (threadIdx.x == 0) self_sg->_flag[block_id] = flag;
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
    int packed_array_num;
    
    void *reg_buffer;
    vllm::RankData* rank_data;
    vllm::RankSignals rank_signals;
    vllm::Signal *self_signal;

    void *output;
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
      Params const* params_ptr
    ):
      tC_gAux(cute::forward<GTensor>(tC_gAux)),
      tC_gRankData(cute::forward<G2Tensor>(tC_gRankData)),
      tC_rAux(cute::forward<RTensor>(tC_rAux)),
      tC_cAux(cute::forward<CTensor>(tC_cAux)),
      threadblock_tile_offset(threadblock_tile_offset),
      thread_idx(thread_idx),
      problem_shape(problem_shape),
      params_ptr(params_ptr) { }

    GTensor tC_gAux;
    G2Tensor tC_gRankData;
    RTensor tC_rAux;
    CTensor tC_cAux;
    gemm::GemmCoord threadblock_tile_offset;
    int thread_idx;
    Params const* params_ptr;
    ProblemShape problem_shape;

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

    __device__ void unpack_and_print(VecType v, bool guard) {
        // 1. 先把 128 bit 视为 4×32 bit 容器
        union {
            VecType   u128;
            uint32_t  u32[4];
        } tmp = {v};

        // 2. 每 32 bit 再解释成 2×bfloat16（高位/低位各 16 bit）
        __nv_bfloat16 bf[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t word = tmp.u32[i];
            bf[i] = *reinterpret_cast<__nv_bfloat16*>(&word);   // 直接按位拷
        }

        // 3. 打印（device printf 需 %f，会自动把 bfloat16 提升成 float）
        printf("bf16: %f %f %f %f (%d)\n",
              __bfloat162float(bf[0]),
              __bfloat162float(bf[1]),
              __bfloat162float(bf[2]),
              __bfloat162float(bf[3]), guard);
    }

    CUTLASS_DEVICE void
    end_step(int step_idx) {
      auto src_v = filter(tC_rAux);
      auto coord_v = filter(tC_cAux(_,_,_,step_idx));
      auto dst_v = filter(tC_gAux(_,_,_,step_idx));
      
      // printf("offset: %d - %d, %d, %d - %d, %p, %d.\n", thread_idx, (int)threadblock_tile_offset.m(), (int)threadblock_tile_offset.n(), (int)threadblock_tile_offset.k(), step_idx, (void*)&dst_v(0), elem_less(coord_v(0), problem_shape));
      // if (thread0()) {
      //   // cute::print(size(src_v));
      //   printf("offset: %d.\n", thread_idx);
      //   // cute::print(threadblock_tile_offset.m());
      //   // cute::print(threadblock_tile_offset.n());
      //   // cute::print(threadblock_tile_offset.k());
      //   // printf("\n");
      //   // printf("offset: %d, %d, %d.\n", (int)threadblock_tile_offset.m(), (int)threadblock_tile_offset.n(), (int)threadblock_tile_offset.k());
      // }

      auto make_tCg_view = [&](const void* base_ptr) {
        Tensor m = make_tensor(make_gmem_ptr((Element*)base_ptr), problem_shape, params_ptr->dAux);                 // (M,N,L)
        Tensor g = recast<VecType>(group_modes<3,6>(ThreadMap::partition(m, thread_idx, threadblock_tile_offset)));
        return filter(g(_,_,_,step_idx));
      };

      auto out_v = make_tCg_view(params_ptr->output);
      auto rank0_v = make_tCg_view(params_ptr->rank_data->ptrs[0]);
      auto rank1_v = make_tCg_view(params_ptr->rank_data->ptrs[1]);
      auto rank2_v = make_tCg_view(params_ptr->rank_data->ptrs[2]);
      auto rank3_v = make_tCg_view(params_ptr->rank_data->ptrs[3]);
      auto rank4_v = make_tCg_view(params_ptr->rank_data->ptrs[4]);
      auto rank5_v = make_tCg_view(params_ptr->rank_data->ptrs[5]);
      auto rank6_v = make_tCg_view(params_ptr->rank_data->ptrs[6]);
      auto rank7_v = make_tCg_view(params_ptr->rank_data->ptrs[7]);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        bool guard = elem_less(coord_v(i), problem_shape);
        // printf("i: %d, guard: %d \n", i, guard);
        // unpack_and_print(src_v(i));
        cutlass::arch::global_store<VecType, sizeof(VecType)>(src_v(i), (void*)&dst_v(i), guard);

#ifdef ENABLE_ALLREDUCE
        int block_id = threadblock_tile_offset.m() * (get<1>(problem_shape) / 128) * 8 + threadblock_tile_offset.n() + step_idx;
        xop_barrier_at_start<2>(params_ptr->rank_signals, params_ptr->self_signal, params_ptr->rank, block_id);
        // printf("block_id: %d", block_id);
        if (guard != 0) {
          using T = nv_bfloat16;
          nv_bfloat16 const *rank0_data = reinterpret_cast<nv_bfloat16 const *>(&rank0_v(i));
          nv_bfloat16 const *rank1_data = reinterpret_cast<nv_bfloat16 const *>(&rank1_v(i));
          nv_bfloat16 const *rank2_data = reinterpret_cast<nv_bfloat16 const *>(&rank2_v(i));
          nv_bfloat16 const *rank3_data = reinterpret_cast<nv_bfloat16 const *>(&rank3_v(i));
          nv_bfloat16 const *rank4_data = reinterpret_cast<nv_bfloat16 const *>(&rank4_v(i));
          nv_bfloat16 const *rank5_data = reinterpret_cast<nv_bfloat16 const *>(&rank5_v(i));
          nv_bfloat16 const *rank6_data = reinterpret_cast<nv_bfloat16 const *>(&rank6_v(i));
          nv_bfloat16 const *rank7_data = reinterpret_cast<nv_bfloat16 const *>(&rank7_v(i));
          nv_bfloat16 *output_data = reinterpret_cast<nv_bfloat16 *>(&out_v(i));

          int cnt = 8; // 128/8/2
          if (params_ptr->world_size == 2) {
            for (int j=0; j<cnt; j++) {
              output_data[j] = __hadd(rank0_data[j], rank1_data[j]);
            }
          }
          else if (params_ptr->world_size == 4) {
            for (int j=0; j<cnt; j++) {
              nv_bfloat16 temp = rank0_data[j];
              temp = __hadd(temp, rank1_data[j]);
              temp = __hadd(temp, rank2_data[j]);
              temp = __hadd(temp, rank3_data[j]);
              output_data[j] = temp;
            }
          }
          else if (params_ptr->world_size == 6) {
            for (int j=0; j<cnt; j++) {
              nv_bfloat16 temp = rank0_data[j];
              temp = __hadd(temp, rank1_data[j]);
              temp = __hadd(temp, rank2_data[j]);
              temp = __hadd(temp, rank3_data[j]);
              temp = __hadd(temp, rank4_data[j]);
              temp = __hadd(temp, rank5_data[j]);
              output_data[j] = temp;
            }
          }
          if (params_ptr->world_size == 8) {
            for (int j=0; j<cnt; j++) {
              nv_bfloat16 temp = rank0_data[j];
              temp = __hadd(temp, rank1_data[j]);
              temp = __hadd(temp, rank2_data[j]);
              temp = __hadd(temp, rank3_data[j]);
              temp = __hadd(temp, rank4_data[j]);
              temp = __hadd(temp, rank5_data[j]);
              temp = __hadd(temp, rank6_data[j]);
              temp = __hadd(temp, rank7_data[j]);
              output_data[j] = temp;
            }
          }
        }
        xop_barrier_at_end<2>(params_ptr->rank_signals, params_ptr->self_signal, params_ptr->rank, block_id);
#else
        Tensor mReg = make_tensor(make_gmem_ptr((Element*)params_ptr->reg_buffer), problem_shape, params_ptr->dAux);
        Tensor tC_gRankData2 = recast<VecType>(group_modes<3,6>(ThreadMap::partition(mReg, thread_idx, threadblock_tile_offset)));
        auto dst2_v = filter(tC_gRankData2(_,_,_,step_idx));

        cutlass::arch::global_store<VecType, sizeof(VecType)>(src_v(i), (void*)&dst2_v(i), guard);
        // unpack_and_print(dst2_v(i), guard);
#endif
      }
    }

    CUTLASS_DEVICE void
    end_epilogue() {
      // #ifdef ENABLE_ALLREDUCE
      // // printf("hello end_epilogue: %d, %d, %d\n", params_ptr->world_size, params_ptr->rank, params_ptr->packed_array_num);
      // cross_device_reduce_1stage_2<nv_bfloat16, 2>(params_ptr->rank_data, params_ptr->rank_signals, params_ptr->self_signal,
      //   reinterpret_cast<nv_bfloat16*>(params_ptr->output),
      //   params_ptr->rank, params_ptr->packed_array_num);

      // #endif
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
      params_ptr
    );
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////
