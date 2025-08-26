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

/////////////////////////////////////////////////////////////////////////////////////////////////

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

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gAux,
      RTensor&& tC_rAux,
      CTensor&& tC_cAux,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      tC_gAux(cute::forward<GTensor>(tC_gAux)),
      tC_rAux(cute::forward<RTensor>(tC_rAux)),
      tC_cAux(cute::forward<CTensor>(tC_cAux)),
      problem_shape(problem_shape),
      params_ptr(params_ptr) { }

    GTensor tC_gAux;
    RTensor tC_rAux;
    CTensor tC_cAux;
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
      // if (thread0())
      //   cute::print(src_v);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        bool guard = elem_less(coord_v(i), problem_shape);
        // unpack_and_print(src_v(i));
        cutlass::arch::global_store<VecType, sizeof(VecType)>(src_v(i), (void*)&dst_v(i), guard);
        unpack_and_print(dst_v(i), guard);
      }
    }

    CUTLASS_DEVICE void
    end_epilogue() {
      // printf("hello end_epilogue.\n");
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

    // Generate the pred tensor
    Tensor cAux = make_identity_tensor(mAux.shape());
    Tensor tC_cAux = outer_partition(
      group_modes<3,6>(ThreadMap::partition(cAux, thread_idx, threadblock_tile_offset)),
      Shape<Int<VecLength>>{},
      (_0{})
    );

    return Callbacks<
      decltype(tC_gAux), decltype(tC_rAux),
      decltype(tC_cAux), ProblemShape>(
      cute::move(tC_gAux),
      cute::move(tC_rAux),
      cute::move(tC_cAux),
      problem_shape,
      params_ptr
    );
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////
