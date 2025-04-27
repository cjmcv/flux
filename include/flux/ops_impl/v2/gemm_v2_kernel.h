//===- gemm_v2_impl.hpp ------------------------------------------- C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

// This file should be included before any other cutlass device headers.
// The order of cutlass headers is carefully adjusted
#pragma once
#include <type_traits>
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"
#include "cutlass/gemm/kernel/default_gemm_with_absmax.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/common_cuda.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
// #include "flux/ops_impl/gemm_operator_base_default_impl.hpp"

#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/config.hpp"
#include "cute/container/tuple.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/util/type_traits.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/trace.h"
#include "cutlass/arch/arch.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/kernel/default_gemm_with_absmax.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/epilogue/threadblock/fusion/visitor_load.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"

#include "flux/gemm_args.h"

namespace bytedance::flux {
template <class Tuple>
constexpr auto
to_gemm_shape(Tuple tuple) {
  return cutlass::gemm::
      GemmShape<cute::size<0>(tuple), cute::size<1>(tuple), cute::size<2>(tuple)>();
}

namespace gemm_v2_impl {
template <class TBSwizzle, class AlignmentC, class EVT>
struct KernelParams {
  auto
  tb_swizzle() {
    return make_declval<TBSwizzle>();
  }
  constexpr int
  alignment_c() {
    return AlignmentC{};
  }
  auto
  evt() {
    return make_declval<EVT>();
  }
};

}  // namespace gemm_v2_impl

template <class GemmMetaT, class GemmHParamsT>
struct GemmV2Single_Kernel {

  auto
  gemm_kernel() const {
    auto params = this->default_kernel_params();
    return this->default_gemm_kernel(params);
  }

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  using ArchTag = decltype(to_cutlass_archtag(meta.arch()));
  using OpClass = cutlass::arch::OpClassTensorOp; // OpClassSimt

  using ElementA = decltype(to_cutlass_element(dt_conf.a()));
  using ElementB = decltype(to_cutlass_element(dt_conf.b()));
  using ElementC = decltype(to_cutlass_element(dt_conf.c()));
  using ElementD = decltype(to_cutlass_element(dt_conf.d()));
  using ElementAccumulator = decltype(to_cutlass_element(dt_conf.acc()));
  using ElementCNonVoid = cute::conditional_t<cute::is_void_v<ElementC>, ElementD, ElementC>;
  using ElementScale = float;  // for S8 GEMM dequant
  static constexpr int AlignmentA = 128 / cute::sizeof_bits_v<ElementA>;
  static constexpr int AlignmentB = 128 / cute::sizeof_bits_v<ElementB>;
  static constexpr bool has_bias = not cute::is_void_v<ElementC>;
  using GmemLayoutA = decltype(to_cutlass_layout_a(meta.gemm_layout()));
  using GmemLayoutB = decltype(to_cutlass_layout_b(meta.gemm_layout()));
  using GmemLayoutC = decltype(to_cutlass_layout_c(meta.gemm_layout()));
  using GmemLayoutD = GmemLayoutC;
  using StrideC = cutlass::gemm::TagToStrideC_t<GmemLayoutC>;
  using StrideD = cutlass::gemm::TagToStrideC_t<GmemLayoutD>;
  using TileShape = decltype(hparams.tile_shape());
  using ThreadblockShape = decltype(to_gemm_shape(TileShape{}));
  static constexpr auto gemm_v2_hparams = to_gemm_v2_hparams(hparams.impl_spec());
  using WarpShape = decltype(to_gemm_shape(gemm_v2_hparams.warp_shape()));
  using InstructionShape = decltype(to_gemm_shape(gemm_v2_hparams.instruction_shape()));
  static constexpr int EVTEpilogueStages = 1;

  template <class... Ts>
  auto
  output_tile_thread_map(gemm_v2_impl::KernelParams<Ts...> params) const {
    using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        ThreadblockShape,
        WarpShape,
        ElementCNonVoid,
        params.alignment_c(),
        EVTEpilogueStages>;
    return make_declval<OutputTileThreadMap>();
  }

  template <class... Ts>
  auto
  evt_d(gemm_v2_impl::KernelParams<Ts...> params) const {
    return this->default_evt_d(params);
  }

  template <class... Ts>
  auto
  default_evt_d(gemm_v2_impl::KernelParams<Ts...> params) const {
    using namespace cutlass::epilogue::threadblock;
    using ElementCompute = ElementD;
    using EVT_Compute0 = Sm80EVT<                         // 只有80和90，没有89
        VisitorCompute<
            cutlass::multiplies,
            ElementD,
            ElementCompute,
            cutlass::FloatRoundStyle::round_to_nearest>,  // alpha * acc
        VisitorScalarBroadcast<ElementAccumulator>,       // alpha
        VisitorAccFetch                                   // acc
        >;
    if constexpr (cute::is_void_v<ElementC>) {  // no bias
      return make_declval<EVT_Compute0>();
    } else {
      using OutputTileThreadMap = decltype(this->output_tile_thread_map(params));
      // NOTE: Cutlass 2.x evt does not have alternative to Sm90SrcFetch that
      // fetches the C tensor of the epilogue. So we need to do AuxLoad for C
      using C = VisitorAuxLoad<
          OutputTileThreadMap,
          ElementCNonVoid,
          cute::Stride<int64_t, cute::_1, int64_t>  // StrideMNL
          >;
      using EVT_Compute1 = Sm80EVT<  // D
          VisitorCompute<
              cutlass::multiply_add,
              ElementD,
              ElementCompute,
              cutlass::FloatRoundStyle::round_to_nearest>,  // beta * C + (alpha * acc)
              VisitorScalarBroadcast<ElementAccumulator>,       // beta
              C,                                                // C
              EVT_Compute0>;
      return make_declval<EVT_Compute1>();
    }
  }

  auto
  default_kernel_params() const {
    using TBSwizzle = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

    using AlignmentC_Type = cute::Int<128 / cute::sizeof_bits_v<ElementCNonVoid>>;

    using namespace cutlass::epilogue::threadblock;
    auto kparams = gemm_v2_impl::KernelParams<TBSwizzle, AlignmentC_Type, void>();
    using OutputTileThreadMap = decltype(this->output_tile_thread_map(kparams));

    using EVT_D = decltype(this->evt_d(kparams));

    using StoreD = VisitorAuxStore<
        OutputTileThreadMap,
        ElementD,
        cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, cute::_1, int64_t>>;
    using EVT = Sm80EVT<StoreD, EVT_D>;
    return gemm_v2_impl::KernelParams<TBSwizzle, AlignmentC_Type, EVT>();
  }

  template <class... Ts>
  auto
  default_gemm_kernel(gemm_v2_impl::KernelParams<Ts...> params) const {

    using ElementCompute = ElementD;

    using Impl = cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA,
        GmemLayoutA,
        cutlass::ComplexTransform::kNone,
        AlignmentA,
        ElementB,
        GmemLayoutB,
        cutlass::ComplexTransform::kNone,
        AlignmentB,
        ElementCNonVoid,
        GmemLayoutC,
        params.alignment_c(),
        ElementAccumulator,
        ElementCompute,
        OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        decltype(params.evt()),
        decltype(params.tb_swizzle()),
        hparams.mainloop_stage(),
        cutlass::arch::OpMultiplyAdd,
        EVTEpilogueStages>;
    return make_declval<typename Impl::GemmKernel>();
  }
};

}  // namespace bytedance::flux
