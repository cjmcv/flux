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
#include "flux/ops_impl/gemm_operator_base_default_impl.hpp"

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

template <
    class GemmMetaT,
    class GemmHParamsT,
    class GemmKernelT>
class GemmV2Single_Device
    : public GemmOperatorBaseDefaultImplMixin<
          GemmV2Single_Device<GemmMetaT, GemmHParamsT, GemmKernelT>>,
      public GemmV2Single_Kernel<GemmMetaT, GemmHParamsT> {
 public:

  /////////////////////////////////////////////////////////////////////////////
  using KernelBuilder = GemmV2Single_Kernel<GemmMetaT, GemmHParamsT>;
  using Base = GemmV2Single_Device<GemmMetaT, GemmHParamsT, GemmKernelT>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV2Single_Device)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  auto
  to_gemm_args_impl(SingleGemmArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;

    using ElementA = decltype(to_cutlass_element(dt_conf.a()));
    using ElementB = decltype(to_cutlass_element(dt_conf.b()));
    using ElementC = decltype(to_cutlass_element(dt_conf.c()));
    using ElementD = decltype(to_cutlass_element(dt_conf.d()));

    auto ptr_A = static_cast<ElementA const *>(args.input);
    auto ptr_B = static_cast<ElementB const *>(args.weight);
    auto ptr_C = static_cast<ElementC *>(const_cast<void *>(args.bias));
    auto ptr_D = static_cast<ElementD *>(args.output);

    auto stride_C = cutlass::make_cute_packed_stride(
        typename Base::StrideC{}, cute::make_shape(args.m, args.n, 1));
    auto stride_D = stride_C;

    auto callback_args =
        this->default_get_callback_args(ptr_C, stride_C, ptr_D, stride_D, args.alpha, args.beta);
    int stride_b = this->get_stride_b(args.n, args.k);
    auto const &v2_hparams = to_gemm_v2_hparams(hparams.impl_spec());
    int avail_sms = -1;
    if (hparams.gemm_kind() == _GemmStreamK{} and v2_hparams.streamk_mode() == _StreamkDP{}) {
      avail_sms = 1;
    }

    auto gemm_args = GemmArguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        {args.m, args.n, args.k},                 // problem_size
        1,                                        // split_k factors
        callback_args,
        ptr_A,            // ptr_A
        ptr_B,            // ptr_B
        nullptr,          // ptr_C (unused)
        nullptr,          // ptr_D
        args.m * args.k,  // batch_stride_A
        args.n * args.k,  // batch_stride_B
        args.m * args.n,  // batch_stride_C (unused)
        args.m * args.n,  // batch_stride_D (unused)
        args.k,           // stride_a
        stride_b,         // stride_b
        args.n,           // stride_c
        args.n,           // stride_d
        avail_sms);       // avail_sms
    return gemm_args;
  }


  auto
  to_gemm_args(std::any const &args, void *args_workspace) const {
    return to_gemm_args_impl(std::any_cast<SingleGemmArguments>(args));
  }
  
  
  /////////////////////////////////////////////////////////////////////////////


  // using Base = GemmOperatorBaseDefaultImplMixin<GemmV2Single_Device>;
  // using KernelBuilder = KernelBuilder_;
  // FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV2Single_Device)

  // static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  using KernelBuilder::has_bias;
  using typename KernelBuilder::ElementA;
  using typename KernelBuilder::ElementB;
  using typename KernelBuilder::ElementC;
  using typename KernelBuilder::ElementCNonVoid;
  using typename KernelBuilder::ElementD;
  using typename KernelBuilder::ElementScale;
  using typename KernelBuilder::GmemLayoutB;
  using typename KernelBuilder::GmemLayoutC;
  using typename KernelBuilder::StrideC;
  using typename KernelBuilder::StrideD;
  using typename KernelBuilder::ThreadblockShape;
  using typename KernelBuilder::TileShape;

  auto
  default_gemm_device() const {
    return make_declval<cutlass::gemm::device::GemmUniversalBase<GemmKernelT>>();
  }

 public:
  //////////////////////////
  // CRTP functions
  //////////////////////////
  auto
  gemm_device() const {
    return this->default_gemm_device();
  }

  // auto
  // to_gemm_args(std::any const &args, void *args_workspace) const {
  //   return static_cast<DerivedImpl const *>(this)->to_gemm_args(args, args_workspace);
  // }

 protected:
  int
  get_stride_b(int n, int k) const {
    if constexpr (cute::is_same_v<GmemLayoutB, cutlass::layout::RowMajor>) {
      return n;
    } else {
      static_assert(
          cute::is_same_v<GmemLayoutB, cutlass::layout::ColumnMajor>, "requires ColumnMajor.");
      return k;
    }
  }

  int
  get_stride_c(int m, int n) const {
    if constexpr (cute::is_same_v<GmemLayoutC, cutlass::layout::RowMajor>) {
      return n;
    } else {
      static_assert(
          cute::is_same_v<GmemLayoutC, cutlass::layout::ColumnMajor>, "requires ColumnMajor.");
      return m;
    }
  }

  // used for comm ops that doesn't have customized evt
  template <class PtrC, class StrideC, class PtrD, class StrideD, class Alpha, class Beta>
  auto
  default_get_callback_args(
      PtrC ptr_C, StrideC stride_C, PtrD ptr_D, StrideD stride_D, Alpha alpha, Beta beta) const {
    using EVT = identity_t<decltype(KernelBuilder().default_kernel_params().evt())>;
    if constexpr (has_bias) {
      return typename EVT::Arguments{
          // unary op: aux store D
          {
              // ternary op : beta * C + (alpha * acc)
              {{beta}},                        // leaf op+args : beta
              {ptr_C, ElementC{0}, stride_C},  // leaf op+args : C
              {
                  // binary op : alpha * acc
                  {{alpha}},  // leaf op+args : alpha
                  {},         // leaf op+args : acc
                  {}          // binary args : multiplies
              },              // end binary op
              {}              // ternary args : multiply_add
          },
          {ptr_D, stride_D}  // unary args: aux store D
      };
    } else {
      return typename EVT::Arguments{
          // unary op: aux store D
          {
              {{alpha}},  // leaf op+args : alpha
              {},         // leaf op+args : acc
              {}          // binary args : multiplies
          },
          {ptr_D, stride_D}  // unary args: aux store D
      };
    }
  }
};

}  // namespace bytedance::flux
