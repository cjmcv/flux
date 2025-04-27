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

#include "gemm_v2_kernel.h"
#include "flux/gemm_args.h"

namespace bytedance::flux {

template <
    class GemmMetaT,
    class GemmHParamsT,
    class GemmKernelT>
class GemmV2Single_Device
    : public GemmOperatorBase,
      public GemmV2Single_Kernel<GemmMetaT, GemmHParamsT> {

 private:
  std::any gemm_op_;

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

  void
  initialize(std::any const &args, void *workspace = nullptr, void *stream = nullptr) override {
    uint8_t *workspace_ptr = reinterpret_cast<uint8_t *>(workspace);
    std::size_t workspace_offset = 0;

    void *args_workspace = workspace_ptr + workspace_offset;
    workspace_offset += this->get_args_workspace_size(args);
    workspace_offset = cutlass::round_nearest(workspace_offset, cutlass::MinWorkspaceAlignment);
    this->initialize_args_workspace(args, args_workspace, stream);

    using Gemm = identity_t<decltype(gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;
    GemmArguments const &gemm_args = to_gemm_args(args, args_workspace);
    CUTLASS_CHECK(Gemm::can_implement(gemm_args));
    auto cu_stream = static_cast<cudaStream_t>(stream);
    void *gemm_workspace = workspace_ptr + workspace_offset;

    this->gemm_op_ = Gemm{};
    Gemm &gemm_op = std::any_cast<Gemm &>(this->gemm_op_);
    CUTLASS_CHECK(gemm_op.initialize(gemm_args, gemm_workspace, cu_stream));
  }

  void
  run(std::any const &args,
      void *workspace = nullptr,
      void *stream = nullptr,
      bool launch_with_pdl = false) override {
    this->initialize(args, workspace, stream);
    this->run(stream, launch_with_pdl);
  }

  void
  run(void *stream = nullptr, bool launch_with_pdl = false) override {
    using Gemm = identity_t<decltype(gemm_device())>;
    auto cu_stream = static_cast<cudaStream_t>(stream);
    Gemm &gemm_op = std::any_cast<Gemm &>(this->gemm_op_);
    CUTLASS_ASSERT(launch_with_pdl == false);
    CUTLASS_CHECK(gemm_op.run(cu_stream));
  }

  std::size_t
  get_workspace_size(std::any const &args) const override {
    std::size_t workspace_size = 0;
    workspace_size += this->get_args_workspace_size(args);
    workspace_size = cutlass::round_nearest(workspace_size, cutlass::MinWorkspaceAlignment);
    using Gemm = identity_t<decltype(gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;
    const GemmArguments &gemm_args = to_gemm_args(args, nullptr);
    workspace_size += Gemm::get_workspace_size(gemm_args);
    workspace_size = cutlass::round_nearest(workspace_size, cutlass::MinWorkspaceAlignment);
    return workspace_size;
  }

  std::size_t
  get_barrier_workspace_size(std::any const &) const override {
    return 0;
  }
  

  /////////////////////////////////////////////////////////////////////////////
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

 public:
  //////////////////////////
  // CRTP functions
  //////////////////////////
  auto
  gemm_device() const {
    return make_declval<cutlass::gemm::device::GemmUniversalBase<GemmKernelT>>();
  }

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
