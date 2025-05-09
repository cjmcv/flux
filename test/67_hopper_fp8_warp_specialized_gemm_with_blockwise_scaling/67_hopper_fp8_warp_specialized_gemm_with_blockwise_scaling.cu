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
    \brief Blocked scale Hopper FP8 GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture
    This example demonstrate a blocked scaled FP8 GEMM using the new CUTLASS 3.0.
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:
    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA)
    which are more efficient than the Ampere tensor core instructions.
    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster.
    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).
    4. This example shows all important fusions used by FP8 gemm kernels, i.e., blocked scale factor for
    A, B tensor, the abs_max value of D tensor.
    5. A simple way to tune the CTA rasterization direction and swizzle pattern of Hopper kernels. Both the
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can
    improve performance.
    Examples:
      $ ./examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling  \
        --m=2816 --n=3072 --k=16384 \
        --save_aux=false --save_amax=false \
        --device_scale=false --raster=h --swizzle=2
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"

// Includes from examples directory
// #include "helper.h"
#include "ctlop/common_cuda.h"
#include "hopper_fp8_commandline.hpp"
#include "reference/host/gemm_with_blockwise_scaling.h"

#include "ctlop/ops_impl/gemm_normal/gemm_v3_blockscale_fp8_impl.h"

using namespace ctlop;
using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
// C matrix configuration
using         ElementC    = cutlass::bfloat16_t;                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
using ElementAccumulator  = float;
// Auxiliary matrix configuration and other fusion types
using         ElementAux   = ElementC;
using         LayoutAux    = LayoutC;

using         TileShape    = cute::Shape<cute::_128, cute::_128, cute::_128>;
/// Initialization
#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////
constexpr bool IsDFp8 =
  cute::is_same_v<ElementD, cutlass::float_e4m3_t> or
  cute::is_same_v<ElementD, cutlass::float_e5m2_t>;

constexpr bool IsAuxFp8 =
  cute::is_same_v<ElementAux, cutlass::float_e4m3_t> or
  cute::is_same_v<ElementAux, cutlass::float_e5m2_t>;

template <class GemmImpl>
struct Buffer {
  using ElementBlockScale = typename GemmImpl::ElementBlockScale;
  using ElementScalar     = typename GemmImpl::ElementScalar;
  using ElementAmax       = typename GemmImpl::ElementAmax;

  uint64_t seed;

  cutlass::HostTensor<ElementA  , LayoutA  > tensor_A;
  cutlass::HostTensor<ElementB  , LayoutB  > tensor_B;
  cutlass::HostTensor<ElementC  , LayoutC  > tensor_C;
  cutlass::HostTensor<ElementD  , LayoutD  > tensor_D;
  cutlass::HostTensor<ElementBlockScale, LayoutA> blockscale_tensor_A;
  cutlass::HostTensor<ElementBlockScale, LayoutB> blockscale_tensor_B;
  cutlass::HostTensor<ElementD  , LayoutD  > tensor_ref_D;
  cutlass::HostTensor<ElementAux, LayoutAux> tensor_aux;
  cutlass::HostTensor<ElementAux, LayoutAux> tensor_ref_aux;
  
  using LayoutScalar = cutlass::layout::PackedVectorLayout;
  // cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_alpha;
  // cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_beta;
  // cutlass::HostTensor<ElementScalar, LayoutScalar> scale_A;
  // cutlass::HostTensor<ElementScalar, LayoutScalar> scale_B;
  // cutlass::HostTensor<ElementScalar, LayoutScalar> scale_C;
  // cutlass::HostTensor<ElementScalar, LayoutScalar> scale_D;
  // cutlass::HostTensor<ElementScalar, LayoutScalar> scale_aux;
  cutlass::HostTensor<ElementAmax  , LayoutScalar> abs_max_D;
  cutlass::HostTensor<ElementAmax  , LayoutScalar> reference_abs_max_D;
  cutlass::HostTensor<ElementAmax  , LayoutScalar> abs_max_aux;
  cutlass::HostTensor<ElementAmax  , LayoutScalar> reference_abs_max_aux;

  /// Helper to initialize a block of device data
  template <typename Element, typename Layout>
  bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<Element>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, bits_input);
    }
    else if (dist_kind == cutlass::Distribution::AllZeros) {
      cutlass::reference::host::TensorFill(view);
    }
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
    }
    else {
      throw std::runtime_error("Not implementated.");
    }

    return true;
  }

  /// Helper to initialize a block of device data (scale_tensors)
  template <typename Element, typename Layout>
  bool initialize_scale_tensor(
    cutlass::TensorView<Element, Layout> view,
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      double scope_max, scope_min;

      scope_min = -1;
      scope_max = 1;

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min);
    }
    else if (dist_kind == cutlass::Distribution::AllZeros) {
      cutlass::reference::host::TensorFill(view);
    }
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {

      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
    }
    else {
      throw std::runtime_error("Not implementated.");
    }

    return true;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const Options &options) {

    // Find Block Scaling tensor shapes based on problem shape and TileShape
    auto gemm_problem_shape = cute::make_shape(options.m, options.n, options.k);
    auto blockscale_shape = shape(get<1>(cute::zipped_divide(cute::make_layout(gemm_problem_shape), TileShape{})));
    auto blockscale_m = cute::get<0>(blockscale_shape);
    auto blockscale_n = cute::get<1>(blockscale_shape);
    auto blockscale_k = cute::get<2>(blockscale_shape);

    auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
    auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);
    auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
    auto blockscale_a_coord = cutlass::make_Coord(blockscale_m * options.l, blockscale_k);
    auto blockscale_b_coord = cutlass::make_Coord(blockscale_k, blockscale_n * options.l);

    tensor_A.resize(a_coord);
    blockscale_tensor_A.resize(blockscale_a_coord);
    tensor_B.resize(b_coord);
    blockscale_tensor_B.resize(blockscale_b_coord);
    tensor_C.resize(c_coord);
    tensor_D.resize(c_coord);
    tensor_ref_D.resize(c_coord);

    cutlass::Distribution::Kind dist_A = cutlass::Distribution::Uniform;
    cutlass::Distribution::Kind dist_B = cutlass::Distribution::Uniform;
    cutlass::Distribution::Kind dist_C = cutlass::Distribution::Uniform;
    cutlass::Distribution::Kind dist_scaleA = cutlass::Distribution::Uniform;
    cutlass::Distribution::Kind dist_scaleB = cutlass::Distribution::Uniform;

    initialize_tensor(tensor_A.host_view(), dist_A, seed + 2022);
    initialize_tensor(tensor_B.host_view(), dist_B, seed + 2023);
    initialize_tensor(tensor_C.host_view(), dist_C, seed + 2024);
    initialize_scale_tensor(blockscale_tensor_A.host_view(), dist_scaleA, seed + 2025);
    initialize_scale_tensor(blockscale_tensor_B.host_view(), dist_scaleB, seed + 2026);

  #if 0 // Dump blockscaled tensors
    std::cout << "blockscale_tensor_A: " << blockscale_a_coord << std::endl;
    std::cout << blockscale_tensor_A.host_view() << "\n";
    std::cout << "blockscale_tensor_B: " << blockscale_b_coord << std::endl;
    std::cout << blockscale_tensor_B.host_view() << "\n";
  #endif

    // Print block scaling tensors on the host side.
    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
    blockscale_tensor_A.sync_device();
    blockscale_tensor_B.sync_device();

    if (options.save_aux) {
      tensor_aux.resize(c_coord);
      tensor_aux.sync_device();
      tensor_ref_aux.resize(c_coord);
    }

    // if (options.device_scale) {
    //   scalar_alpha.resize(cutlass::make_Coord(1));
    //   scalar_beta.resize(cutlass::make_Coord(1));
    //   scale_A.resize(cutlass::make_Coord(1));
    //   scale_B.resize(cutlass::make_Coord(1));
    //   scale_C.resize(cutlass::make_Coord(1));
    //   scale_D.resize(cutlass::make_Coord(1));
    //   scale_aux.resize(cutlass::make_Coord(1));

    //   cutlass::reference::host::TensorFill(scalar_alpha.host_view(), options.alpha);
    //   cutlass::reference::host::TensorFill(scalar_beta.host_view(), options.beta);
    //   cutlass::reference::host::TensorFill(scale_A.host_view(), options.scale_a);
    //   cutlass::reference::host::TensorFill(scale_B.host_view(), options.scale_b);
    //   cutlass::reference::host::TensorFill(scale_C.host_view(), options.scale_c);
    //   cutlass::reference::host::TensorFill(scale_D.host_view(), options.scale_d);
    //   cutlass::reference::host::TensorFill(scale_aux.host_view(), options.scale_aux);

    //   scalar_alpha.sync_device();
    //   scalar_beta.sync_device();
    //   scale_A.sync_device();
    //   scale_B.sync_device();
    //   scale_C.sync_device();
    //   scale_D.sync_device();
    //   scale_aux.sync_device();
    // }

    if (IsDFp8 && options.save_amax) {
      abs_max_D.resize(cutlass::make_Coord(1));
      initialize_tensor(abs_max_D.host_view(), cutlass::Distribution::AllZeros, 0);
      abs_max_D.sync_device();
      reference_abs_max_D.resize(cutlass::make_Coord(1));
      initialize_tensor(reference_abs_max_D.host_view(), cutlass::Distribution::AllZeros, 0);
    }

    if (IsAuxFp8 && options.save_aux && options.save_amax) {
      abs_max_aux.resize(cutlass::make_Coord(1));
      initialize_tensor(abs_max_aux.host_view(), cutlass::Distribution::AllZeros, 0);
      abs_max_aux.sync_device();
      reference_abs_max_aux.resize(cutlass::make_Coord(1));
      initialize_tensor(reference_abs_max_aux.host_view(), cutlass::Distribution::AllZeros, 0);
    }
  }  
};



template <class GemmImpl>
bool verify(const Options &options, Buffer<GemmImpl> &buffer) {
  //
  // Compute reference output
  //
  using ElementCompute    = typename GemmImpl::ElementCompute;
  using Gemm              = typename GemmImpl::Gemm;
  using ElementScalar     = typename GemmImpl::ElementScalar;
  using ElementAmax       = typename GemmImpl::ElementAmax;
  using ActivationFunctor = typename GemmImpl::Gemm::EpilogueOutputOp::ActivationFn;

  using StrideA = typename GemmImpl::StrideA;
  using StrideB = typename GemmImpl::StrideB;
  using StrideC = typename GemmImpl::StrideC;
  using StrideD = typename GemmImpl::StrideD;
  using StrideAux = typename GemmImpl::StrideAux;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));
  StrideAux stride_aux = stride_D;

  // Block scaling tensors shapes based CTA Block (TileShape) and GEMM Problem shape
  auto gemm_problem_shape = cute::make_shape(options.m, options.n, options.k);
  auto blockscale_m = ceil_div(options.m, get<0>(TileShape{}));
  auto blockscale_n = ceil_div(options.n, get<1>(TileShape{}));
  auto blockscale_k = ceil_div(options.k, get<2>(TileShape{}));

  // Create instantiation for device reference gemm kernel
  auto A = cute::make_tensor(buffer.tensor_A.host_data(),
                             cute::make_layout(
                                cute::make_shape(options.m, options.k, options.l),
                                stride_A
                              )
                            );
  auto B = cute::make_tensor(buffer.tensor_B.host_data(),
                             cute::make_layout(
                               cute::make_shape(options.n, options.k, options.l),
                               stride_B
                              )
                            );
  auto C = cute::make_tensor(buffer.tensor_C.host_data(),
                             cute::make_layout(
                                cute::make_shape(options.m, options.n, options.l),
                                stride_C
                              )
                            );
  auto D = cute::make_tensor(buffer.tensor_ref_D.host_data(),
                             cute::make_layout(
                                cute::make_shape(options.m, options.n, options.l),
                                stride_D
                              )
                            );
  auto Aux = cute::make_tensor(buffer.tensor_ref_aux.host_data(),
                               cute::make_layout(
                                  cute::make_shape(options.m, options.n, options.l),
                                  stride_aux
                                )
                              );

  auto blockscale_A = cute::make_tensor(buffer.blockscale_tensor_A.host_data(),
                                        cute::make_layout(
                                          cute::make_shape(blockscale_m, blockscale_k, options.l),
                                          cute::make_stride(1, blockscale_m, blockscale_m * blockscale_k)
                                        )
                                      );
  auto blockscale_B = cute::make_tensor(buffer.blockscale_tensor_B.host_data(),
                                        cute::make_layout(
                                          cute::make_shape(blockscale_n, blockscale_k, options.l),
                                          cute::make_stride(1, blockscale_n, blockscale_n * blockscale_k)
                                        )
                                      );

  using unused_t = decltype(D);

  cutlass::reference::host::GettMainloopParams<ElementAccumulator,
                                               decltype(A), decltype(B),
                                               decltype(blockscale_A), decltype(blockscale_B),
                                               TileShape> mainloop_params{
                                               A, B,                         // Operand Tensors
                                               blockscale_A, blockscale_B    // Blockwise scaling Tensors
                                              };

  cutlass::reference::host::GettEpilogueParams<
      ElementScalar,
      ElementScalar,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D),
      unused_t, // bias
      decltype(Aux),
      unused_t, // valpha
      unused_t, // vbeta
      ActivationFunctor
  > epilogue_params;

  epilogue_params.C = C;
  epilogue_params.D = D;
  epilogue_params.Aux = Aux;
  epilogue_params.alpha = options.alpha;
  epilogue_params.beta = options.beta;
  epilogue_params.scale_a = options.scale_a;
  epilogue_params.scale_b = options.scale_b;
  epilogue_params.scale_c = options.scale_c;
  epilogue_params.scale_d = options.scale_d;
  epilogue_params.scale_aux = options.scale_aux;
  epilogue_params.abs_max_D = buffer.reference_abs_max_D.host_data();
  epilogue_params.abs_max_Aux = buffer.reference_abs_max_aux.host_data();

  // get reference result
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // compare_reference
  bool passed = true;
  buffer.tensor_D.sync_host();
  passed &= cutlass::reference::host::TensorRelativelyEquals(buffer.tensor_D.host_view(), buffer.tensor_ref_D.host_view(), ElementAux(options.epsilon), ElementAux(options.non_zero_floor));
  double mse = cutlass::reference::host::TensorMSE(buffer.tensor_D.host_view(), buffer.tensor_ref_D.host_view());
  double mre = cutlass::reference::host::TensorMRE(buffer.tensor_D.host_view(), buffer.tensor_ref_D.host_view());
  double max_error = cutlass::reference::host::TensorGreatestError(buffer.tensor_D.host_view(), buffer.tensor_ref_D.host_view());
  std::cout << "  Result MSE: " << mse << ", MRE: " << mre << ", greatest error: " << max_error << std::endl;

#if 0
  std::cout << "tensor_ref_D.host_view() {" << std::endl
            << tensor_ref_D.host_view() << std::endl
            << "}"  << std::endl;
  std::cout << "tensor_D.host_view() {" << std::endl
            << tensor_D.host_view() << std::endl
            << "}"  << std::endl;
#endif

  if (IsDFp8 && options.save_amax) {
    buffer.abs_max_D.sync_host();
    std::cout << "  Abs max D: " << buffer.abs_max_D.at(cutlass::make_Coord(0)) << ", reference: " << buffer.reference_abs_max_D.at(cutlass::make_Coord(0)) << std::endl;
    passed &= cutlass::relatively_equal(buffer.abs_max_D.at(cutlass::make_Coord(0)), buffer.reference_abs_max_D.at(cutlass::make_Coord(0)), ElementScalar(options.epsilon), ElementScalar(options.non_zero_floor));
  }

  if (options.save_aux) {
    buffer.tensor_aux.sync_host();
    passed &= cutlass::reference::host::TensorRelativelyEquals(buffer.tensor_aux.host_view(), buffer.tensor_ref_aux.host_view(), ElementAux(options.epsilon), ElementAux(options.non_zero_floor));
    mse = cutlass::reference::host::TensorMSE(buffer.tensor_aux.host_view(), buffer.tensor_ref_aux.host_view());
    mre = cutlass::reference::host::TensorMRE(buffer.tensor_aux.host_view(), buffer.tensor_ref_aux.host_view());
    max_error = cutlass::reference::host::TensorGreatestError(buffer.tensor_aux.host_view(), buffer.tensor_ref_aux.host_view());
    std::cout << "  Aux MSE: " << mse << ", MRE: " << mre << ", greatest error: " << max_error << std::endl;
    if (IsAuxFp8 && options.save_amax) {
      buffer.abs_max_aux.sync_host();
      std::cout << "  Abs max aux: " << buffer.abs_max_aux.at(cutlass::make_Coord(0)) << ", reference: " << buffer.reference_abs_max_aux.at(cutlass::make_Coord(0)) << std::endl;
      passed &= cutlass::relatively_equal(buffer.abs_max_aux.at(cutlass::make_Coord(0)), buffer.reference_abs_max_aux.at(cutlass::make_Coord(0)), ElementScalar(options.epsilon), ElementScalar(options.non_zero_floor));
    }
  }

  return passed;
}

/// Execute a given example GEMM computation
int run(Options &options)
{  
  using GemmFp8Impl = GemmBlockScaleFp8Impl<ElementA,ElementB,ElementC,float,LayoutA,LayoutB,LayoutC, cutlass::arch::Sm90, cutlass::gemm::PersistentScheduler, TileShape, Shape<_1,_2,_1>, RasterOrderOptions::AlongN, 2>;
  //
  using ME = UnifiedMetaEnum;
  GemmConfigRegister& ins = GemmConfigRegister::instance();
  std::vector<int8_t> id_meta = {0,(int8_t)ME::GemmBolckScaleFp8,(int8_t)ME::E4M3, (int8_t)ME::E4M3, (int8_t)ME::BF16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm90};
  ins.add(id_meta, /*op*/[]() { return new GemmFp8Impl();});
  //

  Buffer<GemmFp8Impl> buffer;
  buffer.initialize(options);

  RtBlockScaleFp8ArgumentsV3 rt_args;
  {
    rt_args.m = options.m;
    rt_args.n = options.n;
    rt_args.k = options.k;
    rt_args.l = options.l;
  
    rt_args.alpha = options.alpha;
    rt_args.beta = options.beta;
    // rt_args.d_scalar_alpha = buffer.scalar_alpha.device_data();
    // rt_args.d_scalar_beta = buffer.scalar_beta.device_data();
  
    //
    rt_args.scale_a = 1.f, rt_args.scale_b = 1.f, rt_args.scale_c = 1.f, rt_args.scale_d = 1.f, rt_args.scale_aux = 1.f;
    // rt_args.d_scale_A = buffer.scale_A.device_data();
    // rt_args.d_scale_B = buffer.scale_B.device_data();
    // rt_args.d_scale_C = buffer.scale_C.device_data();
    // rt_args.d_scale_D = buffer.scale_D.device_data();
    // rt_args.d_scale_aux = buffer.scale_aux.device_data();

    rt_args.ptr_A = buffer.tensor_A.device_data();
    rt_args.ptr_B = buffer.tensor_B.device_data();
    rt_args.ptr_C = buffer.tensor_C.device_data();
    rt_args.ptr_D = buffer.tensor_D.device_data();
  
    rt_args.d_blockscale_A = buffer.blockscale_tensor_A.device_data();
    rt_args.d_blockscale_B = buffer.blockscale_tensor_B.device_data();
  
    // debug
    rt_args.save_aux = options.save_aux;
    rt_args.save_amax = options.save_amax;
    rt_args.d_tensor_aux = buffer.tensor_aux.device_data();
    rt_args.d_abs_max_aux = buffer.abs_max_aux.device_data();
    rt_args.d_abs_max_D = buffer.abs_max_D.device_data();
  }

  // GemmFp8Impl gemm;
  GemmBase *gemm = ins.GetOp(id_meta, false);
  gemm->initialize(&rt_args);
  gemm->run();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  if (options.verify) {
    result.passed = verify<GemmFp8Impl>(options, buffer);

    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;
  }
  else {
    result.passed = true;
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    for (int iter = 0; iter < options.warmup + options.iterations; ++iter) {
      if (iter == options.warmup)
        timer.start();
      gemm->run();
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    std::cout << "  Rasterization: " << "Along N" << " with a maximum CTA swizzle of " << 1 << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return result.passed;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
    return 0;
  }
  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  bool passed = run(options);
  if (!passed)
    return -1;
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
