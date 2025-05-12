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
    \brief Grouped scale Hopper FP8 Grouped GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture
    This example demonstrates a grouped scaled FP8 Grouped GEMM using the new CUTLASS 3.0.
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:
    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA)
    which are more efficient than the Ampere tensor core instructions.
    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster. This example also showcases on-the-fly modification of TMA
    descriptors to move between groups/problem_count (represented by groups).
    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).
    4. A simple way to tune the CTA rasterization direction and swizzle pattern of Hopper kernels. Both the
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can
    improve performance.
    Examples:
      $ ./examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling  \
        --m=2816 --n=3072 --k=16384 --save_aux=false --save_amax=false \
        --raster=h --swizzle=2 --benchmark=./test_benchmark.txt

      Where the test_benchmark.txt may look as such:
        0 256x512x128
        1 256x512x512
        2 512x256x128
        3 256x256x128
        4 256x512x1024
        5 1024x512x128 and so on
*/

#include <iostream>
#include <optional>
#include <fstream>
#include <sstream>
#include <vector>
#include <cfloat>

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
#include "cutlass/util/reference/device/tensor_fill.h"

// #include "helper.h"
#include "ctlop/common_cuda.h"
#include "hopper_fp8_commandline.hpp"
#include "reference/host/gemm_with_groupwise_scaling.h"
#include "ctlop/ops_impl/gemm_normal/gemm_v3_grouped_blockscale_fp8_impl.h"

using namespace ctlop;
using namespace cute;

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
using         ElementC    = cutlass::bfloat16_t;                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementBlockScale   = float;                                          // Element type for blockscaling during accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation

using TileShape     = Shape<_128,_128,_128>;

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 3)) {
    std::cerr << "This example requires CUDA 12.3 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  std::vector<int64_t> offset_A;
  std::vector<int64_t> offset_B;
  std::vector<int64_t> offset_C;
  std::vector<int64_t> offset_D;
  std::vector<int64_t> offset_blockscale_A;
  std::vector<int64_t> offset_blockscale_B;

  std::vector<ElementAccumulator> alpha_host;
  std::vector<ElementAccumulator> beta_host;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementD> block_D;
  cutlass::DeviceAllocation<ElementBlockScale> blockscale_block_A;
  cutlass::DeviceAllocation<ElementBlockScale> blockscale_block_B;

  cutlass::DeviceAllocation<const ElementA *> ptr_A;
  cutlass::DeviceAllocation<const ElementB *> ptr_B;
  cutlass::DeviceAllocation<const ElementC *> ptr_C;
  cutlass::DeviceAllocation<ElementD *> ptr_D;
  cutlass::DeviceAllocation<ElementD *> ptr_ref_D;
  cutlass::DeviceAllocation<const ElementBlockScale *> ptr_blockscale_A;
  cutlass::DeviceAllocation<const ElementBlockScale *> ptr_blockscale_B;

  cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
  cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
  cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
  cutlass::DeviceAllocation<ElementAccumulator> block_beta;


  std::vector<ElementA> block_A_host(block_A.size());
  std::vector<ElementB> block_B_host(block_B.size());
  std::vector<ElementC> block_C_host(block_C.size());
  std::vector<ElementD> block_D_host_kernel(block_D.size());
  std::vector<ElementD> block_D_host_ref(block_D.size());
  std::vector<ElementBlockScale> blockscale_block_A_host(blockscale_block_A.size());
  std::vector<ElementBlockScale> blockscale_block_B_host(blockscale_block_B.size());

  ///////////
  using GemmGroupedFp8Impl = GemmGroupedBlockScaleFp8Impl<ElementA,ElementB,ElementC,float,LayoutA,LayoutB,LayoutC, cutlass::arch::Sm90, cutlass::gemm::PersistentScheduler, TileShape, Shape<_1,_2,_1>, RasterOrderOptions::AlongN, 2>;

  using ProblemShape = typename GemmGroupedFp8Impl::ProblemShape;
  RtGroupedBlockScaleFp8ArgumentsV3 *rt_args = new RtGroupedBlockScaleFp8ArgumentsV3();
  {
    int m = 1024, n = 512, k = 1024;
    rt_args->groups = 10;
    for (int i=0; i <rt_args->groups; i++) {
      rt_args->problem_sizes.push_back(m);
      rt_args->problem_sizes.push_back(n);
      rt_args->problem_sizes.push_back(k);
    }
  }



  const int ScaleMsPerTile = 128;
  const int ScaleNsPerTile = 1;

  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;
  int64_t total_elements_blockscale_A = 0;
  int64_t total_elements_blockscale_B = 0;

  offset_A.clear();
  offset_B.clear();
  offset_C.clear();
  offset_D.clear();
  offset_blockscale_A.clear();
  offset_blockscale_B.clear();
  
  for (int32_t i = 0; i < rt_args->groups; ++i) {

    auto M = rt_args->problem_sizes[3*i+0];
    auto N = rt_args->problem_sizes[3*i+1];
    auto K = rt_args->problem_sizes[3*i+2];
    cute::Shape<int,int,int> problem = {M,N,K};
    
    auto blockscale_shape = shape(get<1>(cute::zipped_divide(cute::make_layout(problem), TileShape{})));
    auto groupscale_m = cute::get<0>(blockscale_shape) * ScaleMsPerTile; // We need to pad along M in scale tensor of A to prevent illegal memory access.
    auto groupscale_n = cute::get<1>(blockscale_shape) * ScaleNsPerTile; // We need to pad along N in scale tensor of A to prevent illegal memory access.
    auto blockscale_k = cute::get<2>(blockscale_shape);

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);
    offset_blockscale_A.push_back(total_elements_blockscale_A);
    offset_blockscale_B.push_back(total_elements_blockscale_B);

    int64_t elements_A = M * K;
    int64_t elements_B = K * N;
    int64_t elements_C = M * N;
    int64_t elements_D = M * N;
    int64_t elements_blockscale_A = groupscale_m * blockscale_k;
    int64_t elements_blockscale_B = groupscale_n * blockscale_k;

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
    total_elements_blockscale_A += elements_blockscale_A;
    total_elements_blockscale_B += elements_blockscale_B;
  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_alpha.reset(rt_args->groups);
  block_beta.reset(rt_args->groups);
  blockscale_block_A.reset(total_elements_blockscale_A);
  blockscale_block_B.reset(total_elements_blockscale_B);

  ///////////////////////////////////////////////////////////////////////////

  

  rt_args->problem_sizes;
  rt_args->groups;

  rt_args->ptr_A;
  rt_args->ptr_B;
  rt_args->ptr_C;
  rt_args->ptr_D;  

  rt_args->alpha = 1.0;
  rt_args->beta = 0.0;


  return 0;
}

///////////////////////////////////////////////////////////////////////////
