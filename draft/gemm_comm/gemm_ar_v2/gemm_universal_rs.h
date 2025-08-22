/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief
*/

#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
// #include "cutlass/gemm/kernel/gemm_universal.h"
#include "gemm_universal_rs.h"

#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
// #include "cutlass/gemm/device/gemm_universal_base.h"
#include "gemm_universal_base_rs.h"

#include "cutlass/layout/permute.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

  
/////////////////////////////////////////////////////////////////////////////////////////////////

/*! 
  GemmUniversal is a stateful, reusable GEMM handle.  Once initialized for a given GEMM computation
  (problem geometry and data references), it can be reused across different GEMM problems having the
  geometry.  (Once initialized, details regarding problem geometry and references to workspace memory
  cannot be updated.)

  The universal GEMM accommodates serial reductions, parallel reductions, batched strided, and 
  batched array variants.
*/
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for.  This is the minimum SM that
    /// supports the intended feature. The device kernel can be built
    /// targeting any SM larger than this number.
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ = threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA = ComplexTransform::kNone,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB = ComplexTransform::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout_ = layout::NoPermute,
    /// Permute operand A
    typename PermuteALayout_ = layout::NoPermute,
    /// Permute operand B
    typename PermuteBLayout_ = layout::NoPermute
>
class GemmUniversalRs : 
  public GemmUniversalBaseRs<
    typename kernel::DefaultGemmUniversalRs<
      ElementA_,
      LayoutA_,
      TransformA,
      AlignmentA,
      ElementB_,
      LayoutB_,
      TransformB,
      AlignmentB,
      ElementC_,
      LayoutC_,
      ElementAccumulator_,
      OperatorClass_,
      ArchTag_,
      ThreadblockShape_,
      WarpShape_,
      InstructionShape_,
      EpilogueOutputOp_,
      ThreadblockSwizzle_,
      Stages,
      Operator_,
      SharedMemoryClearOption::kNone,
      GatherA,
      GatherB,
      ScatterD,
      PermuteDLayout_,
      PermuteALayout_,
      PermuteBLayout_
    >::GemmKernel
  > {

 public:

  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  using PermuteDLayout = PermuteDLayout_;
  using PermuteALayout = PermuteALayout_;
  using PermuteBLayout = PermuteBLayout_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;

  using Base = GemmUniversalBaseRs<
    typename kernel::DefaultGemmUniversalRs<
      ElementA_,
      LayoutA_,
      TransformA,
      AlignmentA,
      ElementB_,
      LayoutB_,
      TransformB,
      AlignmentB,
      ElementC_,
      LayoutC_,
      ElementAccumulator_,
      OperatorClass_,
      ArchTag_,
      ThreadblockShape_,
      WarpShape_,
      InstructionShape_,
      EpilogueOutputOp_,
      ThreadblockSwizzle_,
      Stages,
      Operator_,
      SharedMemoryClearOption::kNone,
      GatherA,
      GatherB,
      ScatterD,
      PermuteDLayout_,
      PermuteALayout_,
      PermuteBLayout_
    >::GemmKernel
  >;

  using Arguments = typename Base::Arguments;
  using GemmKernel = typename Base::GemmKernel;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
