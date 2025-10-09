//===- memory_utils.hpp ------------------------------------------- C++ ---===//
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

#pragma once

#include "cute/arch/copy.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/barrier.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "system_barrier.hpp"

namespace xop {

enum class FluxNamedBarriers : int {
  FirstBarrier = static_cast<int>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier),
  ReduceScatterEpilogue = FirstBarrier,
  ReduceScatterFetch = FirstBarrier + 1,
  ReduceScatterReduce = FirstBarrier + 2,
  AGScatterGather = FirstBarrier,
  AGScatterFetcher = FirstBarrier + 1,
  GatherRSProducer = FirstBarrier,
  GatherRSConsumer = FirstBarrier + 1
};
}  // namespace bytedance::flux

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

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

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace arch
}  // namespace cutlass
