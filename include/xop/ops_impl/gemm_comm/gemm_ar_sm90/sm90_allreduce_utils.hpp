
#pragma once
#include "cute/arch/cluster_sm90.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"
#include "cutlass/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "xop/xop.h"

//////////////////////////////////////////////
// reference: cutlass/arch/memory.h
namespace cutlass {
namespace arch {
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

}  // namespace arch
}  // namespace cutlass

namespace xop {


}  // namespace bytedance::flux
