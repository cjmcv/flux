
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/barrier.h"

namespace cutlass {

namespace detail {
/////////////////////////////////////////////////////////////////////////////////////////////////

struct SingleThreadSync {
  CUTLASS_DEVICE
  static void sync() {}
};

template <class Sync>
struct GenericSystemBarrier : public GenericBarrier<Sync> {
 protected:
  /// Load flag, as a strong acquire operation (int specialization)
  CUTLASS_DEVICE
  static int
  ld_acquire(int *ptr) {
    int state = 0;
    // Acquire pattern using acquire modifier
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
    return state;
  }

  /// Reduce into flag, with release pattern (int specialization)
  CUTLASS_DEVICE
  static void
  red_release(int *ptr, int val) {
    // Release pattern using acq_rel fence + relaxed modifier.  (The fence also releases data
    // that was weakly-written by other threads prior to the last syncthreads)
    asm volatile("fence.acq_rel.sys;\n");
    asm volatile("red.relaxed.sys.global.add.s32 [%0], %1;\n" : : "l"(ptr), "r"(val));
  }

 public:
  /// Uses thread[0] to wait for at least the specified count of signals on the given flag counter
  // CUTLASS_DEVICE
  // static int
  // check_value(void *lock_ptr, int thread_idx, int flag_idx) {
  //   int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
  //   return ld_acquire(flag_ptr);
  // }

  // CUTLASS_DEVICE
  // static void
  // wait_lt(void *lock_ptr, int thread_idx, int flag_idx, int count) {
  //   int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;

  //   // clang-format off
  //   if (thread_idx == 0) {
  //     // Spin-loop
  //     #pragma unroll 1
  //     while (ld_acquire(flag_ptr) < count) {}
  //   }
  //   // clang-format on
  //   Sync::sync();
  // }

  // CUTLASS_DEVICE
  // static void
  // wait_eq(void *lock_ptr, int thread_idx, int flag_idx, int val) {
  //   int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
  //   // clang-format off
  //   if (thread_idx == 0) {
  //     #pragma unroll 1
  //     while (ld_acquire(flag_ptr) != val) {}
  //   }
  //   // clang-format on
  //   Sync::sync();
  // }

  CUTLASS_DEVICE
  static void
  wait_eq_reset(void *lock_ptr, int thread_idx, int flag_idx, int val, int reset_val = 0) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
    // clang-format off
    if (thread_idx == 0) {
      #pragma unroll 1
      while(atomicCAS_system(flag_ptr, val, reset_val) != val) {}
    }
    // clang-format on
    Sync::sync();
  }

  // CUTLASS_DEVICE
  // static void
  // arrive_inc(void *lock_ptr, int thread_idx, int flag_idx, int val = 1) {
  //   int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;

  //   Sync::sync();
  //   if (thread_idx == 0) {
  //     red_release(flag_ptr, val);
  //   }
  // }

  CUTLASS_DEVICE
  static int
  arrive_inc_get(void *lock_ptr, int thread_idx, int flag_idx, int val = 1) {
    // only usable for warp sync
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;

    Sync::sync();
    int old_val = 0;

    if (thread_idx == 0) {
      asm volatile("fence.acq_rel.sys;\n");
      old_val = atomicAdd_system(flag_ptr, val); // red.relaxed.sys.global.add.s32
    }

    if constexpr (cute::is_same_v<Sync, detail::SingleThreadSync>) {
      return old_val + val;
    } else {
      // T __shfl_sync(unsigned mask, T var, int srcLane);
      // 0xffffffff: all 32 threads in the warp.  0: srcLane
      // The value of old_val + val computed by lane 0 is broadcast to all 32 threads in the warp, and the result is stored in ret.
      int ret = __shfl_sync(0xffffffff, old_val + val, 0);
      return ret;
    }
  }
};

template <class Sync>
struct CustomizedGenericBarrier : public GenericBarrier<Sync> {
  CUTLASS_DEVICE
  static void
  wait_eq_reset(void *lock_ptr, int thread_idx, int flag_idx, int val, int reset_val = 0) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
    // clang-format off
    if (thread_idx == 0) {
      #pragma unroll 1
      while(atomicCAS(flag_ptr, val, reset_val) != val) {}
    }
    // clang-format on
    Sync::sync();
  }

  CUTLASS_DEVICE
  static int
  arrive_inc_get(void *lock_ptr, int thread_idx, int flag_idx, int val = 1) {
    // only usable for warp sync
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;

    Sync::sync();
    int old_val = 0;

    if (thread_idx == 0) {
      asm volatile("fence.acq_rel.gpu;\n");
      old_val = atomicAdd(flag_ptr, val);
    }

    if constexpr (cute::is_same_v<Sync, detail::SingleThreadSync>) {
      return old_val + val;
    } else {
      int ret = __shfl_sync(0xffffffff, old_val + val, 0);
      return ret;
    }
  }
};

using SystemBarrier = GenericSystemBarrier<detail::SyncthreadsSync>;

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace detail

}  // namespace cutlass
