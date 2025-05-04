#pragma once
#include <map>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"

#include "ctlop/common_cuda.h"

namespace ctlop {

template <class LayoutA, class LayoutB, class LayoutC>
class ImplHelper {
public:
  ImplHelper(int m, int n, int k): m_(m), n_(n), k_(k) {};

  int get_stride_a() const {
    if constexpr (cute::is_same_v<LayoutA, cutlass::layout::RowMajor>) {
      return k_;
    } else {
      static_assert(cute::is_same_v<LayoutA, cutlass::layout::ColumnMajor>, "requires ColumnMajor.");
      return m_;
    }
  }
  int get_stride_b() const {
    if constexpr (cute::is_same_v<LayoutB, cutlass::layout::RowMajor>) {
      return n_;
    } else {
      static_assert(cute::is_same_v<LayoutB, cutlass::layout::ColumnMajor>, "requires ColumnMajor.");
      return k_;
    }
  }
  int get_stride_c() const {
    if constexpr (cute::is_same_v<LayoutC, cutlass::layout::RowMajor>) {
      return n_;
    } else {
      static_assert(cute::is_same_v<LayoutC, cutlass::layout::ColumnMajor>, "requires ColumnMajor.");
      return m_;
    }
  }

private:
  int m_;
  int n_;
  int k_;
};

} // namespace ctlop