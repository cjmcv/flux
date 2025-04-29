//===- common_cuda.h ---------------------------------------------- C++ ---===//
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

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/float8.h"
#include "flux/flux.h"
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include <type_traits>
#include "cuda_fp16.h"
#include "cuda_bf16.h"

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                            \
  do {                                                                   \
    cutlass::Status error = status;                                      \
    FLUX_CHECK(error == cutlass::Status::kSuccess)                       \
        << "Got cutlass error: " << cutlassGetStatusString(error) << "(" \
        << static_cast<int>(error) << ") at: " << #status << "\n";       \
  } while (0)

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                                                   \
  do {                                                                                       \
    cudaError_t error = status;                                                              \
    FLUX_CHECK(error == cudaSuccess) << "Got bad cuda status: " << cudaGetErrorString(error) \
                                     << "(" << error << ") at: " << #status << "\n";         \
  } while (0)

namespace xop {
 
template <DataTypeEnum E>
auto
to_cutlass_element(cute::C<E> dtype) {
  if constexpr (dtype == _Void{}) {
    return make_declval<void>();
  } else if constexpr (dtype == _FP32{}) {
    return make_declval<float>();
  } else if constexpr (dtype == _FP16{}) {
    return make_declval<cutlass::half_t>();
  } else if constexpr (dtype == _BF16{}) {
    return make_declval<cutlass::bfloat16_t>();
  } else if constexpr (dtype == _E4M3{}) {
    return make_declval<cutlass::float_e4m3_t>();
  } else if constexpr (dtype == _E5M2{}) {
    return make_declval<cutlass::float_e5m2_t>();
  } else if constexpr (dtype == _S32{}) {
    return make_declval<int32_t>();
  } else if constexpr (dtype == _S8{}) {
    return make_declval<int8_t>();
  } else {
    static_assert(cutlass::detail::dependent_false<cute::C<E>>, "unsupported dtype!");
  }
}

template <ArchEnum E>
auto
to_cutlass_archtag(cute::C<E> arch) {
  if constexpr (arch == _Sm80{}) {
    return make_declval<cutlass::arch::Sm80>();
  } else if constexpr (arch == _Sm89{}) {
    return make_declval<cutlass::arch::Sm89>();
  } else if constexpr (arch == _Sm90{}) {
    return make_declval<cutlass::arch::Sm90>();
  } else {
    static_assert(cutlass::detail::dependent_false<cute::C<E>>, "unsupported arch!");
  }
}

inline ArchEnum get_arch() {
  int major, minor;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
  int arch_num = major * 10 + minor;
  FLUX_CHECK(arch_num == 80 || arch_num == 89 || arch_num == 90)
      << "unsupported arch: " << arch_num;
  return ArchEnum{arch_num};
}

template <GemmLayoutEnum E>
auto
to_cutlass_layout_a(cute::C<E> layout) {
  if constexpr (layout == _RCR{} or layout == _RRR{} or layout == _RCC{}) {
    return make_declval<cutlass::layout::RowMajor>();
  } else {
    static_assert(cutlass::detail::dependent_false<cute::C<E>>, "unsupported A layout!");
  }
}

template <GemmLayoutEnum E>
constexpr auto
to_cutlass_layout_b(cute::C<E> layout) {
  if constexpr (layout == _RCR{} or layout == _RCC{}) {
    return make_declval<cutlass::layout::ColumnMajor>();
  } else if constexpr (layout == _RRR{}) {
    return make_declval<cutlass::layout::RowMajor>();
  } else {
    static_assert(cutlass::detail::dependent_false<cute::C<E>>, "unsupported B layout!");
  }
}

template <GemmLayoutEnum E>
auto
to_cutlass_layout_c(cute::C<E> layout) {
  if constexpr (layout == _RCR{} or layout == _RRR{}) {
    return make_declval<cutlass::layout::RowMajor>();
  } else if constexpr (layout == _RCC{}) {
    return make_declval<cutlass::layout::ColumnMajor>();
  } else {
    static_assert(cutlass::detail::dependent_false<cute::C<E>>, "unsupported C layout!");
  }
}

/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU
 * stream
 */
struct GpuTimer {
  cudaStream_t _stream_id{};
  cudaEvent_t _start;
  cudaEvent_t _stop;

  /// Constructor
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&_start));
    CUDA_CHECK(cudaEventCreate(&_stop));
  }

  /// Destructor
  ~GpuTimer() {
    CUDA_CHECK(cudaEventDestroy(_start));
    CUDA_CHECK(cudaEventDestroy(_stop));
  }

  /// Start the timer for a given stream (defaults to the default stream)
  void
  start(cudaStream_t stream_id = nullptr) {
    _stream_id = stream_id;
    CUDA_CHECK(cudaEventRecord(_start, _stream_id));
  }

  /// Stop the timer
  void
  stop() {
    CUDA_CHECK(cudaEventRecord(_stop, _stream_id));
  }

  /// Return the elapsed time (in milliseconds)
  float
  elapsed_millis() {
    float elapsed = 0.0;
    CUDA_CHECK(cudaEventSynchronize(_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
    return elapsed;
  }
};

}  // namespace xop
