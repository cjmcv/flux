
#pragma once

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/float8.h"
#include "xop/xop.h"
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
    XOP_CHECK(error == cutlass::Status::kSuccess)                       \
        << "Got cutlass error: " << cutlassGetStatusString(error) << "(" \
        << static_cast<int>(error) << ") at: " << #status << "\n";       \
  } while (0)

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                                                   \
  do {                                                                                       \
    cudaError_t error = status;                                                              \
    XOP_CHECK(error == cudaSuccess) << "Got bad cuda status: " << cudaGetErrorString(error) \
                                     << "(" << error << ") at: " << #status << "\n";         \
  } while (0)

namespace xop {

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
