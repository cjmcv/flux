
#pragma once

#ifdef ENABLE_QWEN3_06B
#include "m1/silu_mul_tl_1_3072.cuh"
// #include "m32/silu_mul_tl_32_3072.cuh"
#endif

#ifdef ENABLE_QWEN3_4B
#include "m1/silu_mul_tl_1_9728.cuh"
#include "m32/silu_mul_tl_32_9728.cuh"
#endif

namespace kernel {

template <typename T,
  int THREAD_NUM,
  int TILE_DIM_X, 
  int TILE_DIM_Y, 
  int TILE_DIM_Z,
  int M,
  int N,
  int I_STRIDE,
  int O_STRIDE>
__device__ __forceinline__ void silu_mul_kernel(const int bx, const int by, const int bz,
                                           void const *input_ptr,
                                           void *output_ptr,
                                           int num_active_tokens) {

#ifdef ENABLE_QWEN3_06B
  if constexpr (M == 1) { 
    if constexpr (N == 3072) {
      silu_mul_kernel_1_3072<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M,N,I_STRIDE,O_STRIDE>(bx, by, bz, input_ptr, output_ptr, num_active_tokens);return;
    }
  }
  // else if constexpr (M == 32) {           
  //   if constexpr (N == 3072) {
  //     silu_mul_kernel_32_3072<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M,N,I_STRIDE,O_STRIDE>(bx, by, bz, input_ptr, output_ptr, num_active_tokens);return;
  //   }
  // }
#endif // ENABLE_QWEN3_06B

#ifdef ENABLE_QWEN3_4B
  if constexpr (M == 1) { 
    if constexpr (N == 9728) {
      silu_mul_kernel_1_9728<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M,N,I_STRIDE,O_STRIDE>(bx, by, bz, input_ptr, output_ptr, num_active_tokens);return;
    }
  }
  else if constexpr (M == 32) {           
    if constexpr (N == 9728) {                                   
      silu_mul_kernel_32_9728<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M,N,I_STRIDE,O_STRIDE>(bx, by, bz, input_ptr, output_ptr, num_active_tokens);return;
    }
  }
#endif // ENABLE_QWEN3_4B

  printf("Error: [silu_mul_kernel_%d_%d] There is no suitable microkernel!\n", M,N);
}

} // namespace kernel
