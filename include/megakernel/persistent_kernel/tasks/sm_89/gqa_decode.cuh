
#pragma once

#ifdef ENABLE_QWEN3_06B
#include "m1/gqa_decode_tl_1_8192_16_16_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_32_16_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_64_16_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_128_16_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_256_16_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_512_16_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_1024_16_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_2048_16_8_128.cuh"
#endif // ENABLE_QWEN3_06B

#ifdef ENABLE_QWEN3_4B
#include "m1/gqa_decode_tl_1_8192_16_32_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_32_32_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_64_32_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_128_32_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_256_32_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_512_32_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_1024_32_8_128.cuh"
#include "m1/gqa_decode_tl_1_8192_2048_32_8_128.cuh"
#endif // ENABLE_QWEN3_4B

namespace kernel {

template <typename T,
  int THREAD_NUM,
  int SUB_KERNEL_ID,
  int M, 
  int HEAD,
  int GROUPS,
  int DIM>
__device__ __forceinline__ void gqa_decode_kernel(const int bx, const int by, const int bz,
                                           const void* __restrict__ q, 
                                           const void* __restrict__ k, 
                                           const void* __restrict__ v,
                                           const void* __restrict__ edge_ptr,
                                           const void* __restrict__ mask_ptr, 
                                           void* __restrict__ output_ptr,
                                           void* __restrict__ glse_ptr,
                                           void* __restrict__ output_partial_ptr) {
  const int step = ((int*)edge_ptr)[0];
  // printf("step: %d, (%d,%d,%d).\n", step, M,HEAD,SUB_KERNEL_ID);
#ifdef ENABLE_QWEN3_06B
  if constexpr (M == 1) { 
    if constexpr (HEAD == 16) {
      if constexpr (SUB_KERNEL_ID == 0) {
        if (step >= 0 && step < 16) {
          flashattn_kernel_1_8192_16_16_8_128<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 16 && step < 32) {
          flashattn_kernel_1_8192_32_16_8_128<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 32 && step < 64) {
          flashattn_kernel_1_8192_64_16_8_128<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 64 && step < 128) {
          flashattn_kernel_1_8192_128_16_8_128<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 128 && step < 256) {
          flashattn_kernel_1_8192_256_16_8_128<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 256 && step < 512) {
          flashattn_kernel_1_8192_512_16_8_128__0<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 512 && step < 1024) {
          flashattn_kernel_1_8192_1024_16_8_128__0<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 1024 && step < 2048) {
          flashattn_kernel_1_8192_2048_16_8_128__0<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
      }
      else if constexpr (SUB_KERNEL_ID == 1) {
        if (step >= 0 && step < 64) {}
        // else if (step >= 64 && step < 128) {
        //   flashattn_kernel_1_8192_128_16_8_128__1<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        // }
        // else if (step >= 128 && step < 256) {
        //   flashattn_kernel_1_8192_256_16_8_128__1<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        // }
        else if (step >= 256 && step < 512) {
          flashattn_kernel_1_8192_512_16_8_128__1<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 512 && step < 1024) {
          flashattn_kernel_1_8192_1024_16_8_128__1<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 1024 && step < 2048) {
          flashattn_kernel_1_8192_2048_16_8_128__1<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
      }
    }
  }
#endif // ENABLE_QWEN3_06B

#ifdef ENABLE_QWEN3_4B
  if constexpr (M == 1) { 
    if constexpr (HEAD == 32) {
      if constexpr (SUB_KERNEL_ID == 0) {
        if (step >= 0 && step < 16) {
          flashattn_kernel_1_8192_16_32_8_128<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 16 && step < 32) {
          flashattn_kernel_1_8192_32_32_8_128<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 32 && step < 64) {
          flashattn_kernel_1_8192_64_32_8_128<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 64 && step < 128) {
          flashattn_kernel_1_8192_128_32_8_128__0<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 128 && step < 256) {
          flashattn_kernel_1_8192_256_32_8_128__0<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 256 && step < 512) {
          flashattn_kernel_1_8192_512_32_8_128__0<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 512 && step < 1024) {
          flashattn_kernel_1_8192_1024_32_8_128__0<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
        else if (step >= 1024 && step < 2048) {
          flashattn_kernel_1_8192_2048_32_8_128__0<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
      }
      else if constexpr (SUB_KERNEL_ID == 1) {
        if (step >= 0 && step < 64) {}
        else {
          flashattn_kernel_1_8192_2048_32_8_128__1<T, THREAD_NUM, SUB_KERNEL_ID, M, HEAD, GROUPS, DIM>(bx, by, bz, q, k, v, edge_ptr, mask_ptr, output_ptr, glse_ptr, output_partial_ptr); return;
        }
      }
    }
  }
#endif // ENABLE_QWEN3_4B

  printf("Error: [gqa_decode_kernel_%d_%d_%d_%d][step:%d][id:%d] There is no suitable microkernel!\n", M, HEAD, GROUPS, DIM, step, SUB_KERNEL_ID);
}

} // namespace kernel
