
#pragma once

#ifdef ENABLE_QWEN3_06B
#include "m1/rope_tl_1_1_16_8_128.cuh"
#endif // ENABLE_QWEN3_06B

#ifdef ENABLE_QWEN3_4B
#include "m1/rope_tl_1_1_32_8_128.cuh"
#endif // ENABLE_QWEN3_4B

namespace kernel {

template <typename T,
  int THREAD_NUM,
  int BATCH,
  int SEQLEN,
  int NUM_HEADS_Q,
  int NUM_HEADS_K,
  int HEAD_DIM>
__device__ __forceinline__ void rope_kernel(const int bx, const int by, const int bz,
                                            const void* __restrict__ q, 
                                            const void* __restrict__ k, 
                                            const void* __restrict__ cos,
                                            const void* __restrict__ sin, 
                                            void* __restrict__ q_embed_ptr,
                                            void* __restrict__ k_embed_ptr) {
#ifdef ENABLE_QWEN3_06B
  if constexpr (BATCH == 1 && SEQLEN == 1) { 
    if constexpr (NUM_HEADS_Q == 16 && NUM_HEADS_K == 8 && HEAD_DIM == 128) {
      rope_kernel_1_1_16_8_128<T, THREAD_NUM, BATCH, SEQLEN, NUM_HEADS_Q, NUM_HEADS_K, HEAD_DIM>(bx, by, bz, q, k, cos, sin, q_embed_ptr, k_embed_ptr); return;
    }
  }
#endif // ENABLE_QWEN3_06B

#ifdef ENABLE_QWEN3_4B
if constexpr (BATCH == 1 && SEQLEN == 1) { 
  if constexpr (NUM_HEADS_Q == 32 && NUM_HEADS_K == 8 && HEAD_DIM == 128) {
    rope_kernel_1_1_32_8_128<T, THREAD_NUM, BATCH, SEQLEN, NUM_HEADS_Q, NUM_HEADS_K, HEAD_DIM>(bx, by, bz, q, k, cos, sin, q_embed_ptr, k_embed_ptr); return;
  }
}
#endif // ENABLE_QWEN3_4B

  printf("Error1: [rope_kernel_%d_%d_%d_%d_%d] There is no suitable microkernel!\n", BATCH, SEQLEN, NUM_HEADS_Q, NUM_HEADS_K, HEAD_DIM);
}

} // namespace kernel
