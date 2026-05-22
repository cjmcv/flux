#pragma once

#ifdef ENABLE_QWEN3_06B
#include "m1/linear_gemm_tl_1_6144_1024.cuh"
#include "m1/linear_gemm_add_tl_1_1024_3072.cuh"
// attn
#include "m1/linear_gemm_tl_1_4096_1024.cuh"
#include "m1/linear_gemm_add_tl_1_1024_2048.cuh"
#endif

#ifdef ENABLE_QWEN3_4B
#include "m1/linear_gemm_tl_1_19456_2560.cuh"
#include "m1/linear_gemm_add_tl_1_2560_9728.cuh"
// attn
#include "m1/linear_gemm_tl_1_6144_2560.cuh"
#include "m1/linear_gemm_add_tl_1_2560_4096.cuh"
#endif


namespace kernel {

template <typename T,
    int THREAD_NUM,
    int TILE_DIM_X, 
    int TILE_DIM_Y, 
    int TILE_DIM_Z,
    int M,
    int N,
    int K,
    int O_STRIDE = N,
    int PIPE_MAX = 3,
    bool FUSE_RES = false,
    bool FUSE_SILU_MUL = false>
    __device__ __forceinline__ void linear_kernel(const int bx, const int by, const int bz,
                                                const void* __restrict__ input_ptr,
                                                const void* __restrict__ weight_ptr,
                                                const void* __restrict__ residual_ptr,
                                                void* __restrict__ output_ptr,
                                                int num_active_tokens,
                                                bool residual) {
    // printf("linear_kernel sm89.\n");
#ifdef ENABLE_QWEN3_06B
  if constexpr (FUSE_RES == true) {
    if constexpr (M == 1) {
      if constexpr (N == 1024 && K == 3072) {
        linear_gemm_add_tl_1_1024_3072<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual); return;
      }
      else if constexpr (N == 1024 && K == 2048) {
        linear_gemm_add_tl_1_1024_2048<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual); return;
      }
    } 
  }
  else {
    if constexpr (M == 1) {
      if constexpr (N == 6144 && K == 1024) {
        linear_gemm_tl_1_6144_1024<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual); return;
      }
      else if constexpr (N == 4096 && K == 1024) {
        linear_gemm_tl_1_4096_1024<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual); return;
      }
    }
  }
#endif // ENABLE_QWEN3_06B

#ifdef ENABLE_QWEN3_4B
  if constexpr (FUSE_RES == true) {
    if constexpr (M == 1) {    
      if constexpr (N == 2560 && K == 9728) {
        linear_gemm_add_tl_1_2560_9728<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual); return;
      }
      else if constexpr (N == 2560 && K == 4096) {
        linear_gemm_add_tl_1_2560_4096<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual); return;
      }
    } 
  }
  else {
    if constexpr (M == 1) {
      if constexpr (N == 19456 && K == 2560) {
        linear_gemm_tl_1_19456_2560<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual); return;
      }
      else if constexpr (N == 6144 && K == 2560) {
        linear_gemm_tl_1_6144_2560<T, THREAD_NUM, TILE_DIM_X, TILE_DIM_Y, TILE_DIM_Z, M, N, K, O_STRIDE, PIPE_MAX, FUSE_RES>(
          bx, by, bz, input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, residual); return;
      }
    }
  }
#endif // ENABLE_QWEN3_4B

  printf("Error: [linear_kernel_%d_%d_%d] res[%d] There is no suitable microkernel!\n", M,N,K,FUSE_RES);
}

} // kernel
