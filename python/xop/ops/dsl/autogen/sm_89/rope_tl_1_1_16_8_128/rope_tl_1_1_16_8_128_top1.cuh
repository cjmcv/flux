#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

namespace kernel {

template <typename T,
          int THREAD_NUM,
          int BATCH,
          int SEQLEN,
          int NUM_HEADS_Q,
          int NUM_HEADS_K,
          int HEAD_DIM>
__device__ __forceinline__ void rope_kernel_1_1_16_8_128(const int bx, const int by, const int bz,
                                                   const void* __restrict__ q, 
                                                   const void* __restrict__ k, 
                                                   const void* __restrict__ cos_ptr,
                                                   const void* __restrict__ sin_ptr, 
                                                   void* __restrict__ q_embed_ptr,
                                                   void* __restrict__ k_embed_ptr) {
  static_assert(THREAD_NUM==128);
  static_assert(BATCH==1); static_assert(SEQLEN==1); 
  static_assert(NUM_HEADS_Q==16); static_assert(NUM_HEADS_K==8); static_assert(HEAD_DIM==128);
  if (bx >= 1 || by >= 16 || bz >= 1) { return; }
  
  const bfloat16_t* __restrict__ Q = static_cast<const bfloat16_t*>(q);
  const bfloat16_t* __restrict__ K = static_cast<const bfloat16_t*>(k);
  const bfloat16_t* __restrict__ cos_1 = static_cast<const bfloat16_t*>(cos_ptr);
  const bfloat16_t* __restrict__ sin_1 = static_cast<const bfloat16_t*>(sin_ptr);
  bfloat16_t* __restrict__ Q_embed = static_cast<bfloat16_t*>(q_embed_ptr);
  bfloat16_t* __restrict__ K_embed = static_cast<bfloat16_t*>(k_embed_ptr);
  
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  if (((int)threadIdx.x) < 64) {
    ((bfloat16_t*)buf_dyn_shmem)[((int)threadIdx.x)] = cos_1[((int)threadIdx.x)];
    ((bfloat16_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 64)] = sin_1[((int)threadIdx.x)];
  }
  ((bfloat16_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)] = Q[((((int)by) * 128) + ((int)threadIdx.x))];
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    float a_q = ((float)((bfloat16_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)]);
    float b_q = ((float)((bfloat16_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 192)]);
    float cos_val = ((float)((bfloat16_t*)buf_dyn_shmem)[((int)threadIdx.x)]);
    float sin_val = ((float)((bfloat16_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 64)]);
    float out_first_q = ((a_q * cos_val) - (b_q * sin_val));
    float out_second_q = ((b_q * cos_val) + (a_q * sin_val));
    Q_embed[((((int)by) * 128) + ((int)threadIdx.x))] = ((bfloat16_t)out_first_q);
    Q_embed[(((((int)by) * 128) + ((int)threadIdx.x)) + 64)] = ((bfloat16_t)out_second_q);
  }
  __syncthreads();
  if (((int)by) < 8) {
    ((bfloat16_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)] = K[((((int)by) * 128) + ((int)threadIdx.x))];
  }
  __syncthreads();
  if (((int)by) < 8) {
    if (((int)threadIdx.x) < 64) {
      float a_k = ((float)((bfloat16_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)]);
      float b_k = ((float)((bfloat16_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 192)]);
      float cos_val_1 = ((float)((bfloat16_t*)buf_dyn_shmem)[((int)threadIdx.x)]);
      float sin_val_1 = ((float)((bfloat16_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 64)]);
      float out_first_k = ((a_k * cos_val_1) - (b_k * sin_val_1));
      float out_second_k = ((b_k * cos_val_1) + (a_k * sin_val_1));
      K_embed[((((int)by) * 128) + ((int)threadIdx.x))] = ((bfloat16_t)out_first_k);
      K_embed[(((((int)by) * 128) + ((int)threadIdx.x)) + 64)] = ((bfloat16_t)out_second_k);
    }
  }
}


} // kernel
// Strategy: rope_tl_1_1_16_8_128
// selected_hparams: [1, 1, 1, 1, 128].
// smem: 512 bytes.
// use_cooperative_groups: 0.
// layout: (1, 16, 1), (1, 1, 1)
// block_dim=(128, 1, 1).


extern "C" int create_rope_tl_1_1_16_8_128(bfloat16_t* __restrict__ Q, bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ cos, bfloat16_t* __restrict__ sin, bfloat16_t* __restrict__ Q_embed, bfloat16_t* __restrict__ K_embed) {

	return 0;
}
#define LAUNCH_INFO_rope_tl_1_1_16_8_128 dim3(1, 16, 1), dim3(128, 1, 1), 512, stream

// latency: 0.0128 ms vs [ref-0.03579 sim-1.0], idx: 1