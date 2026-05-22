
#pragma once

#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

#include "runtime_header.h"
using namespace megakernel::runtime;

namespace kernel {

template <typename T, int THREAD_NUM, int TYPE, int M, int N>
__device__ __forceinline__ void debug_kernel(void *ptr) {
  printf("%lld,", ptr);  
}

template <typename T, int THREAD_NUM>
__device__ __forceinline__ void prefetch_kernel_mlp_rms_norm(const int bx, const int by, const int bz,
                                                uint64_t* mbarrier_mem, const CUtensorMap *A_desc, const CUtensorMap *B_desc, 
                                                const void* __restrict__ residual_ptr, const CUtensorMap *C_desc, 
                                                int num_active_tokens,
                                                bool residual) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);

  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(*A_desc);
    tl::prefetch_tma_descriptor(*B_desc);
    tl::prefetch_tma_descriptor(*C_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();

  if (bx < 96){
    if (128 <= ((int)threadIdx.x)) {
      tl::warpgroup_reg_dealloc<24>();
      for (int k = 0; k < 1; ++k) {
        if (tl::tl_shuffle_elect<128>()) {
          mbarrier[(k % 3)].expect_transaction(16384);
          tl::fence_proxy_async();
          tl::tma_load(*B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[((k % 3) * 8192)])), (k * 128), (((int)bx) * 64));
          tl::tma_load(*B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k % 3) * 8192) + 4096)])), ((k * 128) + 64), (((int)bx) * 64));
        }
      }
    }
  }
  // else {
  //   if (128 <= ((int)threadIdx.x)) {
  //     tl::warpgroup_reg_dealloc<24>();
  //     for (int k = 1; k < 3; ++k) {
  //       if (tl::tl_shuffle_elect<128>()) {
  //         mbarrier[(k % 3)].expect_transaction(16384);
  //         tl::fence_proxy_async();
  //         tl::tma_load<tl::CacheHintSm90::EVICT_FIRST>(*B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[((k % 3) * 8192)])), (k * 128), (((int)bx-73) * 64));
  //         tl::tma_load<tl::CacheHintSm90::EVICT_FIRST>(*B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k % 3) * 8192) + 4096)])), ((k * 128) + 64), (((int)bx-73) * 64));
  //       }
  //     }
  //   }
  //   else {
  //     if (bx-96 >= 0) {
  //       tl::warpgroup_reg_dealloc<24>();
  //       for (int k = 1; k < 3; ++k) {
  //         if (tl::tl_shuffle_elect<128>()) {
  //           mbarrier[(k % 3)].expect_transaction(16384);
  //           tl::fence_proxy_async();
  //           tl::tma_load<tl::CacheHintSm90::EVICT_FIRST>(*B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[((k % 3) * 8192)])), (k * 128), (((int)bx-96) * 64));
  //           tl::tma_load<tl::CacheHintSm90::EVICT_FIRST>(*B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k % 3) * 8192) + 4096)])), ((k * 128) + 64), (((int)bx-96) * 64));
  //         }
  //       }        
  //     }
  //   }
  // }

}

template <typename T, int THREAD_NUM>
__device__ __forceinline__ void prefetch_kernel_mlp_silu_mul(const int bx, const int by, const int bz,
                                                uint64_t* mbarrier_mem, const CUtensorMap *A_desc, const CUtensorMap *B_desc, 
                                                const void* __restrict__ residual_ptr, const CUtensorMap *C_desc, 
                                                int num_active_tokens,
                                                bool residual) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);

  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(*A_desc);
    tl::prefetch_tma_descriptor(*B_desc);
    tl::prefetch_tma_descriptor(*C_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();

  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 3; ++k) {
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k % 3)].expect_transaction(16384);
        tl::fence_proxy_async();
        tl::tma_load(*B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[((k % 3) * 8192)])), (k * 128), (((int)bx) * 64));
        tl::tma_load(*B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k % 3) * 8192) + 4096)])), ((k * 128) + 64), (((int)bx) * 64));
      }
    }
  }
}

template <typename T, int THREAD_NUM, int TYPE, int M, int N>
__device__ __forceinline__ void prefetch_kernel(const int bx, const int by, const int bz,
                                                uint64_t* mbarrier_mem, const CUtensorMap *A_desc, const CUtensorMap *B_desc, 
                                                const void* __restrict__ residual_ptr, const CUtensorMap *C_desc, 
                                                int num_active_tokens,
                                                bool residual) {
          
  // if (threadIdx.x == 0)
  //   printf("pre (%d, %d, %d <%d>)\n", bx,by,bz, blockIdx.x);

  if constexpr (TYPE == TASK_RMS_NORM) {
    if constexpr (M == 1 && N == 1024) {
      prefetch_kernel_mlp_rms_norm<T, THREAD_NUM>(bx, by, bz, mbarrier_mem, A_desc, B_desc, residual_ptr, C_desc, num_active_tokens, residual); return;
    }
  }
  else if constexpr (TYPE == TASK_SILU_MUL) {
    if constexpr (M == 1 && N == 3072) {
      prefetch_kernel_mlp_silu_mul<T, THREAD_NUM>(bx, by, bz, mbarrier_mem, A_desc, B_desc, residual_ptr, C_desc, num_active_tokens, residual); return;
    }
  }
  printf("Error: [prefetch_kernel_%d_%d_%d] There is no suitable microkernel!\n", TYPE,M,N);
}

} // namespace kernel
