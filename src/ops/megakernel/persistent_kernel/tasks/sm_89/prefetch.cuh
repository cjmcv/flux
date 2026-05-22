
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

namespace kernel {

template <typename T, int THREAD_NUM, int TYPE, int M, int N>
__device__ __forceinline__ void prefetch_kernel(const int bx, const int by, const int bz,
                                            const int layer_id, 
                                            const void* __restrict__ weight_ptr) {
  // (bx=2, by=1, bz=1)
  // if (threadIdx.x == 0) {
  //   printf("prefetch layer_id: (%d, %lld)\n", layer_id, (long long)weight_ptr);
  // }
  
  // extern __shared__ __align__(1024) uchar buf_dyn_shmem[];

  // const bfloat16_t* __restrict__ B = static_cast<const bfloat16_t*>(weight_ptr);
  // #pragma unroll
  // for (int i_1 = 0; i_1 < 4; ++i_1) {
  //   tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 4096), B+((((((int)bx) * 65536) + (i_1 * 16384)) + ((((int)threadIdx.x) >> 3) * 1024)) + ((((int)threadIdx.x) & 7) * 8)));
  // }
}

} // namespace kernel
