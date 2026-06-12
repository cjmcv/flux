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
        int TILE_DIM_X, 
        int TILE_DIM_Y, 
        int TILE_DIM_Z,
        int M,
        int N>
__device__ __forceinline__ void rms_norm_kernel_16_8_128(const int bx, const int by, const int bz,
                                                            void const *input_ptr,
                                                            void const *weight_ptr,
                                                            void *output_ptr,
                                                            float eps) {
  // static_assert(THREAD_NUM==128);
  static_assert(TILE_DIM_X==1); static_assert(TILE_DIM_Y==1); static_assert(TILE_DIM_Z==1);
  static_assert(M==24); static_assert(N==128);
  if (bx >= 24 || by >= 1 || bz >= 1 || threadIdx.x >= 128) { return; }
  
  const bfloat16_t* __restrict__ A = static_cast<const bfloat16_t*>(input_ptr);
  const bfloat16_t* __restrict__ B = static_cast<const bfloat16_t*>(weight_ptr);
  bfloat16_t* __restrict__ C = static_cast<bfloat16_t*>(output_ptr);
  
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float A_local[1];
  float B_local[1];
  float A_pow_local[1];
  float A_powsum[1];
  ((bfloat16_t*)buf_dyn_shmem)[((int)threadIdx.x)] = A[((((int)bx) * 128) + ((int)threadIdx.x))];
  if (((int)bx) < 16) {
    ((bfloat16_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)] = B[((int)threadIdx.x)];
  } else {
    ((bfloat16_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)] = B[(((int)threadIdx.x) + 128)];
  }
  A_local[0] = ((float)((bfloat16_t*)buf_dyn_shmem)[((int)threadIdx.x)]);
  __syncthreads();
  B_local[0] = ((float)((bfloat16_t*)buf_dyn_shmem)[(((int)threadIdx.x) + 128)]);
  A_pow_local[0] = (A_local[0] * A_local[0]);
  A_powsum[0] = 0x0p+0f/*0.000000e+00*/;
  A_powsum[0] = (A_powsum[0] + A_pow_local[0]);
  __syncthreads();
  A_powsum[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(A_powsum[0], (&(((float*)buf_dyn_shmem)[0])));
  A_powsum[0] = rsqrtf(((A_powsum[0] / 0x1p+7f/*1.280000e+02*/) + 0x1.19799812dea11p-40f/*1.000000e-12*/));
  A_local[0] = (A_local[0] * (A_powsum[0] * B_local[0]));
  C[((((int)bx) * 128) + ((int)threadIdx.x))] = ((bfloat16_t)A_local[0]);
}


} // kernel
// Strategy: rms_norm_tl_24_128
// selected_hparams: [1, 1, 128].
// smem: 512 bytes.
// use_cooperative_groups: 0.
// layout: (24, 1, 1), (1, 1, 1)
// block_dim=(128, 1, 1).


extern "C" int create_rms_norm_tl_24_128(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {

	return 0;
}
#define LAUNCH_INFO_rms_norm_tl_24_128 dim3(24, 1, 1), dim3(128, 1, 1), 512, stream

// latency: 0.01385 ms vs [ref-0.14722 sim-0.99999], idx: -1