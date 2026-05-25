#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void rms_norm_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) rms_norm_kernel(const bfloat16_t* __restrict__ A, const bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float A_local[20];
  bfloat16_t A_shared_local_cast[4];
  float B_local[20];
  bfloat16_t B_shared_local_cast_1[4];
  float A_pow_local[20];
  float A_powsum[1];
  bfloat16_t C_local_cast_2[4];
  #pragma unroll
  for (int i = 0; i < 5; ++i) {
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((i * 512) + (((int)threadIdx.x) * 4))) = *(uint2*)(A + (((((int)blockIdx.x) * 2560) + (i * 512)) + (((int)threadIdx.x) * 4)));
  }
  if (((int)blockIdx.x) < 16) {
    #pragma unroll
    for (int i_1 = 0; i_1 < 5; ++i_1) {
      *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i_1 * 512) + (((int)threadIdx.x) * 4)) + 2560)) = *(uint2*)(B + ((i_1 * 512) + (((int)threadIdx.x) * 4)));
    }
  } else {
    #pragma unroll
    for (int i_2 = 0; i_2 < 5; ++i_2) {
      *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i_2 * 512) + (((int)threadIdx.x) * 4)) + 2560)) = *(uint2*)(B + (((i_2 * 512) + (((int)threadIdx.x) * 4)) + 2560));
    }
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 5; ++i_3) {
    *(uint2*)(A_shared_local_cast + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((i_3 * 512) + (((int)threadIdx.x) * 4)));
    float4 __1;
    uint2 v_ = *(uint2*)(A_shared_local_cast + 0);
    ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
    ((float2*)(&__1))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
    *(float4*)(A_local + (i_3 * 4)) = __1;
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 5; ++i_4) {
    *(uint2*)(B_shared_local_cast_1 + 0) = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i_4 * 512) + (((int)threadIdx.x) * 4)) + 2560));
    float4 __2;
    uint2 v__1 = *(uint2*)(B_shared_local_cast_1 + 0);
    ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[0]);
    ((float2*)(&__2))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__1))[1]);
    *(float4*)(B_local + (i_4 * 4)) = __2;
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 20; ++i_5) {
    A_pow_local[i_5] = (A_local[i_5] * A_local[i_5]);
  }
  A_powsum[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 20; ++rv) {
    A_powsum[0] = (A_powsum[0] + A_pow_local[(((rv % 5) * 4) + (rv / 5))]);
  }
  __syncthreads();
  A_powsum[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(A_powsum[0], (&(((float*)buf_dyn_shmem)[0])));
  A_powsum[0] = rsqrtf(((A_powsum[0] / 0x1.4p+11f/*2.560000e+03*/) + 0x1.19799812dea11p-40f/*1.000000e-12*/));
  #pragma unroll
  for (int i_6 = 0; i_6 < 20; ++i_6) {
    A_local[i_6] = (A_local[i_6] * (A_powsum[0] * B_local[i_6]));
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 5; ++i_7) {
    uint2 __3;
    float4 v__2 = *(float4*)(A_local + (i_7 * 4));
    (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&v__2))[1]);
    *(uint2*)(C_local_cast_2 + 0) = __3;
    *(uint2*)(C + (((((int)blockIdx.x) * 2560) + (i_7 * 512)) + (((int)threadIdx.x) * 4))) = *(uint2*)(C_local_cast_2 + 0);
  }
}


// Strategy: rms_norm_tl_24_2560
// selected_hparams: [1, 1, 128].
// smem: 10240 bytes.
// use_cooperative_groups: 0.
// layout: (24, 1, 1), (1, 1, 1)
// block_dim=(128, 1, 1).


extern "C" int create_rms_norm_tl_24_2560(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {

	return 0;
}
#define LAUNCH_INFO_rms_norm_tl_24_2560 dim3(24, 1, 1), dim3(128, 1, 1), 10240, stream

// latency: 0.00932 ms vs [ref-0.11914 sim-0.99999], idx: -1