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
          int N,
          int I_STRIDE,
          int O_STRIDE>
__device__ __forceinline__ void silu_mul_kernel_1_3072(const int bx, const int by, const int bz,
                                                   void const *input_ptr,
                                                   void *output_ptr,
                                                   int num_active_tokens) {
  static_assert(THREAD_NUM==128);
  static_assert(TILE_DIM_X==64); static_assert(TILE_DIM_Y==16); static_assert(TILE_DIM_Z==1);
  static_assert(M==1); static_assert(N==3072);
  if (bx >= 48 || by >= 1 || bz >= 1) { return; }
  
  const bfloat16_t* __restrict__ A = static_cast<const bfloat16_t*>(input_ptr);
  bfloat16_t* __restrict__ C = static_cast<bfloat16_t*>(output_ptr);
  
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
  uint4 condval;
  if ((((int)threadIdx.x) < 8)) {
    condval = *(uint4*)(A + ((((int)bx) * 64) + (((int)threadIdx.x) * 8)));
  } else {
    condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
  }
  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((int)threadIdx.x) * 8)) = condval;
  bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
  uint4 condval_1;
  if ((((int)threadIdx.x) < 8)) {
    condval_1 = *(uint4*)(A + (((((int)bx) * 64) + (((int)threadIdx.x) * 8)) + 3072));
  } else {
    condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
  }
  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((int)threadIdx.x) * 8) + 1024)) = condval_1;
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float4 __1;
    uint2 v_ = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((i * 512) + (((int)threadIdx.x) * 4)));
    ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
    ((float2*)(&__1))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
    float4 xi = __1;
    float broadcast_var_2 = 0x1p+0f/*1.000000e+00*/;
    float broadcast_var_3 = 0x1p+0f/*1.000000e+00*/;
    float broadcast_var_4 = -0x1p+0f/*-1.000000e+00*/;
    float4 __2;
      float4 v__1 = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
      float4 __3;
        float4 v__2 = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
        float4 __4;
        float4 __5;
          float4 v__3 = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
          __5.x = (xi.x*v__3.x);
          __5.y = (xi.y*v__3.y);
          __5.z = (xi.z*v__3.z);
          __5.w = (xi.w*v__3.w);
        __4.x = expf(__5.x);
        __4.y = expf(__5.y);
        __4.z = expf(__5.z);
        __4.w = expf(__5.w);
        __3.x = (v__2.x+__4.x);
        __3.y = (v__2.y+__4.y);
        __3.z = (v__2.z+__4.z);
        __3.w = (v__2.w+__4.w);
      __2.x = (v__1.x/__3.x);
      __2.y = (v__1.y/__3.y);
      __2.z = (v__1.z/__3.z);
      __2.w = (v__1.w/__3.w);
    float4 sig = __2;
    uint2 __6;
    float4 __7;
      float4 __8;
        __8.x = (xi.x*sig.x);
        __8.y = (xi.y*sig.y);
        __8.z = (xi.z*sig.z);
        __8.w = (xi.w*sig.w);
      float4 __9;
      uint2 v__4 = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i * 512) + (((int)threadIdx.x) * 4)) + 1024));
      ((float2*)(&__9))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[0]);
      ((float2*)(&__9))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[1]);
      __7.x = (__8.x*__9.x);
      __7.y = (__8.y*__9.y);
      __7.z = (__8.z*__9.z);
      __7.w = (__8.w*__9.w);
    (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&__7))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(((float2*)(&__7))[1]);
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i * 512) + (((int)threadIdx.x) * 4)) + 2048)) = __6;
  }
  __syncthreads();
  if (((int)threadIdx.x) < 8) {
    *(uint4*)(C + ((((int)bx) * 64) + (((int)threadIdx.x) * 8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((int)threadIdx.x) * 8) + 2048));
  }
}


} // kernel
// Strategy: silu_mul_tl_1_3072
// selected_hparams: [16, 64, 128].
// smem: 6144 bytes.
// use_cooperative_groups: 0.
// layout: (48, 1, 1), (64, 16, 1)
// block_dim=(128, 1, 1).


extern "C" int create_silu_mul_tl_1_3072(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ C) {

	return 0;
}
#define LAUNCH_INFO_silu_mul_tl_1_3072 dim3(48, 1, 1), dim3(128, 1, 1), 6144, stream

// latency: 0.02239 ms vs [ref-0.04073 sim-1.0], idx: 0