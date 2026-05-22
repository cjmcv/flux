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
__device__ __forceinline__ void silu_mul_kernel_32_3072(const int bx, const int by, const int bz,
                                                   void const *input_ptr,
                                                   void *output_ptr,
                                                   int num_active_tokens) {
  static_assert(THREAD_NUM==128);
  static_assert(TILE_DIM_X==64); static_assert(TILE_DIM_Y==16); static_assert(TILE_DIM_Z==1);
  static_assert(M==32); static_assert(N==3072);
  
  const bfloat16_t* __restrict__ A = static_cast<const bfloat16_t*>(input_ptr);
  bfloat16_t* __restrict__ C = static_cast<bfloat16_t*>(output_ptr);
  
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((int)threadIdx.x) * 8)) = *(uint4*)(A + ((((((int)by) * 98304) + ((((int)threadIdx.x) >> 3) * 6144)) + (((int)bx) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((int)threadIdx.x) * 8) + 1024)) = *(uint4*)(A + (((((((int)by) * 98304) + ((((int)threadIdx.x) >> 3) * 6144)) + (((int)bx) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 3072));
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float4 __1;
    uint2 v_ = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + ((i * 512) + (((int)threadIdx.x) * 4)));
    ((float2*)(&__1))[0] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&(v_)));
    ((float2*)(&__1))[1] = __bfloat1622float2(*(reinterpret_cast<__nv_bfloat162*>(&(v_))+1));
    float4 xi = __1;
    float4 __2;
      float4 v__1 = make_float4(0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/, 0x1p+0f/*1.000000e+00*/);
      float4 __3;
        float4 __4;
        float4 __5;
          float4 v__2 = make_float4(-0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/, -0x1p+0f/*-1.000000e+00*/);
          __5.x = (xi.x*v__2.x);
          __5.y = (xi.y*v__2.y);
          __5.z = (xi.z*v__2.z);
          __5.w = (xi.w*v__2.w);
        __4.x = expf(__5.x);
        __4.y = expf(__5.y);
        __4.z = expf(__5.z);
        __4.w = expf(__5.w);
        __3.x = (v__1.x+__4.x);
        __3.y = (v__1.y+__4.y);
        __3.z = (v__1.z+__4.z);
        __3.w = (v__1.w+__4.w);
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
      uint2 v__3 = *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i * 512) + (((int)threadIdx.x) * 4)) + 1024));
      ((float2*)(&__9))[0] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&(v__3)));
      ((float2*)(&__9))[1] = __bfloat1622float2(*(reinterpret_cast<__nv_bfloat162*>(&(v__3))+1));
      __7.x = (__8.x*__9.x);
      __7.y = (__8.y*__9.y);
      __7.z = (__8.z*__9.z);
      __7.w = (__8.w*__9.w);
    (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(*(float2*)(&(__7)));
    (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(*((float2*)(&(__7))+1));
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((i * 512) + (((int)threadIdx.x) * 4)) + 2048)) = __6;
  }
  __syncthreads();
  *(uint4*)(C + ((((((int)by) * 49152) + ((((int)threadIdx.x) >> 3) * 3072)) + (((int)bx) * 64)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((int)threadIdx.x) * 8) + 2048));
}


} // kernel
// Strategy: silu_mul_tl_32_3072
// selected_hparams: [16, 64, 128].
// smem: 6144 bytes.
// use_cooperative_groups: 0.
// layout: (48, 2, 1), (64, 16, 1)
// block_dim=(128, 1, 1).
// latency: 0.00861