#include <tl_templates/cuda/instruction/mma.h>
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
    int K>
    __device__ __forceinline__ void linear_gemm_add_tl_1_1024_2048(const int bx, const int by, const int bz,
                                                const void* __restrict__ input_ptr, const void* __restrict__ weight_ptr, const void* __restrict__ residual_ptr, void* __restrict__ output_ptr ) {
  // static_assert(THREAD_NUM==128);
  static_assert(TILE_DIM_X==64); static_assert(TILE_DIM_Y==16); static_assert(TILE_DIM_Z==64);
  static_assert(M==1); static_assert(N==1024); static_assert(K==2048);
  if (bx >= 16 || by >= 1 || bz >= 1) { return; }

  const bfloat16_t* __restrict__ A = static_cast<const bfloat16_t*>(input_ptr);
  const bfloat16_t* __restrict__ B = static_cast<const bfloat16_t*>(weight_ptr);
  const bfloat16_t* __restrict__ R = static_cast<const bfloat16_t*>(residual_ptr);
  bfloat16_t* __restrict__ C = static_cast<bfloat16_t*>(output_ptr);
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[8];
  bfloat16_t A_local[8];
  bfloat16_t B_local[8];
  float R_local[8];
  bfloat16_t R_local_cast[2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int k = 0; k < 32; ++k) {
    __syncthreads();
    bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if ((((int)threadIdx.x) < 8)) {
      condval = *(uint4*)(A + ((k * 64) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 3) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)) = condval;
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(B + (((((((int)bx) * 131072) + (i_1 * 32768)) + ((((int)threadIdx.x) >> 3) * 2048)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)));
    }
    __syncthreads();
    for (int ki = 0; ki < 4; ++ki) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4096)])) + 0, A_local + 0);
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 4) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])) + 0, B_local + 0);
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
    }
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    bfloat16_t broadcast_var_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint1 condval_1;
    if (((((i_2 & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < 1)) {
      condval_1 = *(uint1*)(R + (((((((i_2 & 1) * 8192) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)bx) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_2 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    } else {
      condval_1 = make_uint1(__pack_nv_bfloat162(broadcast_var_2, broadcast_var_2));
    }
    *(uint1*)(R_local_cast + 0) = condval_1;
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    float2 condval_2;
    if (((((i_2 & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < 1)) {
      float2 __1;
      uint1 v_ = *(uint1*)(R_local_cast + 0);
      ((float2*)(&__1))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
      condval_2 = __1;
    } else {
      condval_2 = make_float2(broadcast_var_3, broadcast_var_3);
    }
    *(float2*)(R_local + (i_2 * 2)) = condval_2;
  }
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    float2 c_val = *(float2*)(C_local + (i_3 * 2));
    float2 r_val = *(float2*)(R_local + (i_3 * 2));
    uint1 __2;
    float2 __3;
      __3.x = (c_val.x+r_val.x);
      __3.y = (c_val.y+r_val.y);
    (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&__3))[0]);
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_3 & 1) * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((int)threadIdx.x) >> 5) * 16) + ((i_3 >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_3 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __2;
  }
  __syncthreads();
  if (((int)threadIdx.x) < 8) {
    *(uint4*)(C + ((((int)bx) * 64) + (((int)threadIdx.x) * 8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((int)threadIdx.x) * 8));
  }
}


} // kernel
// Strategy: linear_gemm_add_tl_1_1024_2048
// selected_hparams: [16, 64, 64, 1, 0, 128, <GemmWarpPolicy.FullRow: 1>, False].
// smem: 10240 bytes.
// use_cooperative_groups: 0.
// layout: (16, 1, 1), (64, 16, 64)
// block_dim=(128, 1, 1).


extern "C" int create_linear_gemm_add_tl_1_1024_2048(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ R, bfloat16_t* __restrict__ C) {

	return 0;
}
#define LAUNCH_INFO_linear_gemm_add_tl_1_1024_2048 dim3(16, 1, 1), dim3(128, 1, 1), 10240, stream

// latency: 0.02565 ms vs [ref-0.02823 sim-1.0], idx: 19