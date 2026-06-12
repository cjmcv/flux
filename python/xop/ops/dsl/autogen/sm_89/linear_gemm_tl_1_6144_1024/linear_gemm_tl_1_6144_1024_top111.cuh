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
    __device__ __forceinline__ void linear_gemm_tl_1_6144_1024(const int bx, const int by, const int bz,
                                                const void* __restrict__ input_ptr, const void* __restrict__ weight_ptr, const void* __restrict__ residual_ptr, void* __restrict__ output_ptr ) {
  // static_assert(THREAD_NUM==128);
  static_assert(TILE_DIM_X==256); static_assert(TILE_DIM_Y==16); static_assert(TILE_DIM_Z==32);
  static_assert(M==1); static_assert(N==6144); static_assert(K==1024);
  if (bx >= 24 || by >= 1 || bz >= 1) { return; }

  const bfloat16_t* __restrict__ A = static_cast<const bfloat16_t*>(input_ptr);
  const bfloat16_t* __restrict__ B = static_cast<const bfloat16_t*>(weight_ptr);
  const bfloat16_t* __restrict__ R = static_cast<const bfloat16_t*>(residual_ptr);
  bfloat16_t* __restrict__ C = static_cast<bfloat16_t*>(output_ptr);
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[32];
  bfloat16_t A_local[8];
  bfloat16_t B_local[32];
  bfloat16_t C_shared_local_cast[2];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  tl::cp_async_gs_conditional<8>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))])), (&(A[(((int)threadIdx.x) * 4)])), (((int)threadIdx.x) < 8));
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 512)])), (&(B[((((((int)bx) * 262144) + (i_1 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8))])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 31; ++k) {
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki = 0; ki < 2; ++ki) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])) + 0, A_local + 0);
      for (int i_2 = 0; i_2 < 4; ++i_2) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) >> 5) * 2048) + (i_2 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 512)])) + 0, B_local + (i_2 * 8));
      }
      for (int j = 0; j < 4; ++j) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
      }
    }
    __syncthreads();
    tl::cp_async_gs_conditional<8>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))])), (&(A[(((k * 32) + (((int)threadIdx.x) * 4)) + 32)])), (((int)threadIdx.x) < 8));
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((i_3 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 512)])), (&(B[((((((((int)bx) * 262144) + (i_3 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])) + 0, A_local + 0);
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) >> 5) * 2048) + (i_4 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 512)])) + 0, B_local + (i_4 * 8));
    }
    for (int j_1 = 0; j_1 < 4; ++j_1) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j_1 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 8) + 4)));
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 16; ++i_5) {
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_5 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
    *(uint1*)(C_shared_local_cast + 0) = __1;
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_5 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_5 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_5 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_5 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8704)) = *(uint1*)(C_shared_local_cast + 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_6 = 0; i_6 < 4; ++i_6) {
    if (((i_6 * 4) + (((int)threadIdx.x) >> 5)) < 1) {
      *(uint4*)(C + ((((i_6 * 24576) + ((((int)threadIdx.x) >> 5) * 6144)) + (((int)bx) * 256)) + ((((int)threadIdx.x) & 31) * 8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 31) >> 3) * 1024) + (i_6 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_6 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8704));
    }
  }
}


} // kernel
// Strategy: linear_gemm_tl_1_6144_1024
// selected_hparams: [16, 256, 32, 1, 1, 128, <GemmWarpPolicy.Square: 0>, False].
// smem: 25600 bytes.
// use_cooperative_groups: 0.
// layout: (24, 1, 1), (256, 16, 32)
// block_dim=(128, 1, 1).


extern "C" int create_linear_gemm_tl_1_6144_1024(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {

	return 0;
}
#define LAUNCH_INFO_linear_gemm_tl_1_6144_1024 dim3(24, 1, 1), dim3(128, 1, 1), 25600, stream

// latency: 0.12541 ms vs [ref-0.07222 sim-1.0], idx: 101