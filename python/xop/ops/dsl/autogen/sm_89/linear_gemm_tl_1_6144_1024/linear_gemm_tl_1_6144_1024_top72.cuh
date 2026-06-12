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
  static_assert(TILE_DIM_X==256); static_assert(TILE_DIM_Y==16); static_assert(TILE_DIM_Z==128);
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
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((i_1 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)threadIdx.x) & 15) * 8))])), (((i_1 * 8) + (((int)threadIdx.x) >> 4)) < 1));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 32; ++i_2) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 16384) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(B[((((((int)bx) * 262144) + (i_2 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 7; ++k) {
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki = 0; ki < 8; ++ki) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((ki >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_3 = 0; i_3 < 4; ++i_3) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((ki >> 2) * 16384) + ((((int)threadIdx.x) >> 5) * 4096)) + (i_3 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])) + 0, B_local + (i_3 * 8));
      }
      for (int j = 0; j < 4; ++j) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((i_4 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (k * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 128)])), (((i_4 * 8) + (((int)threadIdx.x) >> 4)) < 1));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 32; ++i_5) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 16384) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(B[((((((((int)bx) * 262144) + (i_5 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (k * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 128)])));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
    tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((ki_1 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((ki_1 >> 2) * 16384) + ((((int)threadIdx.x) >> 5) * 4096)) + (i_6 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])) + 0, B_local + (i_6 * 8));
    }
    for (int j_1 = 0; j_1 < 4; ++j_1) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j_1 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 8) + 4)));
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 16; ++i_7) {
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_7 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
    *(uint1*)(C_shared_local_cast + 0) = __1;
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_7 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_7 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_7 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_7 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 34816)) = *(uint1*)(C_shared_local_cast + 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_8 = 0; i_8 < 4; ++i_8) {
    if (((i_8 * 4) + (((int)threadIdx.x) >> 5)) < 1) {
      *(uint4*)(C + ((((i_8 * 24576) + ((((int)threadIdx.x) >> 5) * 6144)) + (((int)bx) * 256)) + ((((int)threadIdx.x) & 31) * 8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 31) >> 3) * 1024) + (i_8 * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_8 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 34816));
    }
  }
}


} // kernel
// Strategy: linear_gemm_tl_1_6144_1024
// selected_hparams: [16, 256, 128, 1, 1, 128, <GemmWarpPolicy.FullRow: 1>, True].
// smem: 77824 bytes.
// use_cooperative_groups: 0.
// layout: (24, 1, 1), (256, 16, 128)
// block_dim=(128, 1, 1).


extern "C" int create_linear_gemm_tl_1_6144_1024(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {

	return 0;
}
#define LAUNCH_INFO_linear_gemm_tl_1_6144_1024 dim3(24, 1, 1), dim3(128, 1, 1), 77824, stream

// latency: 0.07645 ms vs [ref-0.07225 sim-1.0], idx: 134