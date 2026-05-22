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
    int K,
    int O_STRIDE = N,
    int PIPE_MAX = 3,
    bool FUSE_RES = false>
    __device__ __forceinline__ void linear_gemm_tl_32_19456_2560(const int bx, const int by, const int bz,
                                                const void* __restrict__ input_ptr,
                                                const void* __restrict__ weight_ptr,
                                                const void* __restrict__ residual_ptr,
                                                void* __restrict__ output_ptr,
                                                int num_active_tokens,
                                                bool residual) {
  static_assert(THREAD_NUM==128);
  static_assert(TILE_DIM_X==64); static_assert(TILE_DIM_Y==64); static_assert(TILE_DIM_Z==64);
  static_assert(M==32); static_assert(N==19456); static_assert(K==2560);
  
  const bfloat16_t* __restrict__ A = static_cast<const bfloat16_t*>(input_ptr);
  const bfloat16_t* __restrict__ B = static_cast<const bfloat16_t*>(weight_ptr);
  const bfloat16_t* __restrict__ R = static_cast<const bfloat16_t*>(residual_ptr);
  bfloat16_t* __restrict__ C = static_cast<bfloat16_t*>(output_ptr);
  
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[32];
  bfloat16_t A_local[16];
  bfloat16_t B_local[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(float2*)(C_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+(((i_1 * 40960) + ((((int)threadIdx.x) >> 3) * 2560)) + ((((int)threadIdx.x) & 7) * 8)), ((i_1 < 2) && (i_1 < 2)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), B+((((((int)bx) * 163840) + (i_2 * 40960)) + ((((int)threadIdx.x) >> 3) * 2560)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((i_3 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192), A+((((i_3 * 40960) + ((((int)threadIdx.x) >> 3) * 2560)) + ((((int)threadIdx.x) & 7) * 8)) + 64), ((i_3 < 2) && (i_3 < 2)));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_4 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), B+(((((((int)bx) * 163840) + (i_4 * 40960)) + ((((int)threadIdx.x) >> 3) * 2560)) + ((((int)threadIdx.x) & 7) * 8)) + 64));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 38; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((((k + 2) % 3) * 8192) + (i_5 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+(((((i_5 * 40960) + ((((int)threadIdx.x) >> 3) * 2560)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128), ((i_5 < 2) && (i_5 < 2)));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((k + 2) % 3) * 8192) + (i_6 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), B+((((((((int)bx) * 163840) + (i_6 * 40960)) + ((((int)threadIdx.x) >> 3) * 2560)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    for (int ki = 0; ki < 4; ++ki) {
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((k % 3) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_7 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + (i_7 * 8));
      }
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((k % 3) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (i_8 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)])) + 0, B_local + (i_8 * 8));
      }
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_9 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_9 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_10 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])) + 0, A_local + (i_10 * 8));
    }
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) >> 6) * 2048) + (i_11 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 20480)])) + 0, B_local + (i_11 * 8));
    }
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      for (int j_1 = 0; j_1 < 2; ++j_1) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_12 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_12 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 8) + 4)));
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_13 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + (i_13 * 8));
    }
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) >> 6) * 2048) + (i_14 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)])) + 0, B_local + (i_14 * 8));
    }
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      for (int j_2 = 0; j_2 < 2; ++j_2) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_15 * 16) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local + (j_2 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_15 * 16) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j_2 * 8) + 4)));
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_16 = 0; i_16 < 16; ++i_16) {
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_16 * 2));
    *reinterpret_cast<__nv_bfloat162*>(&(__1)) = __float22bfloat162_rn(*(float2*)(&(v_)));
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + ((i_16 >> 3) * 1024)) + ((i_16 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_16 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_16 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_17 = 0; i_17 < 4; ++i_17) {
    if (i_17 < 2) {
      *(uint4*)(C + ((((i_17 * 311296) + ((((int)threadIdx.x) >> 3) * 19456)) + (((int)bx) * 64)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((i_17 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    }
  }
}


} // kernel
// Strategy: linear_gemm_tl_32_19456_2560
// selected_hparams: [64, 64, 64, 1, 3, 128, 0, False].
// smem: 49152 bytes.
// use_cooperative_groups: 0.
// grid_dim=(304, 1, 1), tile_dim=(64, 64, 64),
// block_dim=(128, 1, 1).