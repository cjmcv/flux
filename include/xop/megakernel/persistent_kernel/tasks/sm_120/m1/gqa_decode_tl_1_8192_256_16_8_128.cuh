#include <tl_templates/cuda/instruction/mma.h>
#include <math_constants.h>
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
          int SUB_KERNEL_ID,
          int M, 
          int HEAD,
          int GROUPS,
          int DIM>
__device__ __forceinline__ void flashattn_kernel_1_8192_256_16_8_128__0(const int bx, const int by, const int bz,
                                                   const void* __restrict__ q, 
                                                   const void* __restrict__ k, 
                                                   const void* __restrict__ v,
                                                   const void* __restrict__ edge_ptr, 
                                                   const void* __restrict__ mask_ptr, 
                                                   void* __restrict__ output_ptr,
                                                   void* __restrict__ glse_ptr,
                                                   void* __restrict__ output_partial_ptr) {
  // static_assert(THREAD_NUM==128);
  static_assert(M==1); static_assert(HEAD==16); static_assert(GROUPS==8); static_assert(DIM==128);
  if constexpr (SUB_KERNEL_ID == 0) { if (bx >= 1 || by >= 8 || bz >= 8 || threadIdx.x >= 128) { return; } }
  if constexpr (SUB_KERNEL_ID == 1) { if (bx >= 16 || by >= 1 || bz >= 1 || threadIdx.x >= 128) { return; } }
  const bfloat16_t* __restrict__ Q = static_cast<const bfloat16_t*>(q);
  const bfloat16_t* __restrict__ K = static_cast<const bfloat16_t*>(k);
  const bfloat16_t* __restrict__ V = static_cast<const bfloat16_t*>(v);
  const int* __restrict__ edge = static_cast<const int*>(edge_ptr);
  const uchar* __restrict__ mask = static_cast<const uchar*>(mask_ptr);
  bfloat16_t* __restrict__ Output = static_cast<bfloat16_t*>(output_ptr);
  bfloat16_t* __restrict__ glse = static_cast<bfloat16_t*>(glse_ptr);
  bfloat16_t* __restrict__ Output_partial = static_cast<bfloat16_t*>(output_partial_ptr);

  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  float acc_s[16];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  bfloat16_t acc_s_cast[16];
  bfloat16_t A_local[8];
  bfloat16_t B_local[16];
  bfloat16_t B_local_1[64];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint4 condval;
    if (((((((((int)threadIdx.x) >> 5) + ((int)by)) >> 2) + i) < 2) && (((((((int)threadIdx.x) >> 5) + ((int)by)) >> 2) + i) < 2))) {
      condval = *(uint4*)(Q + (((i * 1024) + (((int)by) * 256)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)), __pack_nv_bfloat162(bfloat16_t(0x0p+0f/*0.000000e+00*/), bfloat16_t(0x0p+0f/*0.000000e+00*/)));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    *(float2*)(acc_o + (i_1 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    logsum[i_2] = 0x0p+0f/*0.000000e+00*/;
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    scores_max[i_3] = -CUDART_INF_F;
  }
  int actual_kv_seqlen = (edge[0] + 1);
  int condval_1;
  if ((((int)bz) == 7)) {
    condval_1 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
  } else {
    condval_1 = (actual_kv_seqlen >> 3);
  }
  if (0 < condval_1) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+(((((((int64_t)i_4) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)bz)) * (((int64_t)actual_kv_seqlen) >> (int64_t)3)) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), (((((((i_4 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8192) && (0 <= (((i_4 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && (0 <= (((i_4 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && ((((i_4 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8192)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 40960), V+(((((((int64_t)i_5) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)bz)) * (((int64_t)actual_kv_seqlen) >> (int64_t)3)) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)), (((((((i_5 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8192) && (0 <= (((i_5 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && (0 <= (((i_5 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && ((((i_5 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8192)));
    }
    tl::cp_async_commit();
  }
  int condval_2;
  if ((((int)bz) == 7)) {
    condval_2 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
  } else {
    condval_2 = (actual_kv_seqlen >> 3);
  }
  if (32 < condval_2) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), K+((((((((int64_t)i_6) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)bz)) * (((int64_t)actual_kv_seqlen) >> (int64_t)3)) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)32768), (((((((i_6 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8160) && (-32 <= (((i_6 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && (-32 <= (((i_6 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && ((((i_6 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8160)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_7 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 49152), V+((((((((int64_t)i_7) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)bz)) * (((int64_t)actual_kv_seqlen) >> (int64_t)3)) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)32768), (((((((i_7 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8160) && (-32 <= (((i_7 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && (-32 <= (((i_7 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && ((((i_7 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8160)));
    }
    tl::cp_async_commit();
  }
  int condval_3;
  if ((((int)bz) == 7)) {
    condval_3 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
  } else {
    condval_3 = (actual_kv_seqlen >> 3);
  }
  if (64 < condval_3) {
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_8 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), K+((((((((int64_t)i_8) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)bz)) * (((int64_t)actual_kv_seqlen) >> (int64_t)3)) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)65536), (((((((i_8 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8128) && (-64 <= (((i_8 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && (-64 <= (((i_8 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && ((((i_8 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8128)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_9 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 57344), V+((((((((int64_t)i_9) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)bz)) * (((int64_t)actual_kv_seqlen) >> (int64_t)3)) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)65536), (((((((i_9 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8128) && (-64 <= (((i_9 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && (-64 <= (((i_9 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && ((((i_9 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8128)));
    }
    tl::cp_async_commit();
  }
  int condval_4;
  if ((((int)bz) == 7)) {
    condval_4 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
  } else {
    condval_4 = (actual_kv_seqlen >> 3);
  }
  for (int k = 0; k < (((condval_4 + 31) >> 5) - 3); ++k) {
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      *(float2*)(acc_s + (i_10 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    tl::cp_async_wait<2>();
    __syncthreads();
    for (int ki = 0; ki < 8; ++ki) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((k % 3) * 4096) + ((ki >> 2) * 2048)) + (i_11 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_11 * 8));
      }
      for (int j = 0; j < 2; ++j) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((k % 3) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_12 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+(((((((((int64_t)k) * (int64_t)32768) + (((int64_t)i_12) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)bz)) * (((int64_t)actual_kv_seqlen) >> (int64_t)3)) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)98304), ((((((((k * 32) + (i_12 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8096) && (-96 <= ((((k * 32) + (i_12 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && (-96 <= ((((k * 32) + (i_12 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && (((((k * 32) + (i_12 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8096)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_13 = 0; i_13 < 16; ++i_13) {
      int condval_6;
      if ((((int)bz) == 7)) {
        condval_6 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
      } else {
        condval_6 = (actual_kv_seqlen >> 3);
      }
      float condval_5;
      if ((((((k * 32) + ((i_13 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_13 & 1)) < condval_6)) {
        condval_5 = acc_s[i_13];
      } else {
        condval_5 = -CUDART_INF_F;
      }
      acc_s[i_13] = condval_5;
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      scores_max_prev[i_14] = scores_max[i_14];
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      scores_max[i_15] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        scores_max[i_16] = max(scores_max[i_16], acc_s[((((rv & 3) * 4) + (i_16 * 2)) + (rv >> 2))]);
      }
      scores_max[i_16] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_16]);
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 2; ++i_17) {
      scores_max[i_17] = max(scores_max[i_17], scores_max_prev[i_17]);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 2; ++i_18) {
      scores_scale[i_18] = exp2f(((scores_max_prev[i_18] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_18] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 16; ++i_19) {
      acc_s[i_19] = exp2f(((acc_s[i_19] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_19 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 2; ++i_20) {
      scores_sum[i_20] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        scores_sum[i_20] = (scores_sum[i_20] + acc_s[((((rv_1 & 3) * 4) + (i_20 * 2)) + (rv_1 >> 2))]);
      }
      scores_sum[i_20] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_20]);
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 2; ++i_21) {
      logsum[i_21] = ((logsum[i_21] * scores_scale[i_21]) + scores_sum[i_21]);
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 8; ++i_22) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s + (i_22 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__1)) = __float22bfloat162_rn(*(float2*)(&(v_)));
      *(uint1*)(acc_s_cast + (i_22 * 2)) = __1;
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 64; ++i_23) {
      acc_o[i_23] = (acc_o[i_23] * scores_scale[((i_23 & 3) >> 1)]);
    }
    tl::cp_async_wait<2>();
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      for (int i_24 = 0; i_24 < 8; ++i_24) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((((k % 3) * 4096) + ((i_24 >> 2) * 2048)) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_24 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_24 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20480)])) + 0, B_local_1 + (i_24 * 8));
      }
      for (int j_1 = 0; j_1 < 8; ++j_1) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_25 = 0; i_25 < 4; ++i_25) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((k % 3) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_25 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 40960), V+(((((((((int64_t)k) * (int64_t)32768) + (((int64_t)i_25) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)bz)) * (((int64_t)actual_kv_seqlen) >> (int64_t)3)) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)98304), ((((((((k * 32) + (i_25 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8096) && (-96 <= ((((k * 32) + (i_25 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && (-96 <= ((((k * 32) + (i_25 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))))) && (((((k * 32) + (i_25 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 3))) < 8096)));
    }
    tl::cp_async_commit();
  }
  int condval_7;
  if ((((int)bz) == 7)) {
    condval_7 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
  } else {
    condval_7 = (actual_kv_seqlen >> 3);
  }
  if (65 <= condval_7) {
    #pragma unroll
    for (int i_26 = 0; i_26 < 8; ++i_26) {
      *(float2*)(acc_s + (i_26 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    tl::cp_async_wait<2>();
    __syncthreads();
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_27 = 0; i_27 < 2; ++i_27) {
        int condval_8;
        if ((((int)bz) == 7)) {
          condval_8 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
        } else {
          condval_8 = (actual_kv_seqlen >> 3);
        }
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((((condval_8 + 31) % 96) >> 5) * 4096) + ((ki_2 >> 2) * 2048)) + (i_27 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_27 * 8));
      }
      for (int j_2 = 0; j_2 < 2; ++j_2) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j_2 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j_2 * 8) + 4)));
      }
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 16; ++i_28) {
      int condval_10;
      if ((((int)bz) == 7)) {
        condval_10 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
      } else {
        condval_10 = (actual_kv_seqlen >> 3);
      }
      int condval_11;
      if ((((int)bz) == 7)) {
        condval_11 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
      } else {
        condval_11 = (actual_kv_seqlen >> 3);
      }
      float condval_9;
      if ((((((((condval_10 + 31) >> 5) * 32) + ((i_28 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_28 & 1)) < (condval_11 + 96))) {
        condval_9 = acc_s[i_28];
      } else {
        condval_9 = -CUDART_INF_F;
      }
      acc_s[i_28] = condval_9;
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 2; ++i_29) {
      scores_max_prev[i_29] = scores_max[i_29];
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 2; ++i_30) {
      scores_max[i_30] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 2; ++i_31) {
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
        scores_max[i_31] = max(scores_max[i_31], acc_s[((((rv_2 & 3) * 4) + (i_31 * 2)) + (rv_2 >> 2))]);
      }
      scores_max[i_31] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_31]);
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 2; ++i_32) {
      scores_max[i_32] = max(scores_max[i_32], scores_max_prev[i_32]);
    }
    #pragma unroll
    for (int i_33 = 0; i_33 < 2; ++i_33) {
      scores_scale[i_33] = exp2f(((scores_max_prev[i_33] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_33] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_34 = 0; i_34 < 16; ++i_34) {
      acc_s[i_34] = exp2f(((acc_s[i_34] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_34 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_35 = 0; i_35 < 2; ++i_35) {
      scores_sum[i_35] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
        scores_sum[i_35] = (scores_sum[i_35] + acc_s[((((rv_3 & 3) * 4) + (i_35 * 2)) + (rv_3 >> 2))]);
      }
      scores_sum[i_35] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_35]);
    }
    #pragma unroll
    for (int i_36 = 0; i_36 < 2; ++i_36) {
      logsum[i_36] = ((logsum[i_36] * scores_scale[i_36]) + scores_sum[i_36]);
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 8; ++i_37) {
      uint1 __2;
      float2 v__1 = *(float2*)(acc_s + (i_37 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__2)) = __float22bfloat162_rn(*(float2*)(&(v__1)));
      *(uint1*)(acc_s_cast + (i_37 * 2)) = __2;
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 64; ++i_38) {
      acc_o[i_38] = (acc_o[i_38] * scores_scale[((i_38 & 3) >> 1)]);
    }
    tl::cp_async_wait<2>();
    __syncthreads();
    for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
      for (int i_39 = 0; i_39 < 8; ++i_39) {
        int condval_12;
        if ((((int)bz) == 7)) {
          condval_12 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
        } else {
          condval_12 = (actual_kv_seqlen >> 3);
        }
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((condval_12 + 31) % 96) >> 5) * 4096) + ((i_39 >> 2) * 2048)) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_39 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_39 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20480)])) + 0, B_local_1 + (i_39 * 8));
      }
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_3 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_3 * 8) + 4)));
      }
    }
  }
  int condval_13;
  if ((((int)bz) == 7)) {
    condval_13 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
  } else {
    condval_13 = (actual_kv_seqlen >> 3);
  }
  if (33 <= condval_13) {
    #pragma unroll
    for (int i_40 = 0; i_40 < 8; ++i_40) {
      *(float2*)(acc_s + (i_40 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_4 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_41 = 0; i_41 < 2; ++i_41) {
        int condval_14;
        if ((((int)bz) == 7)) {
          condval_14 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
        } else {
          condval_14 = (actual_kv_seqlen >> 3);
        }
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((((condval_14 + 31) >> 5) + 1) % 3) * 4096) + ((ki_4 >> 2) * 2048)) + (i_41 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_41 * 8));
      }
      for (int j_4 = 0; j_4 < 2; ++j_4) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j_4 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j_4 * 8) + 4)));
      }
    }
    #pragma unroll
    for (int i_42 = 0; i_42 < 16; ++i_42) {
      int condval_16;
      if ((((int)bz) == 7)) {
        condval_16 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
      } else {
        condval_16 = (actual_kv_seqlen >> 3);
      }
      int condval_17;
      if ((((int)bz) == 7)) {
        condval_17 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
      } else {
        condval_17 = (actual_kv_seqlen >> 3);
      }
      float condval_15;
      if ((((((((condval_16 + 31) >> 5) * 32) + ((i_42 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_42 & 1)) < (condval_17 + 64))) {
        condval_15 = acc_s[i_42];
      } else {
        condval_15 = -CUDART_INF_F;
      }
      acc_s[i_42] = condval_15;
    }
    #pragma unroll
    for (int i_43 = 0; i_43 < 2; ++i_43) {
      scores_max_prev[i_43] = scores_max[i_43];
    }
    #pragma unroll
    for (int i_44 = 0; i_44 < 2; ++i_44) {
      scores_max[i_44] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_45 = 0; i_45 < 2; ++i_45) {
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 8; ++rv_4) {
        scores_max[i_45] = max(scores_max[i_45], acc_s[((((rv_4 & 3) * 4) + (i_45 * 2)) + (rv_4 >> 2))]);
      }
      scores_max[i_45] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_45]);
    }
    #pragma unroll
    for (int i_46 = 0; i_46 < 2; ++i_46) {
      scores_max[i_46] = max(scores_max[i_46], scores_max_prev[i_46]);
    }
    #pragma unroll
    for (int i_47 = 0; i_47 < 2; ++i_47) {
      scores_scale[i_47] = exp2f(((scores_max_prev[i_47] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_47] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_48 = 0; i_48 < 16; ++i_48) {
      acc_s[i_48] = exp2f(((acc_s[i_48] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_48 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_49 = 0; i_49 < 2; ++i_49) {
      scores_sum[i_49] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_5 = 0; rv_5 < 8; ++rv_5) {
        scores_sum[i_49] = (scores_sum[i_49] + acc_s[((((rv_5 & 3) * 4) + (i_49 * 2)) + (rv_5 >> 2))]);
      }
      scores_sum[i_49] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_49]);
    }
    #pragma unroll
    for (int i_50 = 0; i_50 < 2; ++i_50) {
      logsum[i_50] = ((logsum[i_50] * scores_scale[i_50]) + scores_sum[i_50]);
    }
    #pragma unroll
    for (int i_51 = 0; i_51 < 8; ++i_51) {
      uint1 __3;
      float2 v__2 = *(float2*)(acc_s + (i_51 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__3)) = __float22bfloat162_rn(*(float2*)(&(v__2)));
      *(uint1*)(acc_s_cast + (i_51 * 2)) = __3;
    }
    #pragma unroll
    for (int i_52 = 0; i_52 < 64; ++i_52) {
      acc_o[i_52] = (acc_o[i_52] * scores_scale[((i_52 & 3) >> 1)]);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki_5 = 0; ki_5 < 2; ++ki_5) {
      for (int i_53 = 0; i_53 < 8; ++i_53) {
        int condval_18;
        if ((((int)bz) == 7)) {
          condval_18 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
        } else {
          condval_18 = (actual_kv_seqlen >> 3);
        }
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((condval_18 + 31) >> 5) + 1) % 3) * 4096) + ((i_53 >> 2) * 2048)) + (ki_5 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_53 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_53 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20480)])) + 0, B_local_1 + (i_53 * 8));
      }
      for (int j_5 = 0; j_5 < 8; ++j_5) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_5 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_5 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_5 * 8) + 4)));
      }
    }
  }
  int condval_19;
  if ((((int)bz) == 7)) {
    condval_19 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
  } else {
    condval_19 = (actual_kv_seqlen >> 3);
  }
  if (1 <= condval_19) {
    #pragma unroll
    for (int i_54 = 0; i_54 < 8; ++i_54) {
      *(float2*)(acc_s + (i_54 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki_6 = 0; ki_6 < 8; ++ki_6) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_6 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_55 = 0; i_55 < 2; ++i_55) {
        int condval_20;
        if ((((int)bz) == 7)) {
          condval_20 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
        } else {
          condval_20 = (actual_kv_seqlen >> 3);
        }
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((((condval_20 + 31) >> 5) + 2) % 3) * 4096) + ((ki_6 >> 2) * 2048)) + (i_55 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_55 * 8));
      }
      for (int j_6 = 0; j_6 < 2; ++j_6) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_6 * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j_6 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_6 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j_6 * 8) + 4)));
      }
    }
    #pragma unroll
    for (int i_56 = 0; i_56 < 16; ++i_56) {
      int condval_22;
      if ((((int)bz) == 7)) {
        condval_22 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
      } else {
        condval_22 = (actual_kv_seqlen >> 3);
      }
      int condval_23;
      if ((((int)bz) == 7)) {
        condval_23 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
      } else {
        condval_23 = (actual_kv_seqlen >> 3);
      }
      float condval_21;
      if ((((((((condval_22 + 31) >> 5) * 32) + ((i_56 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_56 & 1)) < (condval_23 + 32))) {
        condval_21 = acc_s[i_56];
      } else {
        condval_21 = -CUDART_INF_F;
      }
      acc_s[i_56] = condval_21;
    }
    #pragma unroll
    for (int i_57 = 0; i_57 < 2; ++i_57) {
      scores_max_prev[i_57] = scores_max[i_57];
    }
    #pragma unroll
    for (int i_58 = 0; i_58 < 2; ++i_58) {
      scores_max[i_58] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_59 = 0; i_59 < 2; ++i_59) {
      #pragma unroll
      for (int rv_6 = 0; rv_6 < 8; ++rv_6) {
        scores_max[i_59] = max(scores_max[i_59], acc_s[((((rv_6 & 3) * 4) + (i_59 * 2)) + (rv_6 >> 2))]);
      }
      scores_max[i_59] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_59]);
    }
    #pragma unroll
    for (int i_60 = 0; i_60 < 2; ++i_60) {
      scores_max[i_60] = max(scores_max[i_60], scores_max_prev[i_60]);
    }
    #pragma unroll
    for (int i_61 = 0; i_61 < 2; ++i_61) {
      scores_scale[i_61] = exp2f(((scores_max_prev[i_61] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_61] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_62 = 0; i_62 < 16; ++i_62) {
      acc_s[i_62] = exp2f(((acc_s[i_62] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_62 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_63 = 0; i_63 < 2; ++i_63) {
      scores_sum[i_63] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_7 = 0; rv_7 < 8; ++rv_7) {
        scores_sum[i_63] = (scores_sum[i_63] + acc_s[((((rv_7 & 3) * 4) + (i_63 * 2)) + (rv_7 >> 2))]);
      }
      scores_sum[i_63] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_63]);
    }
    #pragma unroll
    for (int i_64 = 0; i_64 < 2; ++i_64) {
      logsum[i_64] = ((logsum[i_64] * scores_scale[i_64]) + scores_sum[i_64]);
    }
    #pragma unroll
    for (int i_65 = 0; i_65 < 8; ++i_65) {
      uint1 __4;
      float2 v__3 = *(float2*)(acc_s + (i_65 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__4)) = __float22bfloat162_rn(*(float2*)(&(v__3)));
      *(uint1*)(acc_s_cast + (i_65 * 2)) = __4;
    }
    #pragma unroll
    for (int i_66 = 0; i_66 < 64; ++i_66) {
      acc_o[i_66] = (acc_o[i_66] * scores_scale[((i_66 & 3) >> 1)]);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki_7 = 0; ki_7 < 2; ++ki_7) {
      for (int i_67 = 0; i_67 < 8; ++i_67) {
        int condval_24;
        if ((((int)bz) == 7)) {
          condval_24 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 3)));
        } else {
          condval_24 = (actual_kv_seqlen >> 3);
        }
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((condval_24 + 31) >> 5) + 2) % 3) * 4096) + ((i_67 >> 2) * 2048)) + (ki_7 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_67 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_67 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20480)])) + 0, B_local_1 + (i_67 * 8));
      }
      for (int j_7 = 0; j_7 < 8; ++j_7) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_7 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_7 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_7 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_7 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_7 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_7 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_68 = 0; i_68 < 64; ++i_68) {
    acc_o[i_68] = (acc_o[i_68] / logsum[((i_68 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_69 = 0; i_69 < 2; ++i_69) {
    logsum[i_69] = (log2f(logsum[i_69]) + (scores_max[i_69] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
  }
  if ((((int)threadIdx.x) % 4) == 0) {
    #pragma unroll
    for (int i_70 = 0; i_70 < 2; ++i_70) {
      if (((((((int)threadIdx.x) >> 5) * 8) + (i_70 * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1) {
        glse[((((((((int)threadIdx.x) >> 5) * 128) + (i_70 * 64)) + (((int)by) * 16)) + (((((int)threadIdx.x) & 31) >> 2) * 8)) + ((int)bz))] = ((bfloat16_t)logsum[i_70]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_71 = 0; i_71 < 8; ++i_71) {
    tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_71 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((bfloat16_t)acc_o[(i_71 * 8)]), ((bfloat16_t)acc_o[((i_71 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_o[((i_71 * 8) + 2)]), ((bfloat16_t)acc_o[((i_71 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_o[((i_71 * 8) + 4)]), ((bfloat16_t)acc_o[((i_71 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_o[((i_71 * 8) + 6)]), ((bfloat16_t)acc_o[((i_71 * 8) + 7)])));
  }
  __syncthreads();
  *(uint1*)(Output_partial + ((((((int)by) * 2048) + ((((int)threadIdx.x) >> 6) * 1024)) + (((int)bz) * 128)) + ((((int)threadIdx.x) & 63) * 2))) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((int)threadIdx.x) * 2));
}


template <typename T,
          int THREAD_NUM,
          int SUB_KERNEL_ID,
          int M, 
          int HEAD,
          int GROUPS,
          int DIM>
__device__ __forceinline__ void flashattn_kernel_1_8192_256_16_8_128__1(const int bx, const int by, const int bz,
                                                   const void* __restrict__ q, 
                                                   const void* __restrict__ k, 
                                                   const void* __restrict__ v,
                                                   const void* __restrict__ edge_ptr, 
                                                   const void* __restrict__ mask_ptr, 
                                                   void* __restrict__ output_ptr,
                                                   void* __restrict__ glse_ptr,
                                                   void* __restrict__ output_partial_ptr) {
  // static_assert(THREAD_NUM==128);
  static_assert(M==1); static_assert(HEAD==16); static_assert(GROUPS==8); static_assert(DIM==128);
  if constexpr (SUB_KERNEL_ID == 0) { if (bx >= 1 || by >= 8 || bz >= 8 || threadIdx.x >= 128) { return; } }
  if constexpr (SUB_KERNEL_ID == 1) { if (bx >= 16 || by >= 1 || bz >= 1 || threadIdx.x >= 128) { return; } }
  const bfloat16_t* __restrict__ Q = static_cast<const bfloat16_t*>(q);
  const bfloat16_t* __restrict__ K = static_cast<const bfloat16_t*>(k);
  const bfloat16_t* __restrict__ V = static_cast<const bfloat16_t*>(v);
  const int* __restrict__ edge = static_cast<const int*>(edge_ptr);
  const uchar* __restrict__ mask = static_cast<const uchar*>(mask_ptr);
  bfloat16_t* __restrict__ Output = static_cast<bfloat16_t*>(output_ptr);
  bfloat16_t* __restrict__ glse = static_cast<bfloat16_t*>(glse_ptr);
  bfloat16_t* __restrict__ Output_partial = static_cast<bfloat16_t*>(output_partial_ptr);

  float lse_logsum_local[1];
  float o_accum_local[1];
  bfloat16_t lse_local[8];
  float lse_max_local[1];
  bfloat16_t po_local[1];
  float scale_local[1];
  lse_logsum_local[0] = 0x0p+0f/*0.000000e+00*/;
  o_accum_local[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    lse_local[i] = glse[((((int)bx) * 8) + i)];
  }
  lse_max_local[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 8; ++rv) {
    lse_max_local[0] = max(lse_max_local[0], ((float)lse_local[rv]));
  }
  for (int k = 0; k < 8; ++k) {
    lse_logsum_local[0] = (lse_logsum_local[0] + exp2f((((float)lse_local[k]) - lse_max_local[0])));
  }
  lse_logsum_local[0] = (log2f(lse_logsum_local[0]) + lse_max_local[0]);
  for (int k_1 = 0; k_1 < 8; ++k_1) {
    po_local[0] = Output_partial[(((((int)bx) * 1024) + (k_1 * 128)) + ((int)threadIdx.x))];
    scale_local[0] = exp2f((((float)lse_local[k_1]) - lse_logsum_local[0]));
    o_accum_local[0] = (o_accum_local[0] + (((float)po_local[0]) * scale_local[0]));
  }
  Output[((((int)bx) * 128) + ((int)threadIdx.x))] = ((bfloat16_t)o_accum_local[0]);
}


} // kernel
// Strategy: gqa_decode_tl_1_8192_256_16_8_128
// selected_hparams: [32, 64, 8, 3, 128].
// smem: 0 bytes.
// use_cooperative_groups: 0.
// layout: (1, 8, 8), (32, 64, 8), (16, 1, 1), (32, 64, 8)
// block_dim=(128, 1, 1).
// latency: 0 ms vs [ref-0 sim-0], idx: 14