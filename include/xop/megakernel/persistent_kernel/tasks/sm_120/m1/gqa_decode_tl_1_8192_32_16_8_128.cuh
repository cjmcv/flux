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
__device__ __forceinline__ void flashattn_kernel_1_8192_32_16_8_128(const int bx, const int by, const int bz,
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
  if constexpr (SUB_KERNEL_ID == 0) { if (bx >= 1 || by >= 8 || bz >= 1 || threadIdx.x >= 128) { return; } }
  if constexpr (SUB_KERNEL_ID == 1) { if (bx >= 0 || by >= 0 || bz >= 0 || threadIdx.x >= 128) { return; } }
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
  int valid_kv_seqlen = (edge[0] + 1);
  if (0 < valid_kv_seqlen) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+((((i_4 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), V+((((i_5 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    }
    tl::cp_async_commit();
  }
  if (32 < valid_kv_seqlen) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), K+(((((i_6 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_7 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 40960), V+(((((i_7 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768));
    }
    tl::cp_async_commit();
  }
  for (int k = 0; k < (((valid_kv_seqlen + 31) >> 5) - 2); ++k) {
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      *(float2*)(acc_s + (i_8 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki = 0; ki < 8; ++ki) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((k & 1) * 4096) + ((ki >> 2) * 2048)) + (i_9 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_9 * 8));
      }
      for (int j = 0; j < 2; ++j) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((k & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_10 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+((((((((int64_t)k) * (int64_t)32768) + (((int64_t)i_10) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)65536), (k < 254));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_11 = 0; i_11 < 16; ++i_11) {
      float condval_1;
      if ((((((k * 32) + ((i_11 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_11 & 1)) < valid_kv_seqlen)) {
        condval_1 = acc_s[i_11];
      } else {
        condval_1 = -CUDART_INF_F;
      }
      acc_s[i_11] = condval_1;
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      scores_max_prev[i_12] = scores_max[i_12];
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      scores_max[i_13] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        scores_max[i_14] = max(scores_max[i_14], acc_s[((((rv & 3) * 4) + (i_14 * 2)) + (rv >> 2))]);
      }
      scores_max[i_14] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      scores_max[i_15] = max(scores_max[i_15], scores_max_prev[i_15]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      scores_scale[i_16] = exp2f(((scores_max_prev[i_16] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_16] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 16; ++i_17) {
      acc_s[i_17] = exp2f(((acc_s[i_17] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_17 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 2; ++i_18) {
      scores_sum[i_18] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        scores_sum[i_18] = (scores_sum[i_18] + acc_s[((((rv_1 & 3) * 4) + (i_18 * 2)) + (rv_1 >> 2))]);
      }
      scores_sum[i_18] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_18]);
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      logsum[i_19] = ((logsum[i_19] * scores_scale[i_19]) + scores_sum[i_19]);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 8; ++i_20) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s + (i_20 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__1)) = __float22bfloat162_rn(*(float2*)(&(v_)));
      *(uint1*)(acc_s_cast + (i_20 * 2)) = __1;
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 64; ++i_21) {
      acc_o[i_21] = (acc_o[i_21] * scores_scale[((i_21 & 3) >> 1)]);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      for (int i_22 = 0; i_22 < 8; ++i_22) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((((k & 1) * 4096) + ((i_22 >> 2) * 2048)) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_22 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])) + 0, B_local_1 + (i_22 * 8));
      }
      for (int j_1 = 0; j_1 < 8; ++j_1) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_23 = 0; i_23 < 4; ++i_23) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((k & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_23 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), V+((((((((int64_t)k) * (int64_t)32768) + (((int64_t)i_23) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)65536), (k < 254));
    }
    tl::cp_async_commit();
  }
  if (33 <= valid_kv_seqlen) {
    #pragma unroll
    for (int i_24 = 0; i_24 < 8; ++i_24) {
      *(float2*)(acc_s + (i_24 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_25 = 0; i_25 < 2; ++i_25) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((((valid_kv_seqlen + 31) & 63) >> 5) * 4096) + ((ki_2 >> 2) * 2048)) + (i_25 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_25 * 8));
      }
      for (int j_2 = 0; j_2 < 2; ++j_2) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j_2 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j_2 * 8) + 4)));
      }
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 16; ++i_26) {
      float condval_2;
      if ((((((((valid_kv_seqlen + 31) >> 5) * 32) + ((i_26 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_26 & 1)) < (valid_kv_seqlen + 64))) {
        condval_2 = acc_s[i_26];
      } else {
        condval_2 = -CUDART_INF_F;
      }
      acc_s[i_26] = condval_2;
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 2; ++i_27) {
      scores_max_prev[i_27] = scores_max[i_27];
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 2; ++i_28) {
      scores_max[i_28] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 2; ++i_29) {
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
        scores_max[i_29] = max(scores_max[i_29], acc_s[((((rv_2 & 3) * 4) + (i_29 * 2)) + (rv_2 >> 2))]);
      }
      scores_max[i_29] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_29]);
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 2; ++i_30) {
      scores_max[i_30] = max(scores_max[i_30], scores_max_prev[i_30]);
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 2; ++i_31) {
      scores_scale[i_31] = exp2f(((scores_max_prev[i_31] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_31] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 16; ++i_32) {
      acc_s[i_32] = exp2f(((acc_s[i_32] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_32 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_33 = 0; i_33 < 2; ++i_33) {
      scores_sum[i_33] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
        scores_sum[i_33] = (scores_sum[i_33] + acc_s[((((rv_3 & 3) * 4) + (i_33 * 2)) + (rv_3 >> 2))]);
      }
      scores_sum[i_33] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_33]);
    }
    #pragma unroll
    for (int i_34 = 0; i_34 < 2; ++i_34) {
      logsum[i_34] = ((logsum[i_34] * scores_scale[i_34]) + scores_sum[i_34]);
    }
    #pragma unroll
    for (int i_35 = 0; i_35 < 8; ++i_35) {
      uint1 __2;
      float2 v__1 = *(float2*)(acc_s + (i_35 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__2)) = __float22bfloat162_rn(*(float2*)(&(v__1)));
      *(uint1*)(acc_s_cast + (i_35 * 2)) = __2;
    }
    #pragma unroll
    for (int i_36 = 0; i_36 < 64; ++i_36) {
      acc_o[i_36] = (acc_o[i_36] * scores_scale[((i_36 & 3) >> 1)]);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
      for (int i_37 = 0; i_37 < 8; ++i_37) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((valid_kv_seqlen + 31) & 63) >> 5) * 4096) + ((i_37 >> 2) * 2048)) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_37 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_37 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])) + 0, B_local_1 + (i_37 * 8));
      }
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_3 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_3 * 8) + 4)));
      }
    }
  }
  if (1 <= valid_kv_seqlen) {
    #pragma unroll
    for (int i_38 = 0; i_38 < 8; ++i_38) {
      *(float2*)(acc_s + (i_38 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_4 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_39 = 0; i_39 < 2; ++i_39) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((((valid_kv_seqlen + 31) >> 5) + 1) & 1) * 4096) + ((ki_4 >> 2) * 2048)) + (i_39 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_39 * 8));
      }
      for (int j_4 = 0; j_4 < 2; ++j_4) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j_4 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j_4 * 8) + 4)));
      }
    }
    #pragma unroll
    for (int i_40 = 0; i_40 < 16; ++i_40) {
      float condval_3;
      if ((((((((valid_kv_seqlen + 31) >> 5) * 32) + ((i_40 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_40 & 1)) < (valid_kv_seqlen + 32))) {
        condval_3 = acc_s[i_40];
      } else {
        condval_3 = -CUDART_INF_F;
      }
      acc_s[i_40] = condval_3;
    }
    #pragma unroll
    for (int i_41 = 0; i_41 < 2; ++i_41) {
      scores_max_prev[i_41] = scores_max[i_41];
    }
    #pragma unroll
    for (int i_42 = 0; i_42 < 2; ++i_42) {
      scores_max[i_42] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_43 = 0; i_43 < 2; ++i_43) {
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 8; ++rv_4) {
        scores_max[i_43] = max(scores_max[i_43], acc_s[((((rv_4 & 3) * 4) + (i_43 * 2)) + (rv_4 >> 2))]);
      }
      scores_max[i_43] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_43]);
    }
    #pragma unroll
    for (int i_44 = 0; i_44 < 2; ++i_44) {
      scores_max[i_44] = max(scores_max[i_44], scores_max_prev[i_44]);
    }
    #pragma unroll
    for (int i_45 = 0; i_45 < 2; ++i_45) {
      scores_scale[i_45] = exp2f(((scores_max_prev[i_45] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_45] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_46 = 0; i_46 < 16; ++i_46) {
      acc_s[i_46] = exp2f(((acc_s[i_46] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_46 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_47 = 0; i_47 < 2; ++i_47) {
      scores_sum[i_47] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_5 = 0; rv_5 < 8; ++rv_5) {
        scores_sum[i_47] = (scores_sum[i_47] + acc_s[((((rv_5 & 3) * 4) + (i_47 * 2)) + (rv_5 >> 2))]);
      }
      scores_sum[i_47] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_47]);
    }
    #pragma unroll
    for (int i_48 = 0; i_48 < 2; ++i_48) {
      logsum[i_48] = ((logsum[i_48] * scores_scale[i_48]) + scores_sum[i_48]);
    }
    #pragma unroll
    for (int i_49 = 0; i_49 < 8; ++i_49) {
      uint1 __3;
      float2 v__2 = *(float2*)(acc_s + (i_49 * 2));
      *reinterpret_cast<__nv_bfloat162*>(&(__3)) = __float22bfloat162_rn(*(float2*)(&(v__2)));
      *(uint1*)(acc_s_cast + (i_49 * 2)) = __3;
    }
    #pragma unroll
    for (int i_50 = 0; i_50 < 64; ++i_50) {
      acc_o[i_50] = (acc_o[i_50] * scores_scale[((i_50 & 3) >> 1)]);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki_5 = 0; ki_5 < 2; ++ki_5) {
      for (int i_51 = 0; i_51 < 8; ++i_51) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((valid_kv_seqlen + 31) >> 5) + 1) & 1) * 4096) + ((i_51 >> 2) * 2048)) + (ki_5 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_51 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_51 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])) + 0, B_local_1 + (i_51 * 8));
      }
      for (int j_5 = 0; j_5 < 8; ++j_5) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_5 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_5 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_5 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_52 = 0; i_52 < 64; ++i_52) {
    acc_o[i_52] = (acc_o[i_52] / logsum[((i_52 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_53 = 0; i_53 < 2; ++i_53) {
    logsum[i_53] = (log2f(logsum[i_53]) + (scores_max[i_53] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
  }
  __syncthreads();
  #pragma unroll
  for (int i_54 = 0; i_54 < 8; ++i_54) {
    tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_54 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((bfloat16_t)acc_o[(i_54 * 8)]), ((bfloat16_t)acc_o[((i_54 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_o[((i_54 * 8) + 2)]), ((bfloat16_t)acc_o[((i_54 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_o[((i_54 * 8) + 4)]), ((bfloat16_t)acc_o[((i_54 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_o[((i_54 * 8) + 6)]), ((bfloat16_t)acc_o[((i_54 * 8) + 7)])));
  }
  __syncthreads();
  *(uint1*)(Output + ((((int)by) * 256) + (((int)threadIdx.x) * 2))) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((int)threadIdx.x) * 2));
}


} // kernel
// Strategy: gqa_decode_tl_1_8192_32_16_8_128
// selected_hparams: [32, 64, 1, 2, 128].
// smem: 49152 bytes.
// use_cooperative_groups: 0.
// layout: (1, 8, 1), (32, 64, 1)
// block_dim=(128, 1, 1).
// latency: 0 ms vs [ref-0 sim-0], idx: 4