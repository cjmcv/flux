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
__device__ __forceinline__ void flashattn_kernel_1_8192_64_16_8_128__0(const int bx, const int by, const int bz,
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
  if constexpr (SUB_KERNEL_ID == 0) { if (bx >= 1 || by >= 8 || bz >= 2 || threadIdx.x >= 128) { return; } }
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
  float acc_s[32];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  bfloat16_t acc_s_cast[32];
  bfloat16_t A_local[8];
  bfloat16_t B_local[32];
  float scores_max_clear[2];
  bfloat16_t B_local_1[64];
  float scores_max_clear_1[2];
  bfloat16_t O_shared_local_cast[2];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if ((((((((int)threadIdx.x) >> 5) + ((int)by)) >> 2) + i) < 2)) {
      condval = *(uint4*)(Q + (((i * 1024) + (((int)by) * 256)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
    }
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(logsum + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  float broadcast_var_3 = -CUDART_INF_F;
  *(float2*)(scores_max + 0) = make_float2(broadcast_var_3, broadcast_var_3);
  int actual_kv_seqlen = (edge[0] + 1);
  int condval_1;
  if ((((int)bz) == 1)) {
    condval_1 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 1)));
  } else {
    condval_1 = (actual_kv_seqlen >> 1);
  }
  if (0 < condval_1) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(K[(((((((int64_t)i_2) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)bz)) * (((int64_t)actual_kv_seqlen) >> (int64_t)1)) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])), ((((((((int)threadIdx.x) >> 4) + (((int)bz) * (actual_kv_seqlen >> 1))) >> 3) + i_2) < 1024) && (0 <= (((i_2 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 1))))));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(V[(((((((int64_t)i_3) * (int64_t)8192) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)bz)) * (((int64_t)actual_kv_seqlen) >> (int64_t)1)) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8))])), ((((((((int)threadIdx.x) >> 4) + (((int)bz) * (actual_kv_seqlen >> 1))) >> 3) + i_3) < 1024) && (0 <= (((i_3 * 8) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 1))))));
    }
    tl::cp_async_commit();
  }
  int condval_2;
  if ((((int)bz) == 1)) {
    condval_2 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 1)));
  } else {
    condval_2 = (actual_kv_seqlen >> 1);
  }
  for (int k = 0; k < (((condval_2 + 63) >> 6) - 1); ++k) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_4 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki = 0; ki < 8; ++ki) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki >> 2) * 4096) + (i_5 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_5 * 8));
      }
      for (int j = 0; j < 4; ++j) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(K[(((((((((int64_t)k) * (int64_t)65536) + (((int64_t)i_6) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)bz)) * (((int64_t)actual_kv_seqlen) >> (int64_t)1)) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)65536)])), ((((((k * 64) + (i_6 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 1))) < 8128) && (-64 <= ((((k * 64) + (i_6 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 1))))));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_7 = 0; i_7 < 32; ++i_7) {
      int condval_4;
      if ((((int)bz) == 1)) {
        condval_4 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 1)));
      } else {
        condval_4 = (actual_kv_seqlen >> 1);
      }
      float condval_3;
      if ((((((k * 64) + ((i_7 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1)) < condval_4)) {
        condval_3 = acc_s[i_7];
      } else {
        condval_3 = -CUDART_INF_F;
      }
      acc_s[i_7] = condval_3;
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_5 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_5, broadcast_var_5);
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      scores_max_clear[i_8] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        scores_max_clear[i_8] = max(scores_max_clear[i_8], acc_s[((((rv & 7) * 4) + (i_8 * 2)) + (rv >> 3))]);
      }
      scores_max_clear[i_8] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max_clear[i_8]);
      scores_max[i_8] = max(scores_max[i_8], scores_max_clear[i_8]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      scores_max[i_9] = max(scores_max[i_9], scores_max_prev[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      scores_scale[i_10] = exp2f(((scores_max_prev[i_10] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_10] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 32; ++i_11) {
      acc_s[i_11] = exp2f(((acc_s[i_11] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_11 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      scores_sum[i_12] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        scores_sum[i_12] = (scores_sum[i_12] + acc_s[((((rv_1 & 7) * 4) + (i_12 * 2)) + (rv_1 >> 3))]);
      }
      scores_sum[i_12] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_12]);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      logsum[i_13] = ((logsum[i_13] * scores_scale[i_13]) + scores_sum[i_13]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 8; ++i_14) {
      uint2 __1;
      float4 v_ = *(float4*)(acc_s + (i_14 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
      *(uint2*)(acc_s_cast + (i_14 * 4)) = __1;
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 64; ++i_15) {
      acc_o[i_15] = (acc_o[i_15] * scores_scale[((i_15 & 3) >> 1)]);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      for (int i_16 = 0; i_16 < 8; ++i_16) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_16 >> 2) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_16 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])) + 0, B_local_1 + (i_16 * 8));
      }
      for (int j_1 = 0; j_1 < 8; ++j_1) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_17 = 0; i_17 < 8; ++i_17) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_17 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(V[(((((((((int64_t)k) * (int64_t)65536) + (((int64_t)i_17) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + ((((int64_t)((int)bz)) * (((int64_t)actual_kv_seqlen) >> (int64_t)1)) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)65536)])), ((((((k * 64) + (i_17 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 1))) < 8128) && (-64 <= ((((k * 64) + (i_17 * 8)) + (((int)threadIdx.x) >> 4)) + (((int)bz) * (actual_kv_seqlen >> 1))))));
    }
    tl::cp_async_commit();
  }
  int condval_5;
  if ((((int)bz) == 1)) {
    condval_5 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 1)));
  } else {
    condval_5 = (actual_kv_seqlen >> 1);
  }
  if (1 <= condval_5) {
    #pragma unroll
    for (int i_18 = 0; i_18 < 8; ++i_18) {
      float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_18 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_19 = 0; i_19 < 4; ++i_19) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki_2 >> 2) * 4096) + (i_19 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_19 * 8));
      }
      for (int j_2 = 0; j_2 < 4; ++j_2) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j_2 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j_2 * 8) + 4)));
      }
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 32; ++i_20) {
      int condval_7;
      if ((((int)bz) == 1)) {
        condval_7 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 1)));
      } else {
        condval_7 = (actual_kv_seqlen >> 1);
      }
      int condval_8;
      if ((((int)bz) == 1)) {
        condval_8 = (actual_kv_seqlen - (((int)bz) * (actual_kv_seqlen >> 1)));
      } else {
        condval_8 = (actual_kv_seqlen >> 1);
      }
      float condval_6;
      if ((((((((condval_7 + 63) >> 6) * 64) + ((i_20 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_20 & 1)) < (condval_8 + 64))) {
        condval_6 = acc_s[i_20];
      } else {
        condval_6 = -CUDART_INF_F;
      }
      acc_s[i_20] = condval_6;
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_7 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_7, broadcast_var_7);
    #pragma unroll
    for (int i_21 = 0; i_21 < 2; ++i_21) {
      scores_max_clear_1[i_21] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
        scores_max_clear_1[i_21] = max(scores_max_clear_1[i_21], acc_s[((((rv_2 & 7) * 4) + (i_21 * 2)) + (rv_2 >> 3))]);
      }
      scores_max_clear_1[i_21] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max_clear_1[i_21]);
      scores_max[i_21] = max(scores_max[i_21], scores_max_clear_1[i_21]);
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 2; ++i_22) {
      scores_max[i_22] = max(scores_max[i_22], scores_max_prev[i_22]);
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 2; ++i_23) {
      scores_scale[i_23] = exp2f(((scores_max_prev[i_23] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_23] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 32; ++i_24) {
      acc_s[i_24] = exp2f(((acc_s[i_24] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_24 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 2; ++i_25) {
      scores_sum[i_25] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
        scores_sum[i_25] = (scores_sum[i_25] + acc_s[((((rv_3 & 7) * 4) + (i_25 * 2)) + (rv_3 >> 3))]);
      }
      scores_sum[i_25] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_25]);
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 2; ++i_26) {
      logsum[i_26] = ((logsum[i_26] * scores_scale[i_26]) + scores_sum[i_26]);
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 8; ++i_27) {
      uint2 __2;
      float4 v__1 = *(float4*)(acc_s + (i_27 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__2))[1] = __float22bfloat162_rn(((float2*)(&v__1))[1]);
      *(uint2*)(acc_s_cast + (i_27 * 4)) = __2;
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 64; ++i_28) {
      acc_o[i_28] = (acc_o[i_28] * scores_scale[((i_28 & 3) >> 1)]);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
      for (int i_29 = 0; i_29 < 8; ++i_29) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_29 >> 2) * 4096) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_29 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_29 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])) + 0, B_local_1 + (i_29 * 8));
      }
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_3 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_3 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_30 = 0; i_30 < 64; ++i_30) {
    acc_o[i_30] = (acc_o[i_30] / logsum[((i_30 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 2; ++i_31) {
    logsum[i_31] = (log2f(logsum[i_31]) + (scores_max[i_31] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
  }
  if ((((int)threadIdx.x) % 4) == 0) {
    #pragma unroll
    for (int i_32 = 0; i_32 < 2; ++i_32) {
      if (((((((int)threadIdx.x) >> 5) * 8) + (i_32 * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1) {
        glse[((((((((int)threadIdx.x) >> 5) * 32) + (i_32 * 16)) + (((int)by) * 4)) + (((((int)threadIdx.x) & 31) >> 2) * 2)) + ((int)bz))] = ((bfloat16_t)logsum[i_32]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_33 = 0; i_33 < 32; ++i_33) {
    if (((((((int)threadIdx.x) >> 5) * 8) + ((i_33 & 1) * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1) {
      uint1 __3;
      float2 v__2 = *(float2*)(acc_o + (i_33 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
      *(uint1*)(O_shared_local_cast + 0) = __3;
      *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_33 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_33 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(O_shared_local_cast + 0);
    }
  }
  __syncthreads();
  *(uint1*)(Output_partial + ((((((int)by) * 512) + ((((int)threadIdx.x) >> 6) * 256)) + (((int)bz) * 128)) + ((((int)threadIdx.x) & 63) * 2))) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((int)threadIdx.x) * 2));
}


template <typename T,
          int THREAD_NUM,
          int SUB_KERNEL_ID,
          int M, 
          int HEAD,
          int GROUPS,
          int DIM>
__device__ __forceinline__ void flashattn_kernel_1_8192_64_16_8_128__1(const int bx, const int by, const int bz,
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
  if constexpr (SUB_KERNEL_ID == 0) { if (bx >= 1 || by >= 8 || bz >= 2 || threadIdx.x >= 128) { return; } }
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
  bfloat16_t lse_local[2];
  float lse_max_local[1];
  bfloat16_t po_local[1];
  float scale_local[1];
  lse_logsum_local[0] = 0x0p+0f/*0.000000e+00*/;
  o_accum_local[0] = 0x0p+0f/*0.000000e+00*/;
  *(uint1*)(lse_local + 0) = *(uint1*)(glse + (((int)bx) * 2));
  lse_max_local[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 2; ++rv) {
    lse_max_local[0] = max(lse_max_local[0], ((float)lse_local[rv]));
  }
  for (int k = 0; k < 2; ++k) {
    lse_logsum_local[0] = (lse_logsum_local[0] + exp2f((((float)lse_local[k]) - lse_max_local[0])));
  }
  lse_logsum_local[0] = (log2f(lse_logsum_local[0]) + lse_max_local[0]);
  for (int k_1 = 0; k_1 < 2; ++k_1) {
    po_local[0] = Output_partial[(((((int)bx) * 256) + (k_1 * 128)) + ((int)threadIdx.x))];
    scale_local[0] = exp2f((((float)lse_local[k_1]) - lse_logsum_local[0]));
    o_accum_local[0] = (o_accum_local[0] + (((float)po_local[0]) * scale_local[0]));
  }
  Output[((((int)bx) * 128) + ((int)threadIdx.x))] = ((bfloat16_t)o_accum_local[0]);
}


} // kernel
// Strategy: gqa_decode_tl_1_8192_64_16_8_128
// selected_hparams: [64, 64, 2, 1, 128].
// smem: 0 bytes.
// use_cooperative_groups: 0.
// layout: (1, 8, 2), (64, 64, 2), (16, 1, 1), (64, 64, 2)
// block_dim=(128, 1, 1).


extern "C" int create_gqa_decode_tl_1_8192_64_16_8_128(bfloat16_t* __restrict__ Q, bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ V, int* __restrict__ edge, uint8_t* __restrict__ mask, bfloat16_t* __restrict__ glse, bfloat16_t* __restrict__ Output_partial, bfloat16_t* __restrict__ Output) {

	return 0;
}
#define LAUNCH_INFO_gqa_decode_tl_1_8192_64_16_8_128 dim3(16, 1, 1), dim3(128, 1, 1), 0, stream

// latency: 0.01816 ms vs [ref-0.03209 sim-0.99153], idx: 18