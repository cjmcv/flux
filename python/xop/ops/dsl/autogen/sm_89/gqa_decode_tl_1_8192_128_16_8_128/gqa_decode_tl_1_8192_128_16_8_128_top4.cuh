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
__device__ __forceinline__ void flashattn_kernel_1_8192_128_16_8_128(const int bx, const int by, const int bz,
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
  int valid_kv_seqlen = (edge[0] + 1);
  if (0 < valid_kv_seqlen) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(K[((((i_2 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)])), (&(V[((((i_3 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
    }
    tl::cp_async_commit();
  }
  if (64 < valid_kv_seqlen) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(K[(((((i_4 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 32768)])), (&(V[(((((i_5 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)by) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536)])));
    }
    tl::cp_async_commit();
  }
  for (int k = 0; k < (((valid_kv_seqlen + 63) >> 6) - 2); ++k) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_6 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki = 0; ki < 8; ++ki) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((k & 1) * 8192) + ((ki >> 2) * 4096)) + (i_7 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_7 * 8));
      }
      for (int j = 0; j < 4; ++j) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((k & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_8 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(K[((((((((int64_t)k) * (int64_t)65536) + (((int64_t)i_8) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)131072)])), (k < 126));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_9 = 0; i_9 < 32; ++i_9) {
      float condval_1;
      if ((((((k * 64) + ((i_9 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_9 & 1)) < valid_kv_seqlen)) {
        condval_1 = acc_s[i_9];
      } else {
        condval_1 = -CUDART_INF_F;
      }
      acc_s[i_9] = condval_1;
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_5 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_5, broadcast_var_5);
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      scores_max_clear[i_10] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        scores_max_clear[i_10] = max(scores_max_clear[i_10], acc_s[((((rv & 7) * 4) + (i_10 * 2)) + (rv >> 3))]);
      }
      scores_max_clear[i_10] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max_clear[i_10]);
      scores_max[i_10] = max(scores_max[i_10], scores_max_clear[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      scores_max[i_11] = max(scores_max[i_11], scores_max_prev[i_11]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      scores_scale[i_12] = exp2f(((scores_max_prev[i_12] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_12] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 32; ++i_13) {
      acc_s[i_13] = exp2f(((acc_s[i_13] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_13 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      scores_sum[i_14] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        scores_sum[i_14] = (scores_sum[i_14] + acc_s[((((rv_1 & 7) * 4) + (i_14 * 2)) + (rv_1 >> 3))]);
      }
      scores_sum[i_14] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      logsum[i_15] = ((logsum[i_15] * scores_scale[i_15]) + scores_sum[i_15]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 8; ++i_16) {
      uint2 __1;
      float4 v_ = *(float4*)(acc_s + (i_16 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
      *(uint2*)(acc_s_cast + (i_16 * 4)) = __1;
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 64; ++i_17) {
      acc_o[i_17] = (acc_o[i_17] * scores_scale[((i_17 & 3) >> 1)]);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      for (int i_18 = 0; i_18 < 8; ++i_18) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((((k & 1) * 8192) + ((i_18 >> 2) * 4096)) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_18 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_18 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 24576)])) + 0, B_local_1 + (i_18 * 8));
      }
      for (int j_1 = 0; j_1 < 8; ++j_1) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_19 = 0; i_19 < 8; ++i_19) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((k & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_19 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)])), (&(V[((((((((int64_t)k) * (int64_t)65536) + (((int64_t)i_19) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)1024)) + (((int64_t)((int)by)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)131072)])), (k < 126));
    }
    tl::cp_async_commit();
  }
  if (65 <= valid_kv_seqlen) {
    #pragma unroll
    for (int i_20 = 0; i_20 < 8; ++i_20) {
      float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_20 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_21 = 0; i_21 < 4; ++i_21) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((((valid_kv_seqlen + 63) & 127) >> 6) * 8192) + ((ki_2 >> 2) * 4096)) + (i_21 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_21 * 8));
      }
      for (int j_2 = 0; j_2 < 4; ++j_2) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j_2 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j_2 * 8) + 4)));
      }
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 32; ++i_22) {
      float condval_2;
      if ((((((((valid_kv_seqlen + 63) >> 6) * 64) + ((i_22 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_22 & 1)) < (valid_kv_seqlen + 128))) {
        condval_2 = acc_s[i_22];
      } else {
        condval_2 = -CUDART_INF_F;
      }
      acc_s[i_22] = condval_2;
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_7 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_7, broadcast_var_7);
    #pragma unroll
    for (int i_23 = 0; i_23 < 2; ++i_23) {
      scores_max_clear_1[i_23] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
        scores_max_clear_1[i_23] = max(scores_max_clear_1[i_23], acc_s[((((rv_2 & 7) * 4) + (i_23 * 2)) + (rv_2 >> 3))]);
      }
      scores_max_clear_1[i_23] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max_clear_1[i_23]);
      scores_max[i_23] = max(scores_max[i_23], scores_max_clear_1[i_23]);
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 2; ++i_24) {
      scores_max[i_24] = max(scores_max[i_24], scores_max_prev[i_24]);
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 2; ++i_25) {
      scores_scale[i_25] = exp2f(((scores_max_prev[i_25] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_25] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 32; ++i_26) {
      acc_s[i_26] = exp2f(((acc_s[i_26] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_26 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 2; ++i_27) {
      scores_sum[i_27] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
        scores_sum[i_27] = (scores_sum[i_27] + acc_s[((((rv_3 & 7) * 4) + (i_27 * 2)) + (rv_3 >> 3))]);
      }
      scores_sum[i_27] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_27]);
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 2; ++i_28) {
      logsum[i_28] = ((logsum[i_28] * scores_scale[i_28]) + scores_sum[i_28]);
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 8; ++i_29) {
      uint2 __2;
      float4 v__1 = *(float4*)(acc_s + (i_29 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__2))[1] = __float22bfloat162_rn(((float2*)(&v__1))[1]);
      *(uint2*)(acc_s_cast + (i_29 * 4)) = __2;
    }
    #pragma unroll
    for (int i_30 = 0; i_30 < 64; ++i_30) {
      acc_o[i_30] = (acc_o[i_30] * scores_scale[((i_30 & 3) >> 1)]);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
      for (int i_31 = 0; i_31 < 8; ++i_31) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((valid_kv_seqlen + 63) & 127) >> 6) * 8192) + ((i_31 >> 2) * 4096)) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_31 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_31 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 24576)])) + 0, B_local_1 + (i_31 * 8));
      }
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_3 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_3 * 8) + 4)));
      }
    }
  }
  if (1 <= valid_kv_seqlen) {
    #pragma unroll
    for (int i_32 = 0; i_32 < 8; ++i_32) {
      float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_32 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_4 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + 0);
      for (int i_33 = 0; i_33 < 4; ++i_33) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((((valid_kv_seqlen + 63) >> 6) + 1) & 1) * 8192) + ((ki_4 >> 2) * 4096)) + (i_33 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])) + 0, B_local + (i_33 * 8));
      }
      for (int j_4 = 0; j_4 < 4; ++j_4) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j_4 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j_4 * 8) + 4)));
      }
    }
    #pragma unroll
    for (int i_34 = 0; i_34 < 32; ++i_34) {
      float condval_3;
      if ((((((((valid_kv_seqlen + 63) >> 6) * 64) + ((i_34 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_34 & 1)) < (valid_kv_seqlen + 64))) {
        condval_3 = acc_s[i_34];
      } else {
        condval_3 = -CUDART_INF_F;
      }
      acc_s[i_34] = condval_3;
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    float broadcast_var_9 = -CUDART_INF_F;
    *(float2*)(scores_max + 0) = make_float2(broadcast_var_9, broadcast_var_9);
    #pragma unroll
    for (int i_35 = 0; i_35 < 2; ++i_35) {
      scores_max_clear_1[i_35] = -CUDART_INF_F;
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
        scores_max_clear_1[i_35] = max(scores_max_clear_1[i_35], acc_s[((((rv_4 & 7) * 4) + (i_35 * 2)) + (rv_4 >> 3))]);
      }
      scores_max_clear_1[i_35] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max_clear_1[i_35]);
      scores_max[i_35] = max(scores_max[i_35], scores_max_clear_1[i_35]);
    }
    #pragma unroll
    for (int i_36 = 0; i_36 < 2; ++i_36) {
      scores_max[i_36] = max(scores_max[i_36], scores_max_prev[i_36]);
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 2; ++i_37) {
      scores_scale[i_37] = exp2f(((scores_max_prev[i_37] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_37] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_38 = 0; i_38 < 32; ++i_38) {
      acc_s[i_38] = exp2f(((acc_s[i_38] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_38 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_39 = 0; i_39 < 2; ++i_39) {
      scores_sum[i_39] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_5 = 0; rv_5 < 16; ++rv_5) {
        scores_sum[i_39] = (scores_sum[i_39] + acc_s[((((rv_5 & 7) * 4) + (i_39 * 2)) + (rv_5 >> 3))]);
      }
      scores_sum[i_39] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_39]);
    }
    #pragma unroll
    for (int i_40 = 0; i_40 < 2; ++i_40) {
      logsum[i_40] = ((logsum[i_40] * scores_scale[i_40]) + scores_sum[i_40]);
    }
    #pragma unroll
    for (int i_41 = 0; i_41 < 8; ++i_41) {
      uint2 __3;
      float4 v__2 = *(float4*)(acc_s + (i_41 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&v__2))[1]);
      *(uint2*)(acc_s_cast + (i_41 * 4)) = __3;
    }
    #pragma unroll
    for (int i_42 = 0; i_42 < 64; ++i_42) {
      acc_o[i_42] = (acc_o[i_42] * scores_scale[((i_42 & 3) >> 1)]);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
      for (int i_43 = 0; i_43 < 8; ++i_43) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((((valid_kv_seqlen + 63) >> 6) + 1) & 1) * 8192) + ((i_43 >> 2) * 4096)) + (ki_5 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_43 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_43 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 24576)])) + 0, B_local_1 + (i_43 * 8));
      }
      for (int j_5 = 0; j_5 < 8; ++j_5) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_5 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_5 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_5 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_44 = 0; i_44 < 64; ++i_44) {
    acc_o[i_44] = (acc_o[i_44] / logsum[((i_44 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_45 = 0; i_45 < 2; ++i_45) {
    logsum[i_45] = (log2f(logsum[i_45]) + (scores_max[i_45] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
  }
  __syncthreads();
  #pragma unroll
  for (int i_46 = 0; i_46 < 32; ++i_46) {
    if (((((((int)threadIdx.x) >> 5) * 8) + ((i_46 & 1) * 4)) + ((((int)threadIdx.x) & 31) >> 3)) < 1) {
      uint1 __4;
      float2 v__3 = *(float2*)(acc_o + (i_46 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
      *(uint1*)(O_shared_local_cast + 0) = __4;
      *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_46 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_46 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(O_shared_local_cast + 0);
    }
  }
  __syncthreads();
  *(uint1*)(Output + ((((int)by) * 256) + (((int)threadIdx.x) * 2))) = *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((int)threadIdx.x) * 2));
}


} // kernel
// Strategy: gqa_decode_tl_1_8192_128_16_8_128
// selected_hparams: [64, 64, 1, 2, 128].
// smem: 81920 bytes.
// use_cooperative_groups: 0.
// layout: (1, 8, 1), (64, 64, 1)
// block_dim=(128, 1, 1).


extern "C" int create_gqa_decode_tl_1_8192_128_16_8_128(bfloat16_t* __restrict__ Q, bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ V, int* __restrict__ edge, uint8_t* __restrict__ mask, bfloat16_t* __restrict__ glse, bfloat16_t* __restrict__ Output_partial, bfloat16_t* __restrict__ Output) {

	return 0;
}
#define LAUNCH_INFO_gqa_decode_tl_1_8192_128_16_8_128 dim3(1, 8, 1), dim3(128, 1, 1), 81920, stream

// latency: 0.01344 ms vs [ref-0.0299 sim-0.99479], idx: 16