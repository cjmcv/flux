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
    __device__ __forceinline__ void linear_gemm_tl_1_6144_1024(const int bx, const int by, const int bz,
                                                const CUtensorMap *A_desc, const CUtensorMap *B_desc, const void* __restrict__ residual_ptr, const CUtensorMap *C_desc, 
                                                int num_active_tokens,
                                                bool residual) {
  // static_assert(THREAD_NUM==128);
  static_assert(TILE_DIM_X==64); static_assert(TILE_DIM_Y==16); static_assert(TILE_DIM_Z==128);
  static_assert(M==1); static_assert(N==6144); static_assert(K==1024);
  if (bx >= 96 || by >= 1 || bz >= 1) { return; }

  const bfloat16_t* __restrict__ R = static_cast<const bfloat16_t*>(residual_ptr);
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[8];
  bfloat16_t A_local[8];
  bfloat16_t B_local[8];
  __shared__ uint64_t mbarrier_mem[2];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(*A_desc);
    tl::prefetch_tma_descriptor(*B_desc);
    tl::prefetch_tma_descriptor(*C_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 8; ++k) {
      mbarrier[1].wait(((k & 1) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[0].expect_transaction(4096);
        tl::fence_proxy_async();
        tl::tma_load(*A_desc, mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[8192])), (k * 128), 0);
        tl::tma_load(*A_desc, mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[9216])), ((k * 128) + 64), 0);
        mbarrier[0].expect_transaction(16384);
        tl::fence_proxy_async();
        tl::tma_load(*B_desc, mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[0])), (k * 128), (((int)bx) * 64));
        tl::tma_load(*B_desc, mbarrier[0], (&(((bfloat16_t*)buf_dyn_shmem)[4096])), ((k * 128) + 64), (((int)bx) * 64));
      }
      mbarrier[0].arrive();
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      *(float2*)(C_local + (i * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    for (int k_1 = 0; k_1 < 8; ++k_1) {
      mbarrier[0].wait((k_1 & 1));
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])) + 0, A_local + 0);
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])) + 0, B_local + 0);
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
      mbarrier[1].arrive();
    }
    tl::__sync_thread_partial<3, 128>();
    tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)C_local[0]), ((bfloat16_t)C_local[1])), __pack_half2(((bfloat16_t)C_local[2]), ((bfloat16_t)C_local[3])), __pack_half2(((bfloat16_t)C_local[4]), ((bfloat16_t)C_local[5])), __pack_half2(((bfloat16_t)C_local[6]), ((bfloat16_t)C_local[7])));
    tl::fence_proxy_async();
    tl::__sync_thread_partial<3, 128>();
    if (tl::tl_shuffle_elect<128>()) {
      tl::tma_store(*C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), (((int)bx) * 64), 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
}


} // kernel
// Strategy: linear_gemm_tl_1_6144_1024
// selected_hparams: [16, 64, 128, 1, 0, 128, 0, False].
// smem: 20480 bytes.
// use_cooperative_groups: 0.
// layout: (96, 1, 1), (64, 16, 128)
// block_dim=(256, 1, 1).


extern "C" int create_linear_gemm_tl_1_6144_1024(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C, CUtensorMap* out_A_desc, CUtensorMap* out_B_desc, CUtensorMap* out_C_desc, bool to_device) {

	CUtensorMap A_desc;
	CUtensorMapDataType A_desc_type= (CUtensorMapDataType)9;
	cuuint32_t A_desc_tensorRank= 2;
	void *A_desc_globalAddress= A;
	cuuint64_t A_desc_globalDim[2]= {1024,1};
	cuuint64_t A_desc_globalStride[2]= {2,2048};
	cuuint32_t A_desc_boxDim[2]= {64,16};
	cuuint32_t A_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave A_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle A_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion A_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill A_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult A_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &A_desc, A_desc_type, A_desc_tensorRank, A_desc_globalAddress, A_desc_globalDim, A_desc_globalStride + 1, A_desc_boxDim, A_desc_elementStrides, A_desc_interleave, A_desc_swizzle, A_desc_l2Promotion, A_desc_oobFill);

	if (A_desc_result != CUDA_SUCCESS) {
		printf("Error: Failed to initialize the TMA descriptor A_desc");
		return -1;
	}

	CUtensorMap B_desc;
	CUtensorMapDataType B_desc_type= (CUtensorMapDataType)9;
	cuuint32_t B_desc_tensorRank= 2;
	void *B_desc_globalAddress= B;
	cuuint64_t B_desc_globalDim[2]= {1024,6144};
	cuuint64_t B_desc_globalStride[2]= {2,2048};
	cuuint32_t B_desc_boxDim[2]= {64,64};
	cuuint32_t B_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave B_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle B_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion B_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill B_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult B_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &B_desc, B_desc_type, B_desc_tensorRank, B_desc_globalAddress, B_desc_globalDim, B_desc_globalStride + 1, B_desc_boxDim, B_desc_elementStrides, B_desc_interleave, B_desc_swizzle, B_desc_l2Promotion, B_desc_oobFill);

	if (B_desc_result != CUDA_SUCCESS) {
		printf("Error: Failed to initialize the TMA descriptor A_desc");
		return -1;
	}

	CUtensorMap C_desc;
	CUtensorMapDataType C_desc_type= (CUtensorMapDataType)9;
	cuuint32_t C_desc_tensorRank= 2;
	void *C_desc_globalAddress= C;
	cuuint64_t C_desc_globalDim[2]= {6144,1};
	cuuint64_t C_desc_globalStride[2]= {2,12288};
	cuuint32_t C_desc_boxDim[2]= {64,16};
	cuuint32_t C_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave C_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle C_desc_swizzle= (CUtensorMapSwizzle)3;
	CUtensorMapL2promotion C_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill C_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult C_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &C_desc, C_desc_type, C_desc_tensorRank, C_desc_globalAddress, C_desc_globalDim, C_desc_globalStride + 1, C_desc_boxDim, C_desc_elementStrides, C_desc_interleave, C_desc_swizzle, C_desc_l2Promotion, C_desc_oobFill);

	if (C_desc_result != CUDA_SUCCESS) {
		printf("Error: Failed to initialize the TMA descriptor A_desc");
		return -1;
	}
	if (to_device) {
		cudaMemcpy(out_A_desc, &A_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
		cudaMemcpy(out_B_desc, &B_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
		cudaMemcpy(out_C_desc, &C_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
	} else {
		*out_A_desc = A_desc;
		*out_B_desc = B_desc;
		*out_C_desc = C_desc;
	}
	//	linear_kernel<<<dim3(96, 1, 1), dim3(256, 1, 1), 20480, stream>>>(A_desc, B_desc, C_desc);

	return 0;
}


// latency: 0 ms vs [ref-0 sim-0], idx: -1