
#include <cublasLt.h>

#include "common_cublaslt.h"
#include "xop/common_cuda.h"
#include "xop/ops_impl/global_resource.h"

#include <vector>
#include <algorithm>

namespace xop {

#define ENABLE_FP8_BLOCKSCALE false
// GemmLt cublaslt_gemm;
// cublaslt_gemm.init(handle, n, m, k, type_input, type_output, type_compute, true);
// cublasLtMatmulAlgo_t algo;
// cublaslt_gemm.get_algo(0, algo);
// cublaslt_gemm.run(algo, weight.data_ptr(), input.data_ptr(), output.data_ptr());
struct GemmLt {
  cublasLtHandle_t handle_;

  cublasLtMatrixLayout_t a_desc_;
  cublasLtMatrixLayout_t b_desc_;
  cublasLtMatrixLayout_t c_desc_;
  cublasLtMatrixLayout_t d_desc_;

  cublasLtMatmulDesc_t matmul_desc_;

  cublasLtMatmulPreference_t preference_;

  static constexpr int kAlgoMaxNum = 1024;
  cublasLtMatmulHeuristicResult_t heur_res_[kAlgoMaxNum];
  std::vector<cublasLtMatmulAlgo_t> valid_algos_;

  int ret_algo_num_;

  float alpha_;
  float beta_;

  void *workspace_;
  size_t workspace_size_;

  // CUDA_R_16BF
  // CUBLAS_COMPUTE_16F / CUBLAS_COMPUTE_32F
  // CUDA_R_32F
  void init(cublasLtHandle_t handle, int m, int n, int k, 
            cudaDataType_t type_input, cudaDataType_t type_output, 
            cublasComputeType_t type_compute, bool is_tuning = false) {

    handle_ = handle;
    // cublasLtLoggerSetLevel(5);
    cudaDataType_t type_scale = CUDA_R_32F;
    int batch = 1;
    int64_t a_stride = m * k;
    int64_t b_stride = n * k;
    int64_t c_stride = m * n;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
  
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmul_desc_, type_compute, type_scale));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  
    if constexpr (ENABLE_FP8_BLOCKSCALE) {
      // // https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/
      // // https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtBlk128x128Fp8Matmul
      // cublasLtMatmulMatrixScale_t AScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
      // cublasLtMatmulMatrixScale_t BScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
      // cublasLtMatmulMatrixScale_t CScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
      // cublasLtMatmulMatrixScale_t DScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
      // cublasLtMatmulMatrixScale_t DOutScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;

      // // set block scaling mode
      // CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &AScaleMode, sizeof(AScaleMode)));
      // CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &BScaleMode, sizeof(BScaleMode)));
  
      // // set scaling factors
      // CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
      // CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale))); 
      
      // CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc_, CUDA_R_8F_E4M3, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
      // CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc_, CUDA_R_8F_E4M3, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
      // CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_desc_, CUDA_R_16BF, m, n, c_stride));
      // CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&d_desc_, CUDA_R_16BF, m, n, c_stride));
    }
    else {
      CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc_, type_input, k, m, k)); // CUDA_R_16F
      CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc_, type_input, k, n, k));
      CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_desc_, type_output, m, n, m));
    
      CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(a_desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
      CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(b_desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
      CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(c_desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    
      CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(a_desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride, sizeof(a_stride)));
      CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(b_desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride, sizeof(b_stride)));
      CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(c_desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride, sizeof(c_stride)));
    
    }

    alpha_ = 1.f;
    beta_ = 0.f;
    workspace_ = nullptr;
    workspace_size_ = 0;
  
    ret_algo_num_ = 0;
    if (is_tuning) {
      // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
      // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
      // directly come from cudaMalloc)
      cublasLtMatmulPreferenceCreate(&preference_);
      // CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
      
      cublasLtMatmulAlgoGetHeuristic(handle_, matmul_desc_, a_desc_, b_desc_,
                                    c_desc_, c_desc_, preference_, kAlgoMaxNum,
                                    heur_res_, &ret_algo_num_);
      
      for (int i = 0; i < ret_algo_num_; ++i) {
        workspace_size_ = std::max(workspace_size_, heur_res_[i].workspaceSize);
        workspace_ = GlobalBuffer::instance().GetDeviceBuffer(kDevBufferPoolWorkspace, workspace_size_);        
      }
      // for (int i = 0; i < ret_algo_num_; ++i) {
      //   DebugUtils::PrintAlgo(heur_res_[i]);
      // }
    }
  }

  inline int get_algo_num() { return ret_algo_num_; }
  
  bool get_algo(int algo_id, cublasLtMatmulAlgo_t& algo) {
    if (algo_id >= ret_algo_num_) {
      return false;
    }
    memcpy(&algo, &heur_res_[algo_id].algo, sizeof(algo));      
    return true;
  }

  void run(cublasLtMatmulAlgo_t& algo, const void *a, const void *b, void *c, cudaStream_t stream) {
    CUBLASLT_CHECK(cublasLtMatmul(handle_, matmul_desc_, &alpha_, a, a_desc_, b,
                          b_desc_, &beta_, c, c_desc_, c, c_desc_,
                          &algo, workspace_, workspace_size_, stream));
  }
};

} // namespace xop