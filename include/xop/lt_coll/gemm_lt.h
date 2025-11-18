
#include <cublasLt.h>

#include "common_cublaslt.h"
#include "xop/common_cuda.h"
#include "xop/ops_impl/global_resource.h"

#include <vector>
#include <algorithm>

namespace xop {

// false is not supported for now.
#define IS_USE_HEURISTIC true

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
  
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc_, type_input, k, m, k)); // CUDA_R_16F
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc_, type_input, k, n, k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_desc_, type_output, m, n, m));
  
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(a_desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(b_desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(c_desc_, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
  
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(a_desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride, sizeof(a_stride)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(b_desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride, sizeof(b_stride)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(c_desc_, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride, sizeof(c_stride)));
  
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&matmul_desc_, type_compute, type_scale));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  
    alpha_ = 1.f;
    beta_ = 0.f;
    workspace_ = nullptr;
    workspace_size_ = 0;
  
    ret_algo_num_ = 0;
    if (is_tuning) {
      if constexpr (IS_USE_HEURISTIC) {
        cublasLtMatmulPreferenceCreate(&preference_);
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
      else {
        int const max_id = 1024;
        int algo_ids[max_id];
        int algo_cnt = 0;
        cublasLtMatmulAlgoGetIds(
          handle_, type_compute, CUDA_R_32F,
          type_input, type_input, type_output, type_output, 
          max_id, algo_ids, &algo_cnt);

        cublasLtMatmulAlgo_t algo;
        for (int idx = 0; idx < algo_cnt; ++idx) {
          if (cublasLtMatmulAlgoInit(handle, type_compute, CUDA_R_32F, type_input, type_input, type_output, type_output, algo_ids[idx], &algo)
              != CUBLAS_STATUS_SUCCESS) continue;
          if (cublasLtMatmulAlgoCheck(handle, matmul_desc_, a_desc_, b_desc_, c_desc_, c_desc_, &algo, &heur_res_[idx])
              == CUBLAS_STATUS_SUCCESS) {
              valid_algos_.push_back(algo);                  
          }
        }
        ret_algo_num_ = valid_algos_.size();
        printf("ret_algo_num_: %d.\n", ret_algo_num_);
        for (int i = 0; i < ret_algo_num_; ++i) {
          workspace_size_ = std::max(workspace_size_, heur_res_[i].workspaceSize);
          workspace_ = GlobalBuffer::instance().GetDeviceBuffer(kDevBufferPoolWorkspace, workspace_size_);        
        }
        printf("workspace_size_: %ld.\n", workspace_size_);
      }
    }
  }

  inline int get_algo_num() { return ret_algo_num_; }
  
  bool get_algo(int algo_id, cublasLtMatmulAlgo_t& algo) {
    if (algo_id >= ret_algo_num_) {
      return false;
    }
    if constexpr (IS_USE_HEURISTIC) {
      memcpy(&algo, &heur_res_[algo_id].algo, sizeof(algo));      
    }
    else {
      algo = valid_algos_[algo_id];
    }
    return true;
  }

  void run(cublasLtMatmulAlgo_t& algo, const void *a, const void *b, void *c, cudaStream_t stream) {
    CUBLASLT_CHECK(cublasLtMatmul(handle_, matmul_desc_, &alpha_, a, a_desc_, b,
                          b_desc_, &beta_, c, c_desc_, c, c_desc_,
                          &algo, workspace_, workspace_size_, stream));
  }
};

} // namespace xop