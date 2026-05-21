
#include <cublasLt.h>

#include "common_cublaslt.h"
#include "xop/common_cuda.h"
#include "xop/ops_impl/global_resource.h"

#include <vector>
#include <algorithm>

namespace xop {

struct GemmDsl {
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
  
  void init(cublasLtHandle_t handle, int m, int n, int k, 
            cudaDataType_t type_input, cudaDataType_t type_output, 
            cublasComputeType_t type_compute, bool is_tuning = false) {

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