
#pragma once

#include <cublasLt.h>

#include "xop/xop.h"

static char const* CublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

#define CUBLASLT_CHECK(status)                                           \
  do {                                                                   \
    cublasStatus_t error = status;                                       \
    XOP_CHECK(error == CUBLAS_STATUS_SUCCESS)                            \
        << "Got cublasLt error: " << CublasGetErrorString(error) << "("  \
        << static_cast<int>(error) << ") at: " << #status << "\n";       \
  } while (0)

namespace xop {

struct DebugUtils {
  static void PrintAlgo(const cublasLtMatmulHeuristicResult_t& heur_res) {
    cublasLtMatmulAlgo_t algo = heur_res.algo;
    int algo_id, tile, num_splits_k, reduction_scheme, swizzle, custom_option, stages_id, inner_shape_id, cluster_shape_id;

    CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(int), NULL));
    CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(int), NULL));
    CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &num_splits_k, sizeof(int), NULL));
    CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction_scheme, sizeof(int), NULL));
    CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(int), NULL));
    CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom_option, sizeof(int), NULL));
    CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages_id, sizeof(int), NULL));
  //   CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner_shape_id, sizeof(int), NULL));
  //   CUBLASLT_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster_shape_id, sizeof(int), NULL));

    printf("algo={ Id=%d, workspace_size=%ld, waves_count=%.2f, "
                "<tile_id=%d, split_k=%d, reduction_schema=%d, cta_swizzle=%d, custom_option=%d, stages_id=%d, inner_shape_id=%d, cluster_shape_id=%d> }\n",
                  algo_id, heur_res.workspaceSize, heur_res.wavesCount, 
                  tile, num_splits_k, reduction_scheme, swizzle, custom_option, stages_id, inner_shape_id, cluster_shape_id);
  }
};

}  // namespace xop
