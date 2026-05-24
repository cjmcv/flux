
// #define ENABLE_QWEN3_06B
// #define ENABLE_QWEN3_4B

// #define TL_ENABLE_L2_PREFETCH 1
#include "linear.cuh"
#include "silu_mul.cuh"
#include "rmsnorm.cuh"
#include "gqa_decode.cuh"
#include "rope.cuh"
#include "copy.cuh"
#include "prefetch.cuh"

#include "runtime_header.h"
#include <iostream>
#include <map>

namespace megakernel {
namespace runtime {


struct TmaDescTriple {
  CUtensorMap *A_desc;
  CUtensorMap *B_desc;
  CUtensorMap *C_desc;
  CUtensorMap *R_desc;
};

class TmaDescFactory {
public:
  static TmaDescFactory& instance() {
    static TmaDescFactory factory;
    return factory;
  }

  bool find(int task_type, int variant_id, TmaDescTriple& out_desc) {
    auto key = std::make_pair(task_type, variant_id);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      out_desc = it->second;
      return true;
    }
    return false;
  }

  void insert(int task_type, int variant_id, const TmaDescTriple& desc) {
    auto key = std::make_pair(task_type, variant_id);
    cache_[key] = desc;
  }

private:
  TmaDescFactory() = default;
  std::map<std::pair<int, int>, TmaDescTriple> cache_;
};

__host__ void create_linear_gemm_cutensor(int M, int N, int K,
                                          void* __restrict__ A, void* __restrict__ B, void* __restrict__ R, void* __restrict__ C,
                                          CUtensorMap* out_A_desc, CUtensorMap* out_B_desc, CUtensorMap* out_C_desc, CUtensorMap* out_R_desc, bool to_device) {
  if (M==1) {
    // FUSE_RES == true
    if (N == 1024 && K == 3072) {
      create_linear_gemm_add_tl_1_1024_3072((bfloat16_t*)A, (bfloat16_t*)B, (bfloat16_t*)R, (bfloat16_t*)C, out_A_desc, out_B_desc, out_C_desc, to_device);
    }
    else if (N == 1024 && K == 2048) {
      create_linear_gemm_add_tl_1_1024_2048((bfloat16_t*)A, (bfloat16_t*)B, (bfloat16_t*)R, (bfloat16_t*)C, out_A_desc, out_B_desc, out_C_desc, to_device);
    }
    // FUSE_RES == false
    if (N==6144 && K==1024) {
      create_linear_gemm_tl_1_6144_1024((bfloat16_t*)A, (bfloat16_t*)B, (bfloat16_t*)C, out_A_desc, out_B_desc, out_C_desc, to_device);
    }
    else if (N == 4096 && K == 1024) {
      create_linear_gemm_tl_1_4096_1024((bfloat16_t*)A, (bfloat16_t*)B, (bfloat16_t*)C, out_A_desc, out_B_desc, out_C_desc, to_device);
    }
  }
}

__host__ inline void create_tma_desc_by_task(FullTaskDesc &task_desc) {
  switch (task_desc.task_type) {
    case TASK_LINEAR_HOPPER:
    case TASK_LINEAR_WITH_RESIDUAL_HOPPER: {
        int m = task_desc.inputs[0].dim[0];
        int k = task_desc.inputs[0].dim[1];
        int n = task_desc.inputs[1].dim[0];

        TmaDescTriple desc;
        if (TmaDescFactory::instance().find(task_desc.task_type, task_desc.variant_id, desc)) {
            task_desc.inputs[0].tma_desc_ptrs[0] = desc.A_desc;
            task_desc.inputs[1].tma_desc_ptrs[0] = desc.B_desc;
            task_desc.inputs[2].tma_desc_ptrs[0] = desc.R_desc;
            task_desc.outputs[0].tma_desc_ptrs[0] = desc.C_desc;
        } else {
            cudaMalloc(&desc.A_desc, sizeof(CUtensorMap));
            cudaMalloc(&desc.B_desc, sizeof(CUtensorMap));
            cudaMalloc(&desc.R_desc, sizeof(CUtensorMap));
            cudaMalloc(&desc.C_desc, sizeof(CUtensorMap));
            create_linear_gemm_cutensor(m,n,k, task_desc.inputs[0].base_ptr, task_desc.inputs[1].base_ptr, task_desc.inputs[2].base_ptr, task_desc.outputs[0].base_ptr,
                                        desc.A_desc, desc.B_desc, desc.C_desc, desc.R_desc, true);
            TmaDescFactory::instance().insert(task_desc.task_type, task_desc.variant_id, desc);
            task_desc.inputs[0].tma_desc_ptrs[0] = desc.A_desc;
            task_desc.inputs[1].tma_desc_ptrs[0] = desc.B_desc;
            task_desc.inputs[2].tma_desc_ptrs[0] = desc.R_desc;
            task_desc.outputs[0].tma_desc_ptrs[0] = desc.C_desc;
        }

        // printf("hello:%d, %d, %d, %p, %p, %p, (%lld, %lld, %lld).\n",
        //     m,n,k, task_desc.inputs[0].base_ptr, task_desc.inputs[1].base_ptr, task_desc.outputs[0].base_ptr,
        //     task_desc.inputs[0].tma_desc_ptrs[0].opaque[0], task_desc.inputs[1].tma_desc_ptrs[0].opaque[0], task_desc.outputs[0].tma_desc_ptrs[0].opaque[0]);
      break;
    }
    default:
      printf("create_tma_desc_by_task: %d is not supported.\n", task_desc.task_type);
  }
}

} // runtime
} // megakernel