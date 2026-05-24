
#pragma once

// #include <torch/all.h>
#include "megakernel/kernel/device_tensor.h"
#include "megakernel/kernel/operator.h"
#include "megakernel/kernel/runtime.h"
#include "megakernel/kernel/tb_graph.h"
#include "megakernel/kernel/task_register.h"

namespace xop {

// // Fake pointer type, must match fptr_t type in ops.h.
// // We use this type alias to indicate when pointers are passed in as int64_t.
// using fptr_t = int64_t;
// static_assert(sizeof(void*) == sizeof(fptr_t));

// fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs, torch::Tensor& rank_data, int64_t rank, bool full_nvlink);
// void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes);
// void dispose(fptr_t _fa);
// int64_t meta_size();
// void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs);
// std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa);
// void register_graph_buffers(fptr_t _fa, const std::vector<std::vector<int64_t>>& handles, const std::vector<std::vector<int64_t>>& offsets);

// void helloABC(int a);

using namespace megakernel;
using namespace megakernel::threadblock;

class AllReduce {

public:
  AllReduce(int thread_num)    
    : grid_dim(1, 1, 1), block_dim(1, 1, 1), thread_num(128) {printf("hello AllReduce (%d).\n", thread_num);}

  AllReduce(dim3 _grid_dim, dim3 _block_dim, int _thread_num)
    : grid_dim(_grid_dim), block_dim(_block_dim), thread_num(_thread_num) {
  // A bgraph cannot have more than MAX_NUM_THREADBLOCKS_PER_KERNEL threadblocks
  // otherwise we don't have enough buffers in device memory for saving
  // fingerprints
  assert(grid_dim.x * grid_dim.y * grid_dim.z <=
      megakernel::config::MAX_NUM_THREADBLOCKS_PER_KERNEL);
  }

  ~AllReduce() {}

  AllReduce(AllReduce const &) = delete;
  AllReduce &operator=(AllReduce const &) = delete;
  // input operator
      
  kernel::DTensor *new_input(megakernel::kernel::DTensor const *dtensor, int3 input_map){
    TBOperator *op = new TBOperator(grid_dim, *dtensor, input_map);
    assert(op != nullptr);
    operators.push_back(op);
    return &op->dtensor;
  }
  
public:
  dim3 grid_dim, block_dim;
  int thread_num;
  std::vector<megakernel::threadblock::TBOperator *> operators;
};

}  // namespace xop
