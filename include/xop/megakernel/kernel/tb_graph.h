/* Copyright 2023-2024 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "megakernel/config.h"
#include "megakernel/kernel/device_tensor.h"
#include <vector>

namespace megakernel {
namespace threadblock {

class TBOperator {
public:
  TBOperator(dim3 grid_dim,
            megakernel::kernel::DTensor const &_dtensor,
            int3 _input_map)
      : dtensor(_dtensor),
    input_map(_input_map)  {
    kernel::DTensor tensor;
    tensor.num_dims = dtensor.num_dims;
    tensor.data_type = dtensor.data_type;
    for (int i = 0; i < tensor.num_dims; i++) {
      tensor.dim[i] = dtensor.dim[i];
    }

    tensor.guid = kernel::DTensor::next_guid++;
  }

  ~TBOperator() {}

public:
  megakernel::kernel::DTensor dtensor;
  int3 input_map;
};

class TBGraph {

public:
  TBGraph()    
    : grid_dim(1, 1, 1), block_dim(1, 1, 1), thread_num(128) {}

  TBGraph(dim3 _grid_dim, dim3 _block_dim, int _thread_num)
    : grid_dim(_grid_dim), block_dim(_block_dim), thread_num(_thread_num) {
  // A bgraph cannot have more than MAX_NUM_THREADBLOCKS_PER_KERNEL threadblocks
  // otherwise we don't have enough buffers in device memory for saving
  // fingerprints
  assert(grid_dim.x * grid_dim.y * grid_dim.z <=
      megakernel::config::MAX_NUM_THREADBLOCKS_PER_KERNEL);
  }

  ~TBGraph() {
    while (!operators.empty()) {
      delete operators.back();
      operators.pop_back();
    }
  }

  TBGraph(TBGraph const &) = delete;
  TBGraph &operator=(TBGraph const &) = delete;
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

////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace megakernel
