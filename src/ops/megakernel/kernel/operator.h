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

#include "megakernel/kernel/device_tensor.h"
#include "megakernel/kernel/tb_graph.h"
#include <vector>

namespace megakernel {
namespace kernel {


class Graph;

class KNOperator {
public:
  KNOperator(megakernel::type::KNOperatorType _type): op_type(_type) {}
  virtual ~KNOperator() {}

public:
  megakernel::type::KNOperatorType op_type;
  megakernel::kernel::DTensor dtensor;
};

class KNInputOp : public KNOperator {
public:
  KNInputOp(std::vector<int> const &dims,
            std::vector<size_t> const &strides,
            megakernel::type::DataType data_type)
      : KNOperator(megakernel::type::KN_INPUT_OP), input_strides(strides) {
    assert(dims.size() == strides.size());

    dtensor.num_dims = dims.size();
    for (int i = dtensor.num_dims - 1; i >= 0; i--) {
      dtensor.dim[i] = dims[i];
    }
    dtensor.data_type = data_type;
    dtensor.owner_op = this;
    dtensor.guid = DTensor::next_guid++;
  }
  ~KNInputOp() {}

public:
  std::vector<size_t> input_strides;
};

class KNCustomizedOp : public megakernel::kernel::KNOperator {
public:
  KNCustomizedOp(Graph *_kgraph,
                 std::vector<DTensor> const &_inputs,
                 megakernel::threadblock::TBGraph const &_graph)
                 : KNOperator(megakernel::type::KN_CUSTOMIZED_OP),
      bgraph(_graph.grid_dim,
             _graph.block_dim,
             _graph.thread_num) {
    size_t input_idx = 0;
    for (auto const &op : _graph.operators) {
      megakernel::threadblock::TBOperator *input_op =
          static_cast<megakernel::threadblock::TBOperator *>(op);
      bgraph.new_input(&_inputs[input_idx++],
                      input_op->input_map);
    }
  }
  virtual ~KNCustomizedOp() {}

public:
  megakernel::threadblock::TBGraph bgraph;
};

} // namespace kernel
} // namespace megakernel
