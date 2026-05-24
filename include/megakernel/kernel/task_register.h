/* Copyright 2023-2025 CMU
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

#include "megakernel/persistent_kernel/runtime_header.h"
#include "megakernel/kernel/tb_graph.h"
#include "megakernel/kernel/operator.h"
#include "megakernel/code_keeper.h"

namespace megakernel {
namespace runtime {

namespace kn = megakernel::kernel;
namespace tb = megakernel::threadblock;

class TaskRegister {
public:
  TaskRegister() {}

public:
  static TaskRegister *get_instance() {
    static TaskRegister singleton;
    return &singleton;
  }

  int get_fused_start_id(std::vector<int> const &params) {
    int size = params.size();
    for (int i=0; i<size; i++) {
      if (params[i] == 99) {
        return i+1;
      }
    }
    return -1;
  }

  void append_fused_func(tb::TBGraph const &bgraph, std::vector<int> const &params, int fused_params_start_id, megakernel::transpiler::CodeKeeper &code, int type=0, int M=0, int N=0, std::string input_str = "") {
    if (fused_params_start_id == -1) { return ; }

    int extra_func_id = params[fused_params_start_id];
    int extra_bx = params[fused_params_start_id+1];
    int extra_by = params[fused_params_start_id+2]; // 0
    int extra_bz = params[fused_params_start_id+3]; // 0
    // int layer_id = params[fused_params_start_id+4];

    code.inc_indent();
    code.e("  if (task_desc->bx >= $ && task_desc->by == 0 && task_desc->bz == 0) {", 
      bgraph.grid_dim.x-extra_bx);
    // extra_func_id == 0: 对应无参数，只切换逻辑
    if (extra_func_id == 1) {
      // Update kv result to kvcache
      code.e("  kernel::copy_kernel<bfloat16_t, $>(", bgraph.thread_num);
      code.e("    task_desc->bx-$, task_desc->by, task_desc->bz,", bgraph.grid_dim.x-extra_bx);
      code.e("    runtime_config.layer_id,");
      code.e("    *runtime_config.onelayer_size,");
      code.e("    *runtime_config.step,");
      code.e("    *runtime_config.onestep_size,");
      code.e("    runtime_config.kcache_curstep,");
      code.e("    runtime_config.vcache_curstep,");
      code.e("    runtime_config.kcache,");
      code.e("    runtime_config.vcache);");
    }
    else if (extra_func_id == 10) {
      // 权重预加载
      code.e("  kernel::prefetch_kernel<bfloat16_t, $, $, $, $>(", bgraph.thread_num, type, M, N);
      code.e("    task_desc->bx-$, task_desc->by, task_desc->bz,", bgraph.grid_dim.x-extra_bx);
      code.e("    runtime_config.layer_id,");
      code.e("    $);", input_str.c_str());
    }
    else if (extra_func_id == 11) {
      bool with_residual = false;
      code.e("TaskDesc* post_task_desc = task_desc->post_task;");

      // code.e("kernel::debug_kernel<bfloat16_t, $>(", bgraph.thread_num);
      // code.e("    post_task_desc);");

      code.e("if (post_task_desc != nullptr) {");
      code.e("kernel::prefetch_kernel<bfloat16_t, $, $, $, $>(", bgraph.thread_num, type, M, N);
      code.e("    post_task_desc->bx, post_task_desc->by, post_task_desc->bz,");
      code.e("    static_smem,"); // static_smem
      code.e("    post_task_desc->input_tma_desc_ptrs[0][0],");
      code.e("    post_task_desc->input_tma_desc_ptrs[1][0],");
      if (with_residual) {
        code.e("    post_task_desc->input_ptrs[2],"); // task_desc->input_tma_desc_ptrs[2][0]
      } else {
        code.e("    nullptr,");
      }
      code.e("    post_task_desc->output_tma_desc_ptrs[0][0],");
      code.e("    runtime_config.batch_size,");
      if (with_residual) {
        code.e("    runtime_config.my_gpu_id == 0);");
      } else {
        code.e("    false/*residual*/);}");
      }
    }
    code.e("  }");
  }

  int register_rope_task(tb::TBGraph const &bgraph, std::vector<int> const &params) {
    int fused_params_start_id = get_fused_start_id(params);

    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = 4;
    int num_outputs = 2;
    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }

    assert(output_ops[0]->dtensor.num_dims == 4);
    int batch_size = output_ops[0]->dtensor.dim[0];
    int seqlen = output_ops[0]->dtensor.dim[1];
    int heads = output_ops[0]->dtensor.dim[2];
    int dim = output_ops[0]->dtensor.dim[3];
    int groups = output_ops[1]->dtensor.dim[2];

    assert(batch_size == 1);
    // assert(input_ops[0]->dtensor.num_dims == 2);
    // assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
    // assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    code.e("kernel::rope_kernel<bfloat16_t, $, $, $, $, $, $>(",
      bgraph.thread_num, batch_size, seqlen, heads, groups, dim);
    code.e("    task_desc->bx, task_desc->by, task_desc->bz,");
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->input_ptrs[1],");
    code.e("    task_desc->input_ptrs[2],");
    code.e("    task_desc->input_ptrs[3],");
    code.e("    task_desc->output_ptrs[0],");
    if (fused_params_start_id == -1) {
      code.e("    task_desc->output_ptrs[1]);");
    }
    else {
      // rope的输出是 q_out 和 k_out, 其中q_out直接作为attn的输入，而k_out则需要被更新到kvcache里。
      // 这个额外fuse的操作作用：
      //   1. 修改输出指针，将k_out的内存直接指向kcache的具体内存里，即计算完等同于更新完kcache
      //   2. append_fused_func，添加拷贝函数，将之前计算得到的v也一并更新到vcache中，该操作可以与rope完全并行
      int extra_func_id = params[fused_params_start_id];
      int extra_bx = params[fused_params_start_id+1];
      int extra_by = params[fused_params_start_id+2]; // 0
      int extra_bz = params[fused_params_start_id+3]; // 0
      // int layer_id = params[fused_params_start_id+4];
      code.e("    ((bfloat16_t*)runtime_config.kcache) + runtime_config.layer_id * (*runtime_config.onelayer_size) + (*runtime_config.step) * (*runtime_config.onestep_size));");
      append_fused_func(bgraph, params, fused_params_start_id, code);
    }
    return register_task_variant(TASK_ROPE, code.to_string());
  }

  int register_gqa_decode_task(tb::TBGraph const &bgraph, std::vector<int> const &params) {
    int fused_params_start_id = get_fused_start_id(params);

    int sub_kernel_id = params[0];
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = 5;
    int num_outputs = 3;
    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }

    assert(output_ops[0]->dtensor.num_dims == 3);
    int batch_size = output_ops[0]->dtensor.dim[0];
    int head = output_ops[0]->dtensor.dim[1];
    int dim = output_ops[0]->dtensor.dim[2];

    int num_kv_heads_idx = 2; 
    if (fused_params_start_id != -1) {
      num_kv_heads_idx = 3; // 4d kvcache [layer, batch, seqlen, num_kv_heads, dim_per_head]
    }
    

    assert(batch_size == 1);
    // assert(input_ops[0]->dtensor.num_dims == 2);
    // assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
    // assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    if (fused_params_start_id == -1) {
      int num_kv_heads = input_ops[1]->dtensor.dim[2]; // 4d kvcache [batch, seqlen, num_kv_heads, dim_per_head]
      code.e("kernel::gqa_decode_kernel<bfloat16_t, $, $, $, $, $, $>(",
        bgraph.thread_num, sub_kernel_id, batch_size, head, num_kv_heads, dim);
      code.e("    task_desc->bx, task_desc->by, task_desc->bz,");
      code.e("    task_desc->input_ptrs[0],");
      code.e("    task_desc->input_ptrs[1],");
      code.e("    task_desc->input_ptrs[2],");
      code.e("    task_desc->input_ptrs[3],");
      code.e("    runtime_config.step, // task_desc->input_ptrs[4],");
      code.e("    task_desc->output_ptrs[0],");
      code.e("    task_desc->output_ptrs[1],");
      code.e("    task_desc->output_ptrs[2]);");
    }
    else {
      // 直接输入完整的kvcache池，并根据step，从中选取数据
      // 5d kvcache [layer, batch, seqlen, num_kv_heads, dim_per_head]
      int num_kv_heads = input_ops[1]->dtensor.dim[3];

      int extra_func_id = params[fused_params_start_id];
      // int layer_id = params[fused_params_start_id+1];
      code.e("kernel::gqa_decode_kernel<bfloat16_t, $, $, $, $, $, $>(",
        bgraph.thread_num, sub_kernel_id, batch_size, head, num_kv_heads, dim);
      code.e("    task_desc->bx, task_desc->by, task_desc->bz,");
      code.e("    task_desc->input_ptrs[0],");
      code.e("    ((bfloat16_t*)task_desc->input_ptrs[1]) + runtime_config.layer_id * (*runtime_config.onelayer_size), // task_desc->input_ptrs[1]");
      code.e("    ((bfloat16_t*)task_desc->input_ptrs[2]) + runtime_config.layer_id * (*runtime_config.onelayer_size), // task_desc->input_ptrs[2]");
      code.e("    task_desc->input_ptrs[3],");
      code.e("    runtime_config.step, // task_desc->input_ptrs[4],");
      code.e("    task_desc->output_ptrs[0],");
      code.e("    task_desc->output_ptrs[1],");
      code.e("    task_desc->output_ptrs[2]);");
    }
    
    return register_task_variant(TASK_GQA_DECODE, code.to_string());
  }

  int register_rmsnorm_task(tb::TBGraph const &bgraph, std::vector<int> const &params, size_t *num_inputs, size_t *num_outputs) {
    int fused_params_start_id = -1;
    int extra_tensors = 0;
    if (params.size() != 0) {
      fused_params_start_id = get_fused_start_id(params);
      int extra_func_id = params[fused_params_start_id];
      if (extra_func_id == 10) { // 10: 表示扩展的功能是权重预加载
        extra_tensors = 1;
      }      
    }

    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    *num_inputs = 2;
    *num_outputs = 1 + extra_tensors;
    // assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs + extra_tensors);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < *num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }
    assert(output_ops[0]->dtensor.num_dims == 2);
    int batch_size = output_ops[0]->dtensor.dim[0];
    int hidden_dim = output_ops[0]->dtensor.dim[1];
    // Currently assume that each rmsnorm task processes one token
    assert(batch_size == 1);
    assert(input_ops[0]->dtensor.num_dims == 2);
    assert(output_ops[0]->dtensor.dim[0] == input_ops[0]->dtensor.dim[0]);
    assert(output_ops[0]->dtensor.dim[1] == input_ops[0]->dtensor.dim[1]);
    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    code.e("kernel::rms_norm_kernel<bfloat16_t, $, $, $, $, $, $>(", 
      bgraph.thread_num, bgraph.block_dim.x, bgraph.block_dim.y, bgraph.block_dim.z, 
      batch_size, hidden_dim);
    code.e("    task_desc->bx, task_desc->by, task_desc->bz,");
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->input_ptrs[1],");
    code.e("    task_desc->output_ptrs[0],");
    code.e("    1e-12f);");
    if (fused_params_start_id != -1) {
      append_fused_func(bgraph, params, fused_params_start_id, code, TASK_RMS_NORM, batch_size, hidden_dim, "task_desc->output_ptrs[1]");
    }
    return register_task_variant(TASK_RMS_NORM, code.to_string());
  }

  int register_rmsnorm_linear_task(tb::TBGraph const &bgraph, std::vector<int> const &params) {
    assert(params.size() == 0);
    int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = 3;
    int num_outputs = 1;

    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }
    assert(output_ops[0]->dtensor.num_dims == 2);
    batch_size = output_ops[0]->dtensor.dim[0];
    output_size = output_ops[0]->dtensor.dim[1];
    assert(input_ops[0]->dtensor.num_dims == 2);
    reduction_size = input_ops[0]->dtensor.dim[1];
    // get output stride
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    output_stride = static_cast<int>(kn_input_op->input_strides[0]);

    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    code.e("kernel::norm_linear_task_impl<bfloat16_t, $, $, $, $>(",
          batch_size,
          output_size,
          reduction_size,
          output_stride);
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->input_ptrs[1],");
    code.e("    task_desc->input_ptrs[2],");
    code.e("    runtime_config.batch_size,");
    code.e("    1e-6f,");
    code.e("    task_desc->output_ptrs[0]);");
    return register_task_variant(TASK_RMS_NORM_LINEAR, code.to_string());
  }

  int register_linear_task(tb::TBGraph const &bgraph, std::vector<int> const &params, bool with_residual, bool with_silu_mul) {
    assert(params.size() == 0);
    int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = with_residual ? 3 : 2;
    int num_outputs = 1;

    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }
    assert(output_ops[0]->dtensor.num_dims == 2);
    batch_size = output_ops[0]->dtensor.dim[0];
    output_size = output_ops[0]->dtensor.dim[1];
    assert(input_ops[0]->dtensor.num_dims == 2);
    reduction_size = input_ops[0]->dtensor.dim[1];
    // get output stride
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    output_stride = static_cast<int>(kn_input_op->input_strides[0]);

    if (with_silu_mul) {
      reduction_size /= 2;
    }
    megakernel::transpiler::CodeKeeper code;
    code.e("kernel::linear_kernel<bfloat16_t, $, $, $, $, $, $, $, $, $, $, $>(",
          bgraph.thread_num, bgraph.block_dim.x, bgraph.block_dim.y, bgraph.block_dim.z, 
          batch_size,
          output_size,
          reduction_size,
          output_stride,
          3,
          with_residual,
          with_silu_mul);

    code.e("    task_desc->bx, task_desc->by, task_desc->bz,");
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->input_ptrs[1],");
    if (with_residual) {
      code.e("    task_desc->input_ptrs[2],");
    } else {
      code.e("    nullptr,");
    }
    code.e("    task_desc->output_ptrs[0],");
    code.e("    runtime_config.batch_size,");
    if (with_residual) {
      code.e("    runtime_config.my_gpu_id == 0);");
    } else {
      code.e("    false/*residual*/);");
    }
    
    if (with_residual) {
      return register_task_variant(TASK_LINEAR_WITH_RESIDUAL, code.to_string());
    } else {
      return register_task_variant(TASK_LINEAR, code.to_string());
    }
  }
  
  int register_linear_hopper_task(tb::TBGraph const &bgraph, std::vector<int> const &params, bool with_residual, bool with_silu_mul) {
    assert(params.size() == 0);
    int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = with_residual ? 3 : 2;
    int num_outputs = 1;

    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }
    assert(output_ops[0]->dtensor.num_dims == 2);
    batch_size = output_ops[0]->dtensor.dim[0];
    output_size = output_ops[0]->dtensor.dim[1];
    assert(input_ops[0]->dtensor.num_dims == 2);
    reduction_size = input_ops[0]->dtensor.dim[1];
    // get output stride
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    output_stride = static_cast<int>(kn_input_op->input_strides[0]);

    if (with_silu_mul) {
      reduction_size /= 2;
    }
    megakernel::transpiler::CodeKeeper code;
    code.e("kernel::linear_kernel<bfloat16_t, $, $, $, $, $, $, $, $, $, $, $>(",
          bgraph.thread_num, bgraph.block_dim.x, bgraph.block_dim.y, bgraph.block_dim.z, 
          batch_size,
          output_size,
          reduction_size,
          output_stride,
          3,
          with_residual,
          with_silu_mul);

    code.e("    task_desc->bx, task_desc->by, task_desc->bz,");
    code.e("    static_smem,");
    code.e("    task_desc->input_tma_desc_ptrs[0][0],");
    code.e("    task_desc->input_tma_desc_ptrs[1][0],");
    if (with_residual) {
      code.e("    task_desc->input_ptrs[2],"); // task_desc->input_tma_desc_ptrs[2][0]
    } else {
      code.e("    nullptr,");
    }
    code.e("    task_desc->output_tma_desc_ptrs[0][0],");
    code.e("    runtime_config.batch_size,");
    if (with_residual) {
      code.e("    runtime_config.my_gpu_id == 0);");
    } else {
      code.e("    false/*residual*/);");
    }
    
    if (with_residual) {
      return register_task_variant(TASK_LINEAR_WITH_RESIDUAL_HOPPER, code.to_string());
    } else {
      return register_task_variant(TASK_LINEAR_HOPPER, code.to_string());
    }
  }

  int register_silu_mul_task(tb::TBGraph const &bgraph, std::vector<int> const &params, size_t *num_inputs, size_t *num_outputs){
    int fused_params_start_id = -1;
    int extra_tensors = 0;
    if (params.size() != 0) {
      fused_params_start_id = get_fused_start_id(params);
      int extra_func_id = params[fused_params_start_id];
      if (extra_func_id == 10) { // 10: 表示扩展的功能是权重预加载
        extra_tensors = 1;
      }
    }
    
    int batch_size = 0, output_size = 0, input_stride, output_stride;
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    *num_inputs = 1;
    *num_outputs = 1 + extra_tensors;
    // assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < *num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }
    assert(output_ops[0]->dtensor.num_dims == 2);
    batch_size = output_ops[0]->dtensor.dim[0];
    output_size = output_ops[0]->dtensor.dim[1];
    assert(input_ops[0]->dtensor.num_dims == 2);
    assert(input_ops[0]->dtensor.dim[1] == output_size * 2);
    // get input stride
    assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
    input_stride = input_ops[0]->dtensor.dim[1];
    assert(input_stride == static_cast<int>(kn_input_op->input_strides[0]));
    // get output stride
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    output_stride = static_cast<int>(kn_input_op->input_strides[0]);
    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    code.e("kernel::silu_mul_kernel<bfloat16_t, $, $, $, $, $, $, $, $>(",
          bgraph.thread_num, bgraph.block_dim.x, bgraph.block_dim.y, bgraph.block_dim.z, 
          batch_size, output_size,
          input_stride, output_stride);
    code.e("    task_desc->bx, task_desc->by, task_desc->bz,"); // CJM
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->output_ptrs[0],");
    code.e("    runtime_config.batch_size);");
    if (fused_params_start_id != -1) {
      append_fused_func(bgraph, params, fused_params_start_id, code, TASK_SILU_MUL, batch_size, output_size, "task_desc->output_ptrs[1]");
    }
    return register_task_variant(TASK_SILU_MUL, code.to_string());
  }


  int register_silu_mul_linear_with_residual_task(tb::TBGraph const &bgraph, std::vector<int> const &params){
    assert(params.size() == 0);
    int batch_size = 0, output_size = 0, reduction_size = 0, output_stride = 0;
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = 3;
    int num_outputs = 1;

    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }
    assert(output_ops[0]->dtensor.num_dims == 2);
    batch_size = output_ops[0]->dtensor.dim[0];
    output_size = output_ops[0]->dtensor.dim[1];
    assert(input_ops[0]->dtensor.num_dims == 2);
    reduction_size = input_ops[0]->dtensor.dim[1] / 2;
    // get output stride
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    output_stride = static_cast<int>(kn_input_op->input_strides[0]);

    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    code.e("kernel::silu_mul_linear_task_impl<bfloat16_t, $, $, $, $>(",
          batch_size,
          output_size,
          reduction_size,
          output_stride);
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->input_ptrs[1],");
    code.e("    task_desc->input_ptrs[2],");
    code.e("    task_desc->output_ptrs[0],");
    code.e("    runtime_config.my_gpu_id == 0);");
    return register_task_variant(TASK_SILU_MUL_LINEAR_WITH_RESIDUAL,
                                code.to_string());
  }
  
  int register_reduce_task(tb::TBGraph const &bgraph, std::vector<int> const &params){
    // Currently, allreduce task is split to two sub-tasks: allgather + reduce
    // params[0]: num_gpus
    // params[1]: my_gpu_id
    assert(params.size() == 2);
    std::vector<tb::TBOperator *> input_ops;
    std::vector<tb::TBOperator *> output_ops;
    int num_inputs = 2;
    int num_outputs = 1;

    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBOperator *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBOperator *>(op));
      }
    }
    // For now, the memory partition of the input[0] results in a strided
    // 2D tensor, which cannot be directly transferred by a single nvshmem
    // memput. So we use for loop to iterate over the first dim and transfer each
    // row. If the upperlayer changes this layout, this "for-loop" method can
    // fail. So we assert it here just in case.
    assert(input_ops[0]->input_map.x == 1 && input_ops[0]->input_map.y == -1 &&
          input_ops[0]->input_map.z == -1);
    // Currently support 2D reduction, buffer has an extra world_size dim
    assert(input_ops[0]->dtensor.num_dims == 2);
    assert(input_ops[1]->dtensor.num_dims == 3);
    assert(output_ops[0]->dtensor.num_dims == 2);
    int batch_size = input_ops[0]->dtensor.dim[0];
    int output_size = input_ops[0]->dtensor.dim[1];
    // get output stride
    assert(input_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
    kn::KNInputOp *kn_input_op =
        static_cast<kn::KNInputOp *>(input_ops[0]->dtensor.owner_op);
    int input_stride = static_cast<int>(kn_input_op->input_strides[0]);
    kn_input_op = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
    int output_stride = static_cast<int>(kn_input_op->input_strides[0]);
    assert(input_stride == output_stride);
    // Register nvshmem copy task (allgather)
    megakernel::transpiler::CodeKeeper c;
    c.inc_indent();
    c.e("size_t event_index = "
        "get_event_position_index(task_desc->trigger_event);");
    c.inc_indent();
    c.e("int gpu_id = static_cast<int>(get_event_gpu_id(task_desc->trigger_event));");
    c.e("assert(gpu_id < runtime_config.num_gpus);");
    c.e("assert(gpu_id != runtime_config.my_gpu_id);");
    c.e("for (int i = 0; i < $; i++) {", batch_size);
    c.e("  nvshmemx_putmem_signal_block(");
    c.e("      reinterpret_cast<char*>(task_desc->output_ptrs[0]) + i * $ * "
        "sizeof(bfloat16),",
        input_stride);
    c.e("      reinterpret_cast<char*>(task_desc->input_ptrs[0]) + i * $ * "
        "sizeof(bfloat16),",
        output_stride);
    c.e("      task_desc->xfer_size_in_bytes / $,", batch_size);
    c.e("      reinterpret_cast<uint64_t "
        "*>(&runtime_config.all_event_counters[event_index]),");
    c.e("      1 /*signal*/,");
    c.e("      NVSHMEM_SIGNAL_ADD,");
    c.e("      gpu_id);");
    c.e("}");
    register_task_variant(TASK_NVSHMEM_COPY, c.to_string());
    // Register reduction kernel
    megakernel::transpiler::CodeKeeper code;
    code.inc_indent();
    code.e("kernel::reduction_kernel<bfloat16_t, $, $, $, $, $>(",
          params[0],
          params[1],
          batch_size,
          output_size,
          output_stride);
    code.e("    task_desc->input_ptrs[0],");
    code.e("    task_desc->input_ptrs[1],");
    code.e("    task_desc->output_ptrs[0]);");
    return register_task_variant(TASK_REDUCE, code.to_string());
  }
  
  int register_task_variant(TaskType type, std::string const &code) {
    std::vector<std::string> &variants = all_task_variants[type];
    for (size_t i = 0; i < variants.size(); i++) {
      if (variants[i] == code) {
        return (int)(i);
      }
    }
    // Add a new variant
    variants.push_back(code);
    return (int)(variants.size() - 1);
  }

public:
  std::map<TaskType, std::vector<std::string>> all_task_variants;
};

} // namespace runtime
} // namespace megakernel
