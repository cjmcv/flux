/* Copyright 2025 CMU
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
#include <cuda_runtime.h>
#ifdef MPK_ENABLE_TMA
#include <cuda.h>
#endif

namespace megakernel {
namespace runtime {

#if defined(MEGAKERNEL_GRACE_HOPPER) || defined(MEGAKERNEL_GRACE_BLACKWELL)
constexpr int WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE = 6 * 1024;
#else
constexpr int WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE = 3 * 1024;
#endif

#if MPK_TARGET_CC == 120
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    72 * 1024;    // RTX5090 最多98*1024？超了kernel会直接跳空
#elif MPK_TARGET_CC >= 90
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    227 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#elif MPK_TARGET_CC >= 86
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    99 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#elif MPK_TARGET_CC >= 80
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    163 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#else
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    163 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#endif

typedef unsigned long long int TaskId;
unsigned long long int const TASK_INVALID_ID = 0x7fffffffffffffff;
// Task IDs are 64-bit values encoding both the current iteration of the task
// and its index TASK: iteration id: 32, task index: 32
typedef unsigned long long int EventId;
// Event IDs are 64-bit values encoding both the owner of the event and its
// index EVENT: nvshmem_tag: 16, owner_node: 16, event_idx: 32
unsigned long long int const EVENT_NVSHMEM_TAG = 0x1e00000000000000;
unsigned long long int const EVENT_INVALID_ID = 0x7ffffffffffffffe;
typedef unsigned long long int EventCounter;

int const MAX_INPUTS_PER_TASK = 7;
int const MAX_OUTPUTS_PER_TASK = 3;
int const MAX_NUM_WORKERS = 128;

enum TaskType {
  TASK_TERMINATE = 0,
  TASK_BEGIN_TASK_GRAPH = 10,
  // compute task starts from 100
  TASK_RMS_NORM_LINEAR = 102,
  TASK_SILU_MUL_LINEAR_WITH_RESIDUAL = 105,
  TASK_ALLREDUCE = 106,
  TASK_REDUCE = 107,
  TASK_LINEAR_WITH_RESIDUAL = 108,
  TASK_SILU_MUL = 118,
  TASK_RMS_NORM = 119,
  TASK_LINEAR = 120,
  TASK_SILU_MUL_LINEAR = 122,
  TASK_GQA_DECODE = 123,
  TASK_ROPE = 124,
  TASK_NVSHMEM_COPY = 199,
  // hopper
  TASK_HOPPER_TASK_BEGIN = 150,
  TASK_LINEAR_WITH_RESIDUAL_HOPPER = 151,
  TASK_LINEAR_HOPPER = 152,
  TASK_HOPPER_TASK_END = 160,
};

enum EventType {
  EVENT_EMPTY = 900,
  EVENT_LAUNCH_TASKS = 901,
  EVENT_LAUNCH_MASSIVE_TASKS = 902,
  EVENT_LAUNCH_DEPENDENT_TASKS = 903,
  EVENT_END_OF_TASK_GRAPH = 910,
  EVENT_TERMINATION = 911,
  EVENT_INVALID = 999,
};

struct TensorDesc {
  int num_dims;
  int bx; // CJM
  int by;
  int bz;
  void *base_ptr;
#ifdef MPK_ENABLE_TMA
  CUtensorMap *tma_desc_ptrs[megakernel::config::MAX_TMA_DESC_PER_TENSOR];
#endif
  int data_type;
  int dim[megakernel::config::MAX_TENSOR_DIMS];
  int stride[megakernel::config::MAX_TENSOR_DIMS];
};

struct EventDesc {
  EventDesc(void)
      : event_type(EVENT_INVALID), num_triggers(0),
        first_task_id(TASK_INVALID_ID), last_task_id(TASK_INVALID_ID) {}
  EventDesc(EventType type, int nt, TaskId f, TaskId l)
      : event_type(type), num_triggers(nt), first_task_id(f), last_task_id(l) {}
  EventType event_type;
  int num_triggers;
  TaskId first_task_id, last_task_id;
};

struct FullTaskDesc {
  FullTaskDesc(TaskType t, int _variant_id)
      : task_type(t), variant_id(_variant_id), num_inputs(0), num_outputs(0),
        trigger_event(EVENT_INVALID_ID), dependent_event(EVENT_INVALID_ID) {}
  FullTaskDesc() {}
  TaskType task_type;
  unsigned variant_id;
  int num_inputs, num_outputs;
  EventId trigger_event;
  EventId dependent_event;
  TensorDesc inputs[MAX_INPUTS_PER_TASK];
  TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
  union TaskMetadata {
    struct {
      int expert_offset; // Used for MoE
    };
    struct {
      size_t xfer_size_in_bytes; // Used for nvshmem
    };
  } task_metadata;
};

struct alignas(16) TaskDesc {
  TaskDesc(FullTaskDesc t)
      : task_type(t.task_type), variant_id(t.variant_id),
        trigger_event(t.trigger_event), dependent_event(t.dependent_event),
        task_metadata(t.task_metadata), post_task(nullptr) {
    bx = t.inputs[0].bx; // CJM_TODO 只用到了一个, 冗余
    by = t.inputs[0].by;
    bz = t.inputs[0].bz;
    for (int i = 0; i < t.num_inputs; i++) {
      input_ptrs[i] = t.inputs[i].base_ptr;
    }
    for (int i = 0; i < t.num_outputs; i++) {
      output_ptrs[i] = t.outputs[i].base_ptr;
    }
#ifdef MPK_ENABLE_TMA
    for (int i = 0; i < t.num_inputs; i++) {
      for (int k = 0; k < megakernel::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        input_tma_desc_ptrs[i][k] = t.inputs[i].tma_desc_ptrs[k];
      }
    }
    for (int i = 0; i < t.num_outputs; i++) {
      for (int k = 0; k < megakernel::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        output_tma_desc_ptrs[i][k] = t.outputs[i].tma_desc_ptrs[k];
      }
    }
#endif
  }
  TaskDesc() {}
  TaskType task_type;
  unsigned variant_id;
  EventId trigger_event;
  EventId dependent_event;
  int bx;
  int by;
  int bz;
  void *input_ptrs[MAX_INPUTS_PER_TASK];
  void *output_ptrs[MAX_OUTPUTS_PER_TASK];
  TaskDesc *post_task;
#ifdef MPK_ENABLE_TMA
  CUtensorMap *input_tma_desc_ptrs[MAX_INPUTS_PER_TASK][megakernel::config::MAX_TMA_DESC_PER_TENSOR];
  CUtensorMap *output_tma_desc_ptrs[MAX_OUTPUTS_PER_TASK][megakernel::config::MAX_TMA_DESC_PER_TENSOR];
#endif
  FullTaskDesc::TaskMetadata task_metadata;
};

struct RuntimeConfig {
  int num_workers, num_local_schedulers, num_remote_schedulers, num_graphs;
  int num_gpus, my_gpu_id;
  int num_tasks;
  int num_events;
  EventCounter *all_event_counters;
  int *all_event_num_triggers;
  int **static_worker_tasks_index;
  TaskDesc *all_tasks;
  EventDesc *all_events;
  TaskId *first_tasks;
  int batch_size;
  int *step;                    // Metadata for LLM serving
  int *kcache;                 // Metadata for LLM serving
  int *vcache;                 // Metadata for LLM serving
  int *kcache_curstep;         // Metadata for LLM serving
  int *vcache_curstep;         // Metadata for LLM serving
  int *onestep_size;         // Metadata for LLM serving
  int *onelayer_size;           // Metadata for LLM serving
  int layer_id;
  int *infer_cnt; 

  void *profiler_buffer;
  bool is_static_schedule;
};

} // namespace runtime
} // namespace megakernel
