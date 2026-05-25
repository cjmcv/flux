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


#include "profiler.h"

#include "mpk_atoms.cuh"
#include "runtime_header.h"
#ifdef USE_NVSHMEM
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#endif
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(MEGAKERNEL_GRACE_HOPPER)
#include "tasks/sm_90/task_header.cuh"
#elif defined(MEGAKERNEL_GRACE_BLACKWELL)
#include "tasks/sm_120/task_header.cuh"
#else
#include "tasks/sm_89/task_header.cuh"
#endif

#define LIKELY(x)       __builtin_expect(!!(x), 1)
#define UNLIKELY(x)     __builtin_expect(!!(x), 0)

using namespace megakernel::runtime;
using namespace kernel;

#if defined(MEGAKERNEL_GRACE_HOPPER)
#define WORKER_NUM_THREADS 256
#elif defined(MEGAKERNEL_GRACE_BLACKWELL)
#define WORKER_NUM_THREADS 256
#else
#define WORKER_NUM_THREADS 128
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr,                                                          \
              "CUDA error at %s:%d: %s\n",                                     \
              __FILE__,                                                        \
              __LINE__,                                                        \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr,                                                          \
              "CUDA error at %s:%d: %s\n",                                     \
              __FILE__,                                                        \
              __LINE__,                                                        \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif

__device__ __forceinline__ void
    _execute_task(TaskDesc const *task_desc,
                  RuntimeConfig const &runtime_config, uint64_t* static_smem);

__device__ __forceinline__ size_t get_event_gpu_id(EventId event_id) {
  return ((event_id >> 32) & 0xffff);
}

__device__ __forceinline__ size_t get_event_position_index(EventId event_id) {
  return (event_id & 0xffffffff);
}

__global__ __launch_bounds__(WORKER_NUM_THREADS, 1) 
void static_persistent_kernel(RuntimeConfig config) {
  #ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0, 1, (threadIdx.x % WORKER_NUM_THREADS == 0));
  #endif

  __shared__ uint64_t mbarrier_mem[12];

  // int step = *config.step;
  const int worker_id = blockIdx.x;
  int task_num = config.static_worker_tasks_index[worker_id][0];
  int *task_ids = &config.static_worker_tasks_index[worker_id][1];

  for (int i = 0; i < task_num ; i++) {
    int task_idx = task_ids[i]; // worker_id * 9 + i;
    TaskDesc *task_desc = &config.all_tasks[task_idx];
    const EventId dependent_event_id = task_desc->dependent_event;
    const EventId trigger_event_id = task_desc->trigger_event;

    // if ((task_desc->task_type != TASK_LINEAR_HOPPER && threadIdx.x == 0) || 
    //    (task_desc->task_type == TASK_LINEAR_HOPPER && threadIdx.x == 128) ) {
    if (dependent_event_id != EVENT_INVALID_ID) {
      if (threadIdx.x == 0) {
        // printf("%d.", dependent_event_id);
        // Wait until the event has been triggered enough times
        #ifndef NDEBUG
        assert(get_event_gpu_id(dependent_event_id) == config.my_gpu_id);
        #endif
        size_t event_index = get_event_position_index(dependent_event_id);
        
        EventCounter needed_counts = static_cast<EventCounter>(config.all_event_num_triggers[event_index]);
        EventCounter actual_counts = 0;
        // 等待前置任务的 Event 计数达到预期值
        while (actual_counts < needed_counts) {
          actual_counts = ld_acquire_sys_u64(&config.all_event_counters[event_index]);
          // printf("dep(%d):(%d vs %d), ", event_index, actual_counts, needed_counts);
          __nanosleep(2);
        }
      }
      // 其他线程需要等到tid0拿到标志后才能往下执行。否则没拿到标志，即前置任务没算完，其他线程就抢跑了
      __syncthreads();       
    }

  #ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_START(task_desc->task_type, task_idx);
    }
  #endif
    _execute_task(task_desc, config, mbarrier_mem); 
  #ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_END(task_desc->task_type, task_idx);
    }
  #endif

    // Trigger event
    // 执行任务前有一个__syncthreads()确保真正就绪
    if (trigger_event_id != EVENT_INVALID_ID) {  
      if (threadIdx.x == 0) {
        // printf("%d.", trigger_event_id);
        size_t event_index = get_event_position_index(trigger_event_id);
        EventCounter count = atom_add_release_gpu_u64(&config.all_event_counters[event_index], 1);
        // printf("tri(%d):(%d), ", event_index, count);
      }
      // tid0设置标志后需要等待其他线程都达到，这个标志才能真正起效。否则会出现其他线程滞后，导致任务实际上没算完
      __syncthreads(); 
    }
  }
}

template <typename DT>
DT *gpu_malloc(size_t size) {
  void *dst_ptr;
#ifdef USE_NVSHMEM
  dst_ptr = nvshmem_malloc(size);
#else
  cudaMalloc(&dst_ptr, size);
#endif
  return static_cast<DT *>(dst_ptr);
}

void gpu_free(void *ptr) {
#ifdef USE_NVSHMEM
  nvshmem_free(ptr);
#else
  cudaFree(ptr);
#endif
}

// The following function will be generated by the transpiler
static void _init_persistent_kernel(int kernel_id,
                                    std::vector<void *> input_tensors,
                                    std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                    std::vector<TaskId> &first_tasks,
                                    int num_gpus,
                                    int my_gpu_id);

static RuntimeConfig global_runtime_config[100];
extern "C" void init_persistent_kernel(int kernel_id,
                                       std::vector<void *> input_tensors,
                                       std::vector<void *> meta_tensors,
                                       void *profiler_buffer,
                                       int my_rank,
                                       int num_workers,
                                       int num_local_schedulers,
                                       int num_remote_schedulers) {
  printf("init_persistent_kernel: %d.\n", kernel_id);
  // todo: 封装python填充和c++解析函数。
  global_runtime_config[kernel_id].step = nullptr;
  if (meta_tensors.size() >= 1) { 
    global_runtime_config[kernel_id].step = &((int*)meta_tensors[0])[0]; 
    if (meta_tensors.size() == 5) { 
      global_runtime_config[kernel_id].onestep_size = &((int*)meta_tensors[0])[1];
      global_runtime_config[kernel_id].onelayer_size = &((int*)meta_tensors[0])[2];
      global_runtime_config[kernel_id].kcache = (int*)meta_tensors[1];
      global_runtime_config[kernel_id].vcache = (int*)meta_tensors[2];
      global_runtime_config[kernel_id].kcache_curstep = (int*)meta_tensors[3]; 
      global_runtime_config[kernel_id].vcache_curstep = (int*)meta_tensors[4]; 
      printf("addr: %d, %d, %lld, %lld.\n", global_runtime_config[kernel_id].step, global_runtime_config[kernel_id].onestep_size, global_runtime_config[kernel_id].kcache, global_runtime_config[kernel_id].vcache);
    }
  }
  global_runtime_config[kernel_id].num_workers = num_workers;
  global_runtime_config[kernel_id].num_local_schedulers = num_local_schedulers;
  global_runtime_config[kernel_id].num_remote_schedulers = num_remote_schedulers;
  global_runtime_config[kernel_id].profiler_buffer = profiler_buffer;
  int num_schedulers = num_local_schedulers + num_remote_schedulers;

  // Initialize nvshmem
  cudaSetDevice(my_rank);

#ifdef USE_NVSHMEM
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  nvshmem_barrier_all();
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  printf("mype(%d) npes(%d) mype_node(%d)\n", mype, npes, mype_node);
#else
  int mype = 0;
  int npes = 1;
#endif

  global_runtime_config[kernel_id].infer_cnt = gpu_malloc<int>(sizeof(int));
  global_runtime_config[kernel_id].num_gpus = npes;
  global_runtime_config[kernel_id].my_gpu_id = mype;
  global_runtime_config[kernel_id].num_graphs = 1;
  global_runtime_config[kernel_id].is_static_schedule = true;

  std::vector<FullTaskDesc> all_fulltasks;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  _init_persistent_kernel(kernel_id, input_tensors, all_fulltasks, all_events, first_tasks, npes, mype);
  
  std::vector<TaskDesc> all_tasks;
  for (auto const &ft : all_fulltasks) {
    TaskDesc task_desc(ft);
    // Reinterpret part of TaskDesc to save xfer_size information
    if (ft.task_type == TASK_NVSHMEM_COPY) {
      int size_in_bytes = 2;
      for (int i = 0; i < ft.inputs[0].num_dims; i++) {
        size_in_bytes *= ft.inputs[0].dim[i];
      }
      task_desc.task_metadata.xfer_size_in_bytes = size_in_bytes;
    }
    all_tasks.push_back(task_desc);
  }

   ///////////////////////////////////////////////
  // Static Scheduling Scheme
  // is_static_schedule
  
    global_runtime_config[kernel_id].num_tasks = all_tasks.size();
    // int num_workers = global_runtime_config[kernel_id].num_workers;
    int tasks_each_worker = (all_tasks.size() + num_workers - 1) / num_workers;
    int capacity_each_worker = all_tasks.size(); // (tasks_each_worker + 1) * 1.5; // each_worker: 0:len, 1:task0, 2:task1... capacity=task_num+1
    // printf("capacity_each_worker: %d.\n", capacity_each_worker);

    // 按dep event对task分组
    // 一个event对应一组task，该event满足则对应task均可执行。
    std::vector<std::vector<int>> event_task_ids;
    event_task_ids.resize(all_events.size());
    for (int i=0; i<all_tasks.size(); i++) {
      TaskDesc task_desc = all_tasks[i];
      if (task_desc.task_type != TASK_TERMINATE && task_desc.task_type != TASK_BEGIN_TASK_GRAPH) {
        if (task_desc.dependent_event != EVENT_INVALID_ID) {
          event_task_ids[task_desc.dependent_event].push_back(i);
        }
        else {
          event_task_ids[0].push_back(i);
        }
      }
    }
  
    // host存放阶段，用于下面cudaMemcpy到显存
    // host_tasks_index[worker_id] => 属于该worker的所有task
    // host_tasks_index[worker_id][0], 表示该worker有多少个task
    std::vector<std::vector<int>> host_tasks_index;
    host_tasks_index.resize(num_workers);
    for (int i=0; i<num_workers; i++) {
      host_tasks_index[i].resize(capacity_each_worker);
      host_tasks_index[i][0] = 0; // 0 for cnt
    }

    // // 1. 按顺序直接赋值
    // for (int i=0; i<num_workers; i++) {
    //   int cnt = 0;
    //   for (int j=0; j<tasks_each_worker; j++) {
    //     int task_idx = i*tasks_each_worker+j;
    //     if (task_idx < all_tasks.size()) {
    //       cnt++;
    //       host_tasks_index[i][j + 1] = task_idx;
    //     }
    //   }
    //   host_tasks_index[i][0] = cnt;
    // }

    // // 2. 按event分组填充task到worker
    // int wid = 0;
    // for (int ei=0; ei<event_task_ids.size(); ei++) {
    //   int task_id = 0;
    //   int task_num = event_task_ids[ei].size();
    //   if (task_num == 0) continue;

    //   int base_num = task_num / num_workers; // (task_num + num_workers - 1) / num_workers;
    //   int remainder = task_num % num_workers;
      
    //   int tasks_assigned = 0;
    //   int idx = 0;  // 对应该轮的第几个worker，每个event重置1次，使前remainder个worker多拿一个task
    //   while (tasks_assigned < task_num) {
    //     int target_count = base_num + (idx < remainder ? 1 : 0);          // wid 当前该拿多少
    //     int actual_count = min(target_count, task_num - tasks_assigned);  // 边界处理
    //     // printf("[%d]actual_count: %d.\n", ei, actual_count);

    //     // TaskDesc task_desc = all_tasks[event_task_ids[ei][tasks_assigned]];
    //     // if (task_desc.task_type == TASK_SILU_MUL) {
    //     //   static int cnt = 0;
    //     //   if (cnt < (20-4)*2 && wid != 0 && wid != 17 && wid != 18 && wid != 19) {
    //     //     cnt++; 
    //     //     wid = (wid + 1) % num_workers;
    //     //     continue;            
    //     //   }
    //     // }

    //     for (int i = 0; i < actual_count; i++) {
    //       host_tasks_index[wid][host_tasks_index[wid][0] + 1] = event_task_ids[ei][tasks_assigned + i];
    //       host_tasks_index[wid][0]++;
    //     }
    //     tasks_assigned += actual_count;
    //     idx++;
    //     wid = (wid + 1) % num_workers;
    //   }
    // }

    // // 3. 按event分组填充task到worker，每个event的任务都从worker0开始
    // for (int ei=0; ei<event_task_ids.size(); ei++) {
    //   int task_id = 0;
    //   int task_num = event_task_ids[ei].size();
    //   if (task_num == 0) continue;

    //   int wid = 0;
    //   int base_num = task_num / num_workers; // (task_num + num_workers - 1) / num_workers;
    //   int remainder = task_num % num_workers;
      
    //   int tasks_assigned = 0;
    //   int idx = 0;  // 对应该轮的第几个worker，每个event重置1次，使前remainder个worker多拿一个task
    //   while (tasks_assigned < task_num) {
    //     int target_count = base_num + (idx < remainder ? 1 : 0);          // wid 当前该拿多少
    //     int actual_count = min(target_count, task_num - tasks_assigned);  // 边界处理

    //     for (int i = 0; i < actual_count; i++) {
    //       host_tasks_index[wid][host_tasks_index[wid][0] + 1] = event_task_ids[ei][tasks_assigned + i];
    //       host_tasks_index[wid][0]++;
    //     }
    //     tasks_assigned += actual_count;
    //     idx++;
    //     wid = (wid + 1) % num_workers;
    //   }
    // }

    // 3. 按event分组填充task到worker, 不分任务类别，顺序排布
    // 如： w0 0 3 6 
    //      w1 1 4 7
    //      w2 2 5 8
    int pre_task_num = 0;
    int pre_task_start_wid = 0; // 前置任务从哪个worker id开始，则后置需要跳过对应worker id。
    int wid = 0;
    for (int ei=0; ei<event_task_ids.size(); ei++) {
      int task_num = event_task_ids[ei].size();
      // printf("task_num: %d.\n", task_num);
      if (task_num == 0) continue;
        
#ifdef ENABLE_PREFETCH
      // 只支持sm富裕的情况
      int task_id = event_task_ids[ei][0];
      if (all_tasks[task_id].task_type == TASK_LINEAR || all_tasks[task_id].task_type == TASK_LINEAR_WITH_RESIDUAL || 
          all_tasks[task_id].task_type == TASK_LINEAR_HOPPER || all_tasks[task_id].task_type == TASK_LINEAR_WITH_RESIDUAL_HOPPER) {
        wid = pre_task_start_wid + pre_task_num - task_num; // (前置任务数+预取任务数) - 当前任务数 = 前置实际任务数 = 当前任务需要跳过的worker数
        // wid += 74;   // 142 - 64(fused_layout);
        wid = wid % num_workers;
      }
      pre_task_start_wid = wid;
      pre_task_num = task_num;
#endif
      //////////////////////////////////////

      int tasks_assigned = 0;
      while (tasks_assigned < task_num) {
        if (tasks_assigned >= task_num) {
          break;
        }
        host_tasks_index[wid][host_tasks_index[wid][0] + 1] = event_task_ids[ei][tasks_assigned++];
        host_tasks_index[wid][0]++;
        wid = (wid + 1) % num_workers;
      }
    }

    // 图结束event免置位: 直接取最后一个event，将对应的所有task的trigger_event，全部置为EVENT_INVALID_ID
    for (int ei=event_task_ids.size()-1; ei>0; ei--) {
      int task_num = event_task_ids[ei].size();
      if (task_num == 0) continue;

      for (int i=0; i<task_num; i++) {
        int task_id = event_task_ids[ei][i];
        // TaskDesc task_desc = all_tasks[task_id];
        // printf("task_desc.trigger_event: %d.\n", task_desc.trigger_event);
        all_tasks[task_id].trigger_event = EVENT_INVALID_ID;
      }
      break;
    }
    // 同worker同event的非首个任务，不需要等待depent。因为首个任务等待后，已满足依赖要求。
    for (int i=0; i<num_workers; i++) {
      int task_num = host_tasks_index[i][0]; // 0号是数量，1号开始才是id
      for (int j=1; j<task_num; j++) {
        int pre_id = host_tasks_index[i][j];
        int cur_id = host_tasks_index[i][j+1];
        if (all_tasks[pre_id].dependent_event == all_tasks[cur_id].dependent_event) {
          // 确认是同一worker上的非首个相同依赖的task，则跳过判断dep
          all_tasks[cur_id].dependent_event = EVENT_INVALID_ID;
        }
      }
    }

    // 分组结果展示
    for (int i=0; i<all_tasks.size(); i++) {
      TaskDesc task_desc = all_tasks[i];
      printf("task_desc[%d]: type %d, block(%d,%d,%d), dep %d, tri %d, varid %d.\n", i, task_desc.task_type, task_desc.bx, task_desc.by, task_desc.bz, task_desc.dependent_event, task_desc.trigger_event, task_desc.variant_id);
    }
    for (int i=0; i<all_events.size(); i++) {
      EventDesc event_desc = all_events[i];
      printf("event_desc[%d]: type %d, tri %d, task (%d, %d).\n", i, event_desc.event_type, event_desc.num_triggers, event_desc.first_task_id, event_desc.last_task_id);
    }
    for (int i=0; i<event_task_ids.size(); i++) {
      printf("event_group[%d]-(%d): ", i, event_task_ids[i].size());
      for (int j=0; j<event_task_ids[i].size(); j++) {
        printf("%d, ", event_task_ids[i][j]);
      }
      printf("\n");
    }
    for (int i=0; i<num_workers; i++) {
      int num = host_tasks_index[i][0];
      printf("worker[%d]-(%d): ", i, num);
      for (int j=0; j<num; j++) {
        int id = host_tasks_index[i][j+1];
        printf("%d(%d-%d)(%d=%d), ", id, all_tasks[id].task_type, all_tasks[id].bx, all_tasks[id].dependent_event, all_tasks[id].trigger_event);
      }
      printf("\n");
    }
    ////////////////////////////////////////////////

    std::vector<int*> host_tasks_index_arr;
    for (int i = 0; i < num_workers; i++) {
      int *device_tasks_index = gpu_malloc<int>(capacity_each_worker * sizeof(int));
      cudaMemcpy(device_tasks_index, host_tasks_index[i].data(), capacity_each_worker * sizeof(int), cudaMemcpyHostToDevice);
      host_tasks_index_arr.push_back(device_tasks_index);
    }

    global_runtime_config[kernel_id].static_worker_tasks_index = gpu_malloc<int*>(num_workers * sizeof(int*));
    cudaMemcpy(global_runtime_config[kernel_id].static_worker_tasks_index,
               host_tasks_index_arr.data(),
               num_workers * sizeof(int*), cudaMemcpyHostToDevice);
  
  //  Initialize all event counters
  global_runtime_config[kernel_id].all_event_counters = gpu_malloc<EventCounter>(all_events.size() * sizeof(EventCounter));
  global_runtime_config[kernel_id].all_event_num_triggers = gpu_malloc<int>(all_events.size() * sizeof(int));

  std::vector<int> host_all_event_counters;
  for (size_t i = 0; i < all_events.size(); i++) {
    host_all_event_counters.push_back(all_events.at(i).num_triggers);
  }
  cudaMemcpy(global_runtime_config[kernel_id].all_event_num_triggers,
             host_all_event_counters.data(),
             all_events.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  // cudaMemset(global_runtime_config[kernel_id].all_event_counters,
  //            0,
  //            all_events.size() * sizeof(EventCounter));
  //  Initialize all tasks
  global_runtime_config[kernel_id].all_tasks = gpu_malloc<TaskDesc>(all_tasks.size() * sizeof(TaskDesc));
  cudaMemcpy(global_runtime_config[kernel_id].all_tasks,
             all_tasks.data(),
             all_tasks.size() * sizeof(TaskDesc),
             cudaMemcpyHostToDevice);

#ifdef ENABLE_PREFETCH
  TaskDesc* d_all_tasks = global_runtime_config[kernel_id].all_tasks;
  for (int i = 0; i < num_workers; i++) {
      int num = host_tasks_index[i][0];
      for (int j = 0; j < num - 1; j++) {
        int id = host_tasks_index[i][j+1];
        int post_id = host_tasks_index[i][j+2];
        
        TaskDesc* post_task_device_addr = d_all_tasks + post_id;  // device 指针
        cudaMemcpy(&d_all_tasks[id].post_task, 
                  &post_task_device_addr, 
                  sizeof(TaskDesc*), 
                  cudaMemcpyHostToDevice);
      }
  }
#endif


  // Initialize all events
  global_runtime_config[kernel_id].num_events = (int)all_events.size();
  global_runtime_config[kernel_id].all_events = gpu_malloc<EventDesc>(all_events.size() * sizeof(EventDesc));
  cudaMemcpy(global_runtime_config[kernel_id].all_events,
             all_events.data(),
             all_events.size() * sizeof(EventDesc),
             cudaMemcpyHostToDevice);

  // Initialize first tasks
  {
    global_runtime_config[kernel_id].first_tasks = gpu_malloc<TaskId>(first_tasks.size() * sizeof(TaskId));
    cudaMemcpy(global_runtime_config[kernel_id].first_tasks,
               first_tasks.data(),
               first_tasks.size() * sizeof(TaskId),
               cudaMemcpyHostToDevice);
  }

  // Set configuration for kernels
  cudaFuncSetAttribute(static_persistent_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
}

void print_smem_size() {
  int device_id = 0;
  cudaSetDevice(device_id);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);

  // Print GPU shared memory hardware limits (all in English)
  printf("=== GPU Shared Memory Hardware Limits ===\n");
  printf("GPU Model: %s\n", prop.name);
  // printf("Max Dynamic Shared Memory per Block: %lu KB\n", prop.sharedMemPerBlockDynamic / 1024);
  if (prop.sharedMemPerBlock < MAX_DYNAMIC_SHARED_MEMORY_SIZE) {
    printf("Warning: Dynamic shared memory may be insufficient!\n");
  }
  printf("Max Total Shared Memory per Block (Static + Dynamic): %lu KB, prepare to allocate %d KB (Dynamic).\n", prop.sharedMemPerBlock / 1024, MAX_DYNAMIC_SHARED_MEMORY_SIZE / 1024);
  printf("Total Shared Memory per SM: %lu KB\n", prop.sharedMemPerMultiprocessor / 1024);
}

// Entry point for C/C++
// TODO: change launch config
extern "C" void launch_persistent_kernel(int kernel_id, int batch_size, int layer_id, cudaStream_t stream) {
  // printf("launch_persistent_kernel: %d.\n", kernel_id);
  // int device;
  // cudaGetDevice(&device);
  // int sm_count;
  // cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  // global_runtime_config[kernel_id].batch_size = batch_size;
  global_runtime_config[kernel_id].layer_id = layer_id;
  cudaMemsetAsync(global_runtime_config[kernel_id].all_event_counters, 0, 
    sizeof(EventCounter) * global_runtime_config[kernel_id].num_events, stream);
  static_persistent_kernel<<<dim3(global_runtime_config[kernel_id].num_workers, 1, 1),
      dim3(WORKER_NUM_THREADS, 1, 1),
      MAX_DYNAMIC_SHARED_MEMORY_SIZE,
      stream>>>(
      global_runtime_config[kernel_id]);      

  // cudaError_t err = cudaDeviceSynchronize();
  // if (err != cudaSuccess) {
  //   printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  // }
  // printf("Finished Launch Persistent Kernel\n");
}

extern "C" void finalize_persistent_kernel(int kernel_id) {
  gpu_free(global_runtime_config[kernel_id].all_event_counters);
  gpu_free(global_runtime_config[kernel_id].all_event_num_triggers);
  gpu_free(global_runtime_config[kernel_id].all_tasks);
  gpu_free(global_runtime_config[kernel_id].all_events);

  gpu_free(global_runtime_config[kernel_id].infer_cnt);

  int num_workers = global_runtime_config[kernel_id].num_workers;

  if (global_runtime_config[kernel_id].is_static_schedule == true) {
    std::vector<int*> host_tasks_index(num_workers);
    cudaMemcpy(host_tasks_index.data(),
             global_runtime_config[kernel_id].static_worker_tasks_index,
             num_workers * sizeof(int *),
             cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_workers; i++) {
      gpu_free(host_tasks_index[i]);
    }
    gpu_free(global_runtime_config[kernel_id].static_worker_tasks_index);
  }

  gpu_free(global_runtime_config[kernel_id].first_tasks);
#ifdef USE_NVSHMEM
  nvshmem_barrier_all();
  nvshmem_finalize();
#endif
}
