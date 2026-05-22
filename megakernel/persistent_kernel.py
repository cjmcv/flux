import torch
import os
import tempfile
import subprocess
import shutil
import sys
import sysconfig
from pathlib import Path

from .core import *
from .kernel import get_key_paths, KNGraph, TBGraph
from .task_graph_visualizer import display_task_graph


INSTANCE_REUSE_PLUGIN = """
// Plugin
void adjust_params_with_kernel_id(int kernel_id, std::map<std::string, void*> &all_tensors) {
  if (kernel_id == 1) {
    all_tensors["w1"] = all_tensors["w2"];
  }
}
"""

HARD_CODE = """
#include <Python.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

static PyObject *init_func(PyObject *self, PyObject *args) {
  PyObject *input_list, *meta_list, *py_profiler_buffer;
  std::vector<void*> input_tensors;
  std::vector<void*> meta_tensors;
  int kernel_id, my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers;
  void *profiler_buffer;

  if (!PyArg_ParseTuple(args, "iOOOiiii", &kernel_id, &input_list, &meta_list, &py_profiler_buffer, &my_mpi_rank, &num_workers, &num_local_schedulers, &num_remote_schedulers)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }
  // inputs
  if(!PyList_Check(input_list)) {
    PyErr_SetString(PyExc_TypeError, "arg1 must be a list.");
    return NULL;
  }

  Py_ssize_t inputs_size = PyList_Size(input_list);

  for(Py_ssize_t i = 0; i < inputs_size; i++) {
    PyObject *item = PyList_GetItem(input_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (inputs) to void pointer", i);
      return NULL;
    }
    input_tensors.push_back(PyLong_AsVoidPtr(item));
  }
  // meta
  if(!PyList_Check(meta_list)) {
    PyErr_SetString(PyExc_TypeError, "arg1 must be a list.");
    return NULL;
  }

  Py_ssize_t meta_size = PyList_Size(meta_list);

  for(Py_ssize_t i = 0; i < meta_size; i++) {
    PyObject *item = PyList_GetItem(meta_list, i);
    void* tensor = PyLong_AsVoidPtr(item);
    if(!tensor) {
      PyErr_Format(PyExc_TypeError, "Failed to convert item %d (meta) to void pointer", i);
      return NULL;
    }
    meta_tensors.push_back(PyLong_AsVoidPtr(item));
  }
  profiler_buffer = PyLong_AsVoidPtr(py_profiler_buffer);

  init_persistent_kernel(kernel_id, input_tensors, meta_tensors, profiler_buffer, my_mpi_rank, num_workers, num_local_schedulers, num_remote_schedulers);
  Py_RETURN_NONE;
}

static PyObject *launch_func(PyObject *self, PyObject *args) {
  int kernel_id = 0, batch_size = 0, layer_id = 0;
  if (!PyArg_ParseTuple(args, "iii", &kernel_id, &batch_size, &layer_id)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }
  
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  launch_persistent_kernel(kernel_id, batch_size, layer_id, stream);

  Py_RETURN_NONE;
}

static PyObject *finalize_func(PyObject *self, PyObject *args) {
  int kernel_id = 0;
  if (!PyArg_ParseTuple(args, "i", &kernel_id)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }
  finalize_persistent_kernel(kernel_id);

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"init_func", init_func, METH_VARARGS, "initialize persistent kernel"},
  {"launch_func", launch_func, METH_VARARGS, "launch persistent kernel"},
  {"finalize_func", finalize_func, METH_VARARGS, "finalize persistent kernel"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__megakernel_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods,
  NULL, // m_slots
  NULL, // m_traverse
  NULL, // m_clear
  NULL  // m_free
};

PyMODINIT_FUNC PyInit___megakernel_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""

valid_persistent_kernel_modes = {"offline", "online", "onepass"}

def get_compile_command(
    mpk,
    target_cc,
    cc,
    file_name,
    py_include_dir,
    XOP_HOME_path,
    megakernel_inc_path,
    megakernel_deps_path,
    nvshmem_inc_path,
    nvshmem_lib_path,
    mpi_inc_path,
    mpi_lib_path,
    py_so_path,
    profiling,
    enable_prefetch,
    use_nvshmem,
    num_workers=None,
    num_local_schedulers=None,
    num_remote_schedulers=None,
    model_tag="default",
):
    max_worker_per_scheduler = 128
    if num_workers != None and num_local_schedulers != None and num_remote_schedulers != None:
        min_schedulers = 0
        if num_remote_schedulers == 0:
            min_schedulers = num_local_schedulers
        else:
            min_schedulers = min(num_local_schedulers, num_remote_schedulers)
        # advance by 1 for the scheduler who are handling the not divisiable num_worker.
        max_worker_per_scheduler = (num_workers // min_schedulers) + 1
        
    site_packages_path = sysconfig.get_path('purelib')
    torch_pkg_path = os.path.join(site_packages_path, 'torch')
    torch_include_path = os.path.join(torch_pkg_path, 'include')
    torch_lib_path = os.path.join(torch_pkg_path, 'lib')
    print("torch: ", torch_include_path, torch_lib_path)
    
    common_cmd = [
        cc,
        file_name,
        "-O3",
        # Use following flags when debugging
        # "-O0",
        # "-g",
        # "-G",
        "--ptxas-options=-v",
        "-Xptxas=-v",
        "-lineinfo",
        f"-I{py_include_dir}",
        f"-I{megakernel_inc_path}",
        f"-I{os.path.join(megakernel_inc_path, 'megakernel')}",
        f"-I{os.path.join(megakernel_inc_path, 'megakernel/persistent_kernel')}",
        f"-I{os.path.join(megakernel_deps_path, 'cutlass/include')}",
        f"-I{os.path.join(megakernel_deps_path, 'cutlass/tools/util/include')}",
        f"-I{os.path.join(megakernel_deps_path, 'json/include')}",
        f"-I{torch_include_path}",
        f"-DMAX_WORKER_PER_SCHEDULER={max_worker_per_scheduler}",
    ]
    
    flags = [
        "-shared",
        "-std=c++17",
        "-rdc=false" if not use_nvshmem else "-rdc=true",
        "-use_fast_math",
        "-lcuda",
        "-Xcompiler=-fPIC",
        "--expt-relaxed-constexpr",
        "-o",
        py_so_path,
    ]
    flags = flags + [f"-DMPK_TARGET_CC={target_cc}", "-DMEGAKERNEL_BACKEND_USE_CUDA"]
    flags = flags + [f"-ltorch -lc10 -L{torch_lib_path}"]
    
    if (model_tag == "qwen3_06b"):
        flags = flags + ["-DENABLE_QWEN3_06B"]
    elif (model_tag == "qwen3_4b"):
        flags = flags + ["-DENABLE_QWEN3_4B"] 
    else:
        flags = flags + ["-DENABLE_QWEN3_06B"]
        print(f"Do not supported model_tag:{model_tag}, use default flags")
        
    if enable_prefetch:
        flags = flags + ["-DENABLE_PREFETCH"]
        
    if use_nvshmem:
        nvshmem_cmd = [
            f"-I{nvshmem_inc_path}",
            f"-I{mpi_inc_path}",
            f"-L{nvshmem_lib_path}",
            f"-L{mpi_lib_path}",
        ]
        nvshmem_flags = ["-DUSE_NVSHMEM", "-ccbin=mpic++", "-lnvshmem_host", "-lnvshmem_device", "-lmpi"]
        common_cmd = common_cmd + nvshmem_cmd
        flags = flags + nvshmem_flags

    if target_cc == 89:
        specific_cmd = [
            "-arch=sm_89",
            "-gencode=arch=compute_89,code=sm_89",
            "-DNDEBUG",
        ] + (["-DMEGAKERNEL_ENABLE_PROFILER"] if profiling else [])
    elif target_cc == 90:
        # h20=>sm90, h100=>sm90a
        specific_cmd = [
            "-arch=sm_90",
            "-gencode=arch=compute_90,code=sm_90",
            # "-arch=sm_90a",
            # "-gencode=arch=compute_90a,code=sm_90a",
            "-DMPK_ENABLE_TMA",
            "-DMEGAKERNEL_GRACE_HOPPER",
            "-DNDEBUG",
        ] + (["-DMEGAKERNEL_ENABLE_PROFILER"] if profiling else [])
    elif target_cc == 120:
        # h20=>sm90, h100=>sm90a
        specific_cmd = [
            "-arch=sm_120",
            "-gencode=arch=compute_120,code=sm_120",
            "-DMPK_ENABLE_TMA",
            "-DMEGAKERNEL_GRACE_BLACKWELL",
            "-DNDEBUG",
        ] + (["-DMEGAKERNEL_ENABLE_PROFILER"] if profiling else [])
    elif target_cc == 100:
        specific_cmd = [
            "-arch=sm_100a",
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DMPK_ENABLE_TMA",
            "-DMEGAKERNEL_GRACE_BLACKWELL",
        ]
    else:
        specific_cmd = [
            "-arch=native",
        ]
    
    if profiling:
        flags = flags + ["-DMPK_ENABLE_PROFILING"]

    return common_cmd + specific_cmd + flags


class PersistentKernel:
    def __init__(
        self,
        instance_id: int,  # 多实例，用于graph切换
        kernel_num: int,   # 多kernel共享，用于一个kernel切换多份权重，实现多kernel效果。
        mode: str,
        world_size: int,
        mpi_rank: int,
        num_workers: int,
        num_local_schedulers: int,
        num_remote_schedulers: int,
        # max_num_batched_requests: int,
        # max_num_batched_tokens: int,
        meta_tensors: dict,
        profiler_tensor: torch.Tensor,
        trace_name: str,
        # spec_decode_config: SpecDecodeConfig,
        model_tag: str
    ):
        self.model_tag = model_tag
        self.instance_id = instance_id
        self.kernel_num = kernel_num
        
        self.__finalized__ = False
        self._is_compiled = False
        if mode not in valid_persistent_kernel_modes:
            raise ValueError(f"Invalid persistent kernel mode: {mode}")
        self.mode = mode
        self.world_size = world_size
        self.mpi_rank = mpi_rank
        self.num_workers = num_workers
        self.num_local_schedulers = num_local_schedulers
        self.num_remote_schedulers = num_remote_schedulers
        # self.max_num_batched_requests = max_num_batched_requests
        # self.max_num_batched_tokens = max_num_batched_tokens
        self.kn_graph = KNGraph(CyKNGraph())
        self.meta_tensors = meta_tensors
        self.profiler_tensor = profiler_tensor
        self.trace_name = trace_name
        self.use_nvshmem = True if world_size > 1 else False

        self.target_cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
        if self.target_cc >= 90:
            self.thread_num = 256
        else:
            self.thread_num = 128
            
        # For the reuse of instance
        self.repl_weight_mapping = {}

    def attach_input(self, torch_tensor: torch.Tensor, name: str = None) -> DTensor:
        dims = tuple([d for d in torch_tensor.shape])
        strides = tuple([s for s in torch_tensor.stride()])
        # Assert a row-major layout
        for d in range(len(dims) - 1):
            assert strides[d] == strides[d + 1] * dims[d + 1]
        dtype = convert_torch_type_to_dtype(torch_tensor.dtype)
        t = self.kn_graph.new_input(dims=dims, strides=strides, dtype=dtype)
        # FIXME: currently assert that name is not None
        assert name is not None
        self.kn_graph.attach_torch_tensor(t, torch_tensor, name)
        return t

    def new_tensor(
        self,
        dims: tuple,
        strides: tuple = None,
        dtype: dtype = bfloat16,
        name: str = None,
        io_category: str = "cuda_tensor",
    ) -> DTensor:
        # Assert a row-major layout
        # if strides is not None:
        #     for d in range(len(dims) - 1):
        #         assert strides[d] == strides[d + 1] * dims[d + 1]
        t = self.kn_graph.new_input(dims=dims, strides=strides, dtype=dtype)
        # FIXME: currently assert that name is not None
        assert name is not None
        if io_category == "cuda_tensor":
            self.kn_graph.attach_cuda_tensor(t, name)
        elif io_category == "nvshmem_tensor":
            self.kn_graph.attach_nvshmem_tensor(t, name)
        else:
            raise RuntimeError(f"Invalid io_category: {io_category}")
        return t

    def rmsnorm_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        sync_mode: tuple,
        layout: tuple,
        fused_params: list = None, # flag 99, funcid, layout[3]
        fused_tensor: DTensor = None,
    ):
        grid_dim, tile_dim = layout
        assert input.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, tile_dim, self.thread_num))
        print("sync_mode", sync_mode)
        tb_graph.new_input(input, sync_mode)
        tb_graph.new_input(weight, sync_mode)
        tb_graph.new_input(output, (-1, -1, -1))
        if fused_tensor is not None:
            tb_graph.new_input(fused_tensor, (-1, -1, -1))
            self.kn_graph.customized([input, weight, output, fused_tensor], tb_graph)
        else:
            self.kn_graph.customized([input, weight, output], tb_graph)
        self.kn_graph.register_task("rmsnorm", (fused_params if fused_params is not None else []))

    def rmsnorm_linear_layer(
        self,
        input: DTensor,
        weight_norm: DTensor,
        weight_linear: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that the input/weight_linear/output are 2D tensors
        assert input.num_dims == 2
        assert weight_linear.num_dims == 2
        assert output.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1))
        tb_graph.new_input(input, (-1, -1, -1))
        tb_graph.new_input(weight_norm, (-1, -1, -1))
        tb_graph.new_input(weight_linear, (0, -1, -1))
        tb_graph.new_input(output, (1, -1, -1))
        self.kn_graph.customized([input, weight_norm, weight_linear, output], tb_graph)
        self.kn_graph.register_task("rmsnorm_linear")

    # fused_params, micro kernel手动插桩基本思路
    # ┌─────────────────────────────────────────────────────────────────────────────┐
    # │ Kernel A         │ Kernel B        │ Kernel C        │ Kernel D             │
    # ├─────────────────────────────────────────────────────────────────────────────┤
    # │ SM0 ████████████ │ ████████████    │ ████████████    │ ████████████         │
    # │ SM1 ████████████ │ ████████████    │ ████████████    │ ████████████         │
    # │ SM2 ░░░░░░░░░░░░ │ ████████████    │ ░░░░░░░░░░░░    │ ████████████         │
    # │ SM3 ░░░░░░░░░░░░ │ ████████████    │ ░░░░░░░░░░░░    │ ░░░░░░░░░░░░         │
    # └─────────────────────────────────────────────────────────────────────────────┘
    #  时间轴 →                                    ↓
    # ┌─────────────────────────────────────────────────────────────────────────────┐
    # │ Kernel A         │ Kernel B        │ Kernel C        │ Kernel D             │
    # ├─────────────────────────────────────────────────────────────────────────────┤
    # │ SM0 ████████████ │ ████████████    │ ████████████    │ ████████████         │
    # │ SM1 ████████████ │ ████████████    │ ████████████    │ ████████████         │
    # │ SM2 ░░░░░░░░░░░░ │ ████████████    │ ████████████    │ ████████████         │
    # │ SM3 ░░░░░░░░░░░░ │ ████████████    │ ████████████    │ ░░░░░░░░░░░░         │
    # └─────────────────────────────────────────────────────────────────────────────┘
    #                                    插入可并行的micro kernel
    #                                       如pipeline处理相关
    #
    # 操作方式：通过fused_params让对应kernel传入参数，在compile_load时传入meta数据，包含关键的输入输出数据。
    #          内部解释kernel时，根据fused_params结合meta数据，将额外的micro kernel插入到该kernel中，
    #          并通过layout范围来指定哪些sm用于执行原kernel，哪些用于新插入的micro kernel
    def rope_layer(
        self,
        q: DTensor,
        k: DTensor, 
        cos: DTensor,
        sin: DTensor,
        q_embed: DTensor,
        k_embed: DTensor,
        sync_mode: tuple,
        layout: tuple,
        fused_params: list = None, # flag 99 (固定，扩展micro kernel的参数开始标志), funcid, layout[3]
    ):
        assert q.num_dims == 4
        assert k.num_dims == 4
        assert cos.num_dims == 3
        assert sin.num_dims == 3
        assert q_embed.num_dims == 4
        assert k_embed.num_dims == 4
    
        grid_dim, tile_dim = layout
        print(grid_dim, tile_dim, sync_mode)
        tb_graph = TBGraph(CyTBGraph(grid_dim, tile_dim, self.thread_num))
        tb_graph.new_input(q,       sync_mode)
        tb_graph.new_input(k,       sync_mode)
        tb_graph.new_input(cos,     sync_mode)
        tb_graph.new_input(sin,     sync_mode)
        tb_graph.new_input(q_embed, (-1, -1, -1))
        tb_graph.new_input(k_embed, (-1, -1, -1))
        
        self.kn_graph.customized([q, k, cos, sin, q_embed, k_embed], tb_graph)
        self.kn_graph.register_task("rope", [-1]+(fused_params if fused_params is not None else [])) # TASK_ROPE
                        
    def gqa_decode_layer(
        self,
        q: DTensor,
        k_cache: DTensor, 
        v_cache: DTensor,
        edge: DTensor,
        mask: DTensor,
        glse: DTensor,
        out_partial: DTensor,
        output: DTensor,
        sync_mode: tuple,
        layout: tuple,
        fused_params: list = None, # flag 99, funcid
    ):
        assert q.num_dims == 3
        assert output.num_dims == 3
    
        for i in range(0, len(layout), 2):
            grid_dim, tile_dim = layout[i], layout[i+1]
            # print(grid_dim, tile_dim, sync_mode)
            tb_graph = TBGraph(CyTBGraph(grid_dim, tile_dim, self.thread_num))
            tb_graph.new_input(q,       sync_mode)
            tb_graph.new_input(k_cache, sync_mode)
            tb_graph.new_input(v_cache, sync_mode)
            tb_graph.new_input(edge,    sync_mode)
            tb_graph.new_input(mask,    sync_mode)
            tb_graph.new_input(output, (-1, -1, -1))
            tb_graph.new_input(glse,   (-1, -1, -1))
            tb_graph.new_input(out_partial, (-1, -1, -1))
            
            self.kn_graph.customized([q, k_cache, v_cache, edge, mask, output, glse, out_partial], tb_graph)
            if len(layout) == 2:
                self.kn_graph.register_task("gqa_decode", [0]+(fused_params if fused_params is not None else []))
            else:
                self.kn_graph.register_task("gqa_decode", [i//2]+(fused_params if fused_params is not None else [])) # sub kernel id for combined kernel
        
    def linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        sync_mode: tuple,
        layout: tuple,
    ):
        grid_dim, tile_dim = layout
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, tile_dim, self.thread_num))
        tb_graph.new_input(input, sync_mode)
        tb_graph.new_input(weight, sync_mode)
        tb_graph.new_input(output, (-1, -1, -1))
        self.kn_graph.customized([input, weight, output], tb_graph)

        if self.target_cc == 120:
            self.kn_graph.register_task("linear_hopper")
        elif self.target_cc == 100:
            self.kn_graph.register_task("linear_sm100")
        elif self.target_cc == 90:
            self.kn_graph.register_task("linear_hopper")
        elif self.target_cc == 80 or self.target_cc == 89:
            self.kn_graph.register_task("linear")
        else:
            assert False
    
    def linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        sync_mode: tuple,
        layout: tuple,
    ):
        grid_dim, tile_dim = layout
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, tile_dim, self.thread_num))
        tb_graph.new_input(input, sync_mode)
        tb_graph.new_input(weight, sync_mode)
        tb_graph.new_input(residual, sync_mode)
        tb_graph.new_input(output, (-1, -1, -1))
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        
        if self.target_cc == 120:
            self.kn_graph.register_task("linear_with_residual_hopper")
        elif self.target_cc == 100:
            self.kn_graph.register_task("linear_with_residual_sm100")
        elif self.target_cc == 90:
            self.kn_graph.register_task("linear_with_residual_hopper")
        elif self.target_cc == 80 or self.target_cc == 89:
            self.kn_graph.register_task("linear_with_residual")
        else:
            assert False
                    
    def silu_mul_linear_layer(
        self,
        input: DTensor,
        weight: DTensor,
        output: DTensor,
        sync_mode: tuple,
        grid_dim: tuple,
        tile_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, hidden_size / world_size)
        assert weight.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, tile_dim, self.thread_num))
        tb_graph.new_input(input, sync_mode)
        tb_graph.new_input(weight, sync_mode)
        tb_graph.new_input(output, (-1, -1, -1))
        self.kn_graph.customized([input, weight, output], tb_graph)
        if self.target_cc == 80 or self.target_cc == 89:
            self.kn_graph.register_task("silu_mul_linear")
        else:
            assert False

    def silu_mul_layer(
        self,
        input: DTensor,
        output: DTensor,
        sync_mode: tuple, 
        layout: tuple,
        fused_params: list = None, # flag 99, funcid, layout[3]
        fused_tensor: DTensor = None,
    ):
        grid_dim, tile_dim = layout
        # Currently assume that input/output
        assert input.num_dims == 2 # (batch_size, 2 * intermediate_size)
        assert output.num_dims == 2 # (batch_size, intermediate_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, tile_dim, self.thread_num)) # CJM_TODO: thread_num应由megakernel初始化时指定，不能更改
        tb_graph.new_input(input, sync_mode)
        tb_graph.new_input(output, (-1, -1, -1))
        if fused_tensor is not None:
            tb_graph.new_input(fused_tensor, (-1, -1, -1))
            self.kn_graph.customized([input, output, fused_tensor], tb_graph)
        else:
            self.kn_graph.customized([input, output], tb_graph)
        self.kn_graph.register_task("silu_mul", (fused_params if fused_params is not None else []))

    def silu_mul_linear_with_residual_layer(
        self,
        input: DTensor,
        weight: DTensor,
        residual: DTensor,
        output: DTensor,
        grid_dim: tuple,
        block_dim: tuple,
    ):
        # Currently assume that input/output
        assert input.num_dims == 2  # (batch_size, 2*intermediate_size)
        assert weight.num_dims == 2  # (hidden_size, intermediate_size)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1))
        tb_graph.new_input(input, (-1, -1, -1))
        tb_graph.new_input(weight, (0, -1, -1))
        tb_graph.new_input(residual, (1, -1, -1))
        tb_graph.new_input(output, (1, -1, -1))
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        self.kn_graph.register_task("silu_mul_linear_with_residual")

    def gen_plugin_code(self):
        assert len(self.repl_weight_mapping) < self.kernel_num 
        plugin_code = "// Plugin \n"
        
        # replace weights
        func_str = "void adjust_params_with_kernel_id(int kernel_id, std::map<std::string, void*> &all_tensors) {\n"

        if len(self.repl_weight_mapping) != 0:
            sorted_kernel_ids = sorted(self.repl_weight_mapping.keys())
            for idx, kernel_id in enumerate(sorted_kernel_ids):
                t_str = f"  {'if' if idx == 0 else 'else if'} (kernel_id == {kernel_id}) {{\n"
                
                weight_pairs = self.repl_weight_mapping[kernel_id]
                for base_weight, target_weight in weight_pairs:
                    t_str += "    all_tensors[\"{0}\"] = all_tensors[\"{1}\"];\n".format(base_weight, target_weight)
                t_str += "  }\n"
                func_str += t_str
                
        func_str += "}\n"
        
        plugin_code += func_str
        return plugin_code
        
    def compile(self, **kwargs):
        assert not self._is_compiled
        
        enable_prefetch = kwargs.get("enable_prefetch", False)
        output_dir = kwargs.get("output_dir", None)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        MEGAKERNEL_ROOT, INCLUDE_PATH, DEPS_PATH = get_key_paths()
        print("get_key_paths", get_key_paths())
        # tempdir_obj = tempfile.TemporaryDirectory()
        # tempdir = "./gen/" # tempdir_obj.name
        results = self.kn_graph.generate_task_graph(num_gpus=self.world_size, my_gpu_id=self.mpi_rank)

        cuda_code_path = os.path.join(output_dir, "test.cu")
        so_path = os.path.join(output_dir, "test.cpython-38-x86_64-linux-gnu.so")
        # self.kn_graph.visualize(os.path.join(output_dir, "kn_graph"))
        
        # if (kernel_id == 1) {
        if 1:
            plugin_code = self.gen_plugin_code()
            
            # check json file
            json_file_path = os.path.join(output_dir, "task_graph.json")
            with open(json_file_path, "w") as f:
                f.write(results["json_file"])
            with open(cuda_code_path, "w") as f:
                f.write(results["cuda_code"])
                f.write(plugin_code)
                f.write(HARD_CODE)
                
            display_task_graph(json_file_path, False)
        
        # if output_dir is not None:
        #     os.makedirs(output_dir, exist_ok=True)
        #     shutil.copy(cuda_code_path, os.path.join(output_dir, f"test_rank{self.mpi_rank}.cu"))
        #     shutil.copy(json_file_path, os.path.join(output_dir, f"task_graph_rank{self.mpi_rank}.json"))

        cc = shutil.which("nvcc")
        if cc is None:
            raise RuntimeError(
                "nvcc not found. Please make sure you have installed CUDA."
            )
        # This function was renamed and made public in Python 3.10
        if hasattr(sysconfig, "get_default_scheme"):
            scheme = sysconfig.get_default_scheme()
        else:
            scheme = sysconfig._get_default_scheme()
        # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
        # path changes to include 'local'. This change is required to use triton with system-wide python.
        if scheme == "posix_local":
            scheme = "posix_prefix"
        py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

        # find megakernel home
        if "XOP_HOME" in os.environ:
            XOP_HOME_PATH = os.environ.get("XOP_HOME")
        else:
            raise RuntimeError(
                "XOP_HOME unspecified; Please set XOP_HOME to be the root of the Mirage folder"
            )

        NVSHMEM_INC_PATH = None
        NVSHMEM_LIB_PATH = None
        MPI_INC_PATH = None
        MPI_LIB_PATH = None
        if self.use_nvshmem:
            # find nvshmem include folder and library folder
            if "NVSHMEM_INC_PATH" in os.environ:
                NVSHMEM_INC_PATH = os.environ.get("NVSHMEM_INC_PATH")
                header_file_path = os.path.join(NVSHMEM_INC_PATH, "nvshmem.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        "Environment variable NVSHMEM_INC_PATH is set but cannot find nvshmem.h at {header_file_path}"
                    )
            else:
                NVSHMEM_INC_PATH = "/usr/include/nvshmem_12/"
                header_file_path = os.path.join(NVSHMEM_INC_PATH, "nvshmem.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        "Cannot find nvshmem.h, please set environment variable NVSHMEM_INC_PATH"
                    )
            # find nvshmem shared library
            if "NVSHMEM_LIB_PATH" in os.environ:
                NVSHMEM_LIB_PATH = os.environ.get("NVSHMEM_LIB_PATH")
                lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libnvshmem.a")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        "Environment variable NVSHMEM_LIB_PATH is set but cannot find libnvshmem.a at {lib_file_path}"
                    )
            else:
                NVSHMEM_LIB_PATH = "/usr/lib/x86_64-linux-gnu/"
                lib_file_path = os.path.join(NVSHMEM_LIB_PATH, "libnvshmem.a")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        "Cannot find libnvshmem.a, please set environment variable NVSHMEM_LIB_PATH"
                    )
            # find mpi include foler
            if "MPI_INC_PATH" in os.environ:
                MPI_INC_PATH = os.environ.get("MPI_INC_PATH")
                header_file_path = os.path.join(MPI_INC_PATH, "mpi.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        f"Environment variable MPI_INC_PATH is set but cannot find mpi.h at {header_file_path}"
                    )
            else:
                MPI_INC_PATH = "/usr/include/"
                header_file_path = os.path.join(MPI_INC_PATH, "mpi.h")
                if not os.path.exists(header_file_path):
                    raise RuntimeError(
                        f"Cannot find mpi.h, please set environment variable MPI_INC_PATH"
                    )
            # find mpi shared library
            if "MPI_LIB_PATH" in os.environ:
                MPI_LIB_PATH = os.environ.get("MPI_LIB_PATH")
                lib_file_path = os.path.join(MPI_LIB_PATH, "libmpi.so")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        f"Environment variable MPI_LIB_PATH is set but cannot find libmpi.so at {lib_file_path}"
                    )
            else:
                NVSHMEM_LIB_PATH = "/usr/lib/"
                lib_file_path = os.path.join(MPI_LIB_PATH, "libmpi.so")
                if not os.path.exists(lib_file_path):
                    raise RuntimeError(
                        f"Cannot find libmpi.so, please set environment variable MPI_LIB_PATH"
                    )

        cc_cmd = get_compile_command(
            mpk=self,
            target_cc=self.target_cc,
            cc=cc,
            file_name=cuda_code_path,
            py_include_dir=py_include_dir,
            XOP_HOME_path=XOP_HOME_PATH,
            megakernel_inc_path=INCLUDE_PATH,
            megakernel_deps_path=DEPS_PATH,
            nvshmem_inc_path=NVSHMEM_INC_PATH,
            nvshmem_lib_path=NVSHMEM_LIB_PATH,
            mpi_inc_path=MPI_INC_PATH,
            mpi_lib_path=MPI_LIB_PATH,
            py_so_path=so_path,
            profiling=True if self.profiler_tensor is not None else False,
            enable_prefetch=enable_prefetch,
            use_nvshmem=self.use_nvshmem,
            num_workers=self.num_workers,
            num_local_schedulers=self.num_local_schedulers, 
            num_remote_schedulers=self.num_remote_schedulers,
            model_tag=self.model_tag,
        )
        print("Compiling megakernel using the following command line:")
        print(cc_cmd)
        subprocess.check_call(cc_cmd)
        print("Finished megakernel compilation...")
        return so_path

    def load_module(self, so_path, input_tensors=list(), meta_tensors=list()):
        import importlib.util
        spec = importlib.util.spec_from_file_location("__megakernel_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.init_func = getattr(mod, "init_func")
        self.launch_func = getattr(mod, "launch_func")
        self.finalize_func = getattr(mod, "finalize_func")

        input_tensors_ptr = [tensor.data_ptr() for tensor in input_tensors]
        meta_tensors_ptr = [tensor.data_ptr() for tensor in meta_tensors]
        profiler_buffer_ptr = (
            self.profiler_tensor.data_ptr() if self.profiler_tensor is not None else 0
        )
        # print("meta_tensors_ptr ", len(meta_tensors), len(meta_tensors_ptr))
        print("kernel_num: ", self.kernel_num)
        for kernel_id in range(self.kernel_num):
            self.init_func(
                self.instance_id*self.kernel_num + kernel_id,
                input_tensors_ptr,
                meta_tensors_ptr,
                profiler_buffer_ptr,
                self.mpi_rank,
                self.num_workers,
                self.num_local_schedulers,
                self.num_remote_schedulers,
            )

        self._is_compiled = True
        print("Finished megakernel Loading...")
        # self.call_func = getattr(mod, "call_func")
        
    def __call__(self, batch_size, kernel_id=0, layer_id=0):
        self.launch_func(self.instance_id*self.kernel_num + kernel_id, batch_size, layer_id)
        if self.profiler_tensor is not None:
            from .profiler_persistent import export_to_perfetto_trace
            
            if self.trace_name:
                trace_name = self.trace_name + ".perfetto-trace"
            else:
                trace_name = f"megakernel_{self.mpi_rank}.perfetto-trace"

            export_to_perfetto_trace(
                self.profiler_tensor, trace_name
            )

    def __del__(self):
        if not self.__finalized__:
            self.finalize()

    def finalize(self):
        assert not self.__finalized__
        if self._is_compiled:
            self.finalize_func(self.instance_id)
        self.__finalized__ = True
