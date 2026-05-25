import os
import math
from enum import IntEnum
import itertools
from collections.abc import Iterable
from typing import ParamSpec, TypeVar, Literal, Any
import concurrent.futures
from tqdm.auto import tqdm

from tilelang.jit.kernel import JITKernel
# from tilelang.language.v2 import PrimFunc
# from tilelang.utils.profiler import do_bench
from tvm.target import Target

import json
from pathlib import Path

# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from concurrent.futures import ThreadPoolExecutor
import torch
import tvm
from tvm.tir import stmt_functor, Block, For, PrimFunc
from tvm.tir.stmt_functor import ir_transform
import tilelang
import tilelang.language as T

from xop.ops.dsl.pkt_util import TestUtil, TorchRef
from xop.ops.dsl.micro_config import get_arch, get_target_str, is_megakernel_enabled, is_enable_profiling
import xop.util as xutil

class HparamSelectMode(IntEnum):
    HEURISTIC = 0
    TUNING = 1
    TUNED = 2
    SPECIFY = 3


def get_artifact(kernel):
    with tvm.transform.PassContext(opt_level=3, config=kernel.pass_configs), kernel.target:
        artifact = tilelang.lower(
            kernel.prim_func,
            target=kernel.target,
            target_host=kernel.target_host,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return artifact

def get_launch_info(artifact):        
    infos = []
    for g_var, func in artifact.device_mod.functions.items():
        grid_dim = {"blockIdx.x": 1, "blockIdx.y": 1, "blockIdx.z": 1}
        block_dim = {"threadIdx.x": 1, "threadIdx.y": 1, "threadIdx.z": 1}
        dynamic_smem_buf = 0
        use_cooperative_groups = 0
    
        attrs = func.attrs
        if "use_cooperative_groups" in attrs:
            use_cooperative_groups = attrs["use_cooperative_groups"]
        if "dyn_shared_memory_buf" in attrs:
            dynamic_smem_buf = int(attrs["dyn_shared_memory_buf"])
        if "thread_extent" in attrs:
            # Extract block and grid sizes from thread extents
            thread_extent = attrs["thread_extent"]
            for tag, extent in thread_extent.items():
                if tag in grid_dim:
                    grid_dim[tag] = extent
                elif tag in block_dim:
                    block_dim[tag] = extent
        infos.append((grid_dim, block_dim, dynamic_smem_buf, use_cooperative_groups))
    return infos
    
def create_dispatch_func(cuda_src_warpper, code, function_informations):
    from tilelang.jit.adapter.wrapper import L2_PERSISTENT_MAP_CREATE_HANDLE, L2_PERSISTENT_MAP_RESET_HANDLE, PREDEF_HOST_FUNC, match_declare_kernel, parse_function_call_args
    # Extract the set of dynamic symbolic names used in the primary function
    dynamic_symbolic_set = cuda_src_warpper.get_dynamic_symbolic_set(cuda_src_warpper.prim_func)

    function_args = []

    # Collect function arguments based on primary function's parameters and buffer mappings
    # QA(@lei): Why not use device_mod.params?
    # device func lack buffer map (to convert buffer handle to buffer)
    for param in cuda_src_warpper.prim_func.params:
        if param in cuda_src_warpper.prim_func.buffer_map:
            buffer = cuda_src_warpper.prim_func.buffer_map[param]
            function_args.append(
                {
                    "name": buffer.data.name,
                    "type": cuda_src_warpper._lookup_type(buffer.dtype) + "* __restrict__",
                }
            )
        elif isinstance(param, tvm.tir.Var):
            function_args.append({"name": param.name, "type": cuda_src_warpper._lookup_type(param.dtype)})
        else:
            raise ValueError(f"Parameter {param} is not in the buffer map of the primary function.")

    # Add dynamic symbols as integer arguments
    for dyn_sym, dyn_sym_dtype in dynamic_symbolic_set:
        if dyn_sym not in [arg["name"] for arg in function_args]:
            function_args.append({"name": dyn_sym, "type": cuda_src_warpper._lookup_type(dyn_sym_dtype)})
    
    # function_args.append(cuda_src_warpper.get_stream_type())

    has_l2_persistent_map = False
    for function_name, _ in function_informations.items():
        if function_name in cuda_src_warpper.l2_persistent_map:
            has_l2_persistent_map = True
            break

    kernel_launch_code = """"""
    if has_l2_persistent_map:
        kernel_launch_code += L2_PERSISTENT_MAP_CREATE_HANDLE
    desc_name_map: dict[str, str] = {}
    desc_name_var_map: dict[str, tvm.tir.Var] = {}
    for function_name, function_info in function_informations.items():
        block_info = function_info["block_info"]
        grid_info = function_info["grid_info"]
        dynamic_smem_buf = function_info["dynamic_smem_buf"]
        function_params = function_info["function_params"]

        # Find the location of the global kernel function in the code
        index = match_declare_kernel(code, function_name + "(")

        # Analyze the function declaration to prepare for argument extraction
        declaration = code[index:].split(";")[0]

        # Identify the start of the function body to insert arguments
        index = code.index("{", index)

        block_str = (
            f"dim3({cuda_src_warpper._pythonic_expr(block_info[0])}, {cuda_src_warpper._pythonic_expr(block_info[1])}, {cuda_src_warpper._pythonic_expr(block_info[2])})"
        )
        grid_str = (
            f"dim3({cuda_src_warpper._pythonic_expr(grid_info[0])}, {cuda_src_warpper._pythonic_expr(grid_info[1])}, {cuda_src_warpper._pythonic_expr(grid_info[2])})"
        )
        smem_str = 0 if dynamic_smem_buf is None else dynamic_smem_buf
        init_l2_persistent_map = cuda_src_warpper.generate_l2_persistent_map(function_name)
        kernel_launch_code += init_l2_persistent_map

        if cuda_src_warpper.use_cooperative_groups[function_name]:
            args_list = parse_function_call_args(declaration, function_args, function_params, desc_name_map, desc_name_var_map)
            assert len(function_params) == len(args_list), (
                f"Function {function_name} has {len(function_params)} parameters, but {len(args_list)} arguments"
            )
            args_array = [f"(void*)&{arg}" for arg in args_list]
            call_args = f"\tvoid* {function_name}_args[] = {{{', '.join(args_array)}}};\n"
            kernel_launch_code += call_args
            # Using cudaLaunchCooperativeKernel to launch the kernel
            # kernel_launch_code += "\tTILELANG_CHECK(cudaLaunchCooperativeKernel((void*){}, {}, {}, {}, {}, stream));\n".format(
            #     function_name, grid_str, block_str, function_name + "_args", smem_str
            # )
        else:
            args_list = parse_function_call_args(declaration, function_args, function_params, desc_name_map, desc_name_var_map)
            assert len(function_params) == len(args_list), (
                f"Function {function_name} has {len(function_params)} parameters, but {len(args_list)} arguments"
            )
            call_args = ", ".join(args_list)
            kernel_launch_code += f"\t{function_name}<<<{grid_str}, {block_str}, {smem_str}, stream>>>({call_args});\n"
            # kernel_launch_code += f'\tTILELANG_CHECK_LAST_ERROR("{function_name}");\n'
        if has_l2_persistent_map:
            kernel_launch_code += L2_PERSISTENT_MAP_RESET_HANDLE

    # Add output descriptor pointers for TMA descriptors        
    desc_output_code = ""
    if len(desc_name_var_map) != 0:
        for var_name in desc_name_var_map:
            function_args.append({"name": f"out_{var_name}", "type": "CUtensorMap*"})
        function_args.append({"name": f"to_device", "type": "bool"})
        
        desc_output_code += f"\tif (to_device) {{\n"
        for var_name in desc_name_var_map:
            desc_output_code += f"\t\tcudaMemcpy(out_{var_name}, &{var_name}, sizeof(CUtensorMap), cudaMemcpyHostToDevice);\n"                
        desc_output_code += f"\t}} else {{\n"
        for var_name in desc_name_var_map:
            desc_output_code += f"\t\t*out_{var_name} = {var_name};\n"
        desc_output_code += f"\t}}\n"
    
    # Add launch func    
    launch_info = f"#define LAUNCH_INFO {grid_str}, {block_str}, {smem_str}, stream"
        
        # launch_args = []
        # for var_name in desc_name_var_map:
        #     launch_args.append({"name": f"{var_name}", "type": "CUtensorMap"})
        # launch_args.append({"name": f"stream=cudaStreamDefault", "type": "cudaStream_t"})
        
        # def_args = ", ".join([f"{arg['type']} {arg['name']}" for arg in launch_args])
        # launch_func = PREDEF_HOST_FUNC.format(def_args, kernel_launch_code)
        
    # print("function_args", function_args)        
    # Format the function arguments for declaration
    def_args = ", ".join([f"{arg['type']} {arg['name']}" for arg in function_args])
    
    init_tma_descriptor_args = cuda_src_warpper.generate_tma_descriptor_args(desc_name_map, desc_name_var_map)
    # hardcode: R因“Warning: Layout inference failed for buffer R_sh. The buffer cannot be inferred with current layout inference rules”，被更名为“R_1”，需要改回来
    # init_tma_descriptor_args = init_tma_descriptor_args.replace("void *R_desc_globalAddress= R_1", "void *R_desc_globalAddress= R")
    # print("init_tma_descriptor_args", init_tma_descriptor_args)
    # kernel_launch_code = init_tma_descriptor_args + desc_output_code + "\t//" +kernel_launch_code

    # Wrap the kernel dispatch logic in an external C function
    host_func = PREDEF_HOST_FUNC.format(def_args, init_tma_descriptor_args + desc_output_code)
    return host_func + launch_info

def update_lib_code(cuda_src_warpper, code: str):
    # Get the function names
    function_names = cuda_src_warpper.function_names
    # Get the CUDA initialization function
    init_func = cuda_src_warpper.get_init_func()

    # Organize function information for code generation
    function_informations = {}
    for function_name in function_names:
        # Do not update function with dispatch host function
        if (function_name not in cuda_src_warpper.block_info) or (function_name not in cuda_src_warpper.grid_info):
            continue
        assert function_name in cuda_src_warpper.device_mod, f"Function {function_name} not found in device module"
        device_func = cuda_src_warpper.device_mod[function_name]
        kernel_params_cnt = len(device_func.params)
        function_params: list[str] = None

        def visitor(node, fn=function_name, param_cnt=kernel_params_cnt):
            nonlocal function_params
            if isinstance(node, tvm.tir.Call):
                if not (hasattr(node, "op") and node.op == tvm.ir.Op.get("tir.tvm_call_packed")):
                    return
                args = node.args
                if not args or args[0] != fn:
                    return
                if len(args) < 1 + param_cnt:
                    raise AssertionError("tvm_call_packed should have at least 1 argument and match device function parameters")
                function_params = args[1 : 1 + param_cnt]

        stmt_functor.post_order_visit(cuda_src_warpper.host_func.body, visitor)
        assert function_params is not None, "function_params should not be None"

        function_informations[function_name] = {
            "function_name": function_name,
            "block_info": cuda_src_warpper.block_info[function_name],
            "grid_info": cuda_src_warpper.grid_info[function_name],
            "dynamic_smem_buf": cuda_src_warpper.dynamic_smem_buf[function_name],
            "function_params": function_params,
        }

    # Create the host function wrapper for the CUDA kernel
    host_func = create_dispatch_func(cuda_src_warpper, code, function_informations)
    # Combine the source, initialization function, and host function to form the complete library code
    lib_code = cuda_src_warpper.source + init_func + host_func
    # return lib_code
    return host_func
    
def get_dispatch_source(kernel, artifact, kernel_only: bool = True) -> str:
    from tilelang.jit.adapter.wrapper import TLWrapper, TLCUDASourceWrapper
    if isinstance(kernel.prim_func, PrimFunc):
        ir_module = tvm.IRModule({kernel.prim_func.attrs["global_symbol"]: kernel.prim_func})
    else:
        ir_module = kernel.prim_func
    wrapper = TLWrapper(kernel.target)
    wrapper.assign_optimized_module(ir_module)
    wrapper.assign_pass_configs(kernel.pass_configs)
    wrapper.assign_host_module(artifact.host_mod)
    wrapper.assign_device_module(artifact.device_mod)

    wrapper_o = TLCUDASourceWrapper(
        scheduled_ir_module=wrapper.scheduled_ir_module,
        source=artifact.kernel_source,
        target=wrapper.target,
        device_mod=wrapper.device_mod,
        host_mod=wrapper.host_mod,
        pass_configs=wrapper.pass_configs,
    )
    return update_lib_code(wrapper_o, artifact.kernel_source)
    # return wrapper.wrap(my_artifact.kernel_source)
    
# print("artifact: ", artifact)
# T.func_attr({"calling_conv": 2, "dyn_shared_memory_buf": 49152, "target": T.target({"arch": "sm_89", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": 304, "blockIdx.y": 1, "threadIdx.x": 128, "threadIdx.y": 1, "threadIdx.z": 1}, "tir.is_global_func": T.bool(True), "tir.kernel_launch_params": ["blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tir.use_dyn_shared_memory"], "tir.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 1]})

# analyzer = LaunchInfoAnalyzer(kernel.prim_func)
# analyzer.get_threads_layout()
# print(analyzer.grid_dim)
class LaunchInfoAnalyzer:
    def __init__(self, fn: PrimFunc):
        self.prim_func = fn
        self.ir_module = tvm.IRModule({"main": fn})
        self.grid_dim = {"blockIdx.x": 1, "blockIdx.y": 1, "blockIdx.z": 1}
        self.block_dim = {"threadIdx.x": 1, "threadIdx.y": 1, "threadIdx.z": 1}
        self.dyn_shared_memory_buf = 0
        self.loop_stack = []
        
    def get_threads_layout(self):
        """
        Traverse and transform the IR module to extract performance-related information.
        Returns:
            self: The LaunchInfoAnalyzer instance.
        """

        def _ftransform(f, mod, ctx):
            # Initialize the set of global buffers
            self.global_buffers = set(f.buffer_map.values())

            def _pre_visit(stmt):
                """
                Pre-visit callback for IR nodes.
                Args:
                    stmt: The current IR node being visited.
                """
                # print(type(stmt), stmt, "\n\n")
                if isinstance(stmt, tvm.tir.AttrStmt):
                    # Handle thread extent attributes
                    # print(stmt.attr_key)
                    if stmt.attr_key == "thread_extent":
                        iter_var = stmt.node
                        thread_tag = iter_var.thread_tag
                        if thread_tag in self.grid_dim:
                            extent = stmt.value.value if hasattr(stmt.value, "value") else stmt.value
                            self.grid_dim[thread_tag] = extent
                        elif thread_tag in self.block_dim:
                            extent = stmt.value.value if hasattr(stmt.value, "value") else stmt.value
                            self.block_dim[thread_tag] = extent
                elif isinstance(stmt, tvm.tir.For):
                    # Push loop extent onto the stack
                    self.loop_stack.append(stmt.extent)
                # elif isinstance(stmt, tvm.tir.Evaluate):
                #     # Handle Evaluate nodes containing calls
                #     value = stmt.value
                #     if isinstance(value, tvm.tir.Call):
                #         if value.op.name == "tl.copy":
                #             self._analyze_copy(value)
                #         elif value.op.name == "tl.gemm":
                #             self._analyze_gemm(value)
                return None

            def _post_visit(stmt):
                """
                Post-visit callback for IR nodes.
                Args:
                    stmt: The current IR node being visited.
                """
                if isinstance(stmt, tvm.tir.For) and self.loop_stack:
                    self.loop_stack.pop()
                return None

            # Use IR transformation to traverse and modify the function body
            new_body = ir_transform(f.body, _pre_visit, _post_visit)
            return f.with_body(new_body)

        # Apply the custom PrimFunc pass
        tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)(self.ir_module)
        return self
    
    def get_smem_bytes(self):
        smem_bytes = 0
        num_stages = 1
        
        def collect(node):
            nonlocal smem_bytes
            nonlocal num_stages
            if isinstance(node, Block):
                for buf in node.alloc_buffers:
                    scope = buf.scope()
                    if str(scope).startswith("shared"):
                        numel = 1
                        for s in buf.shape:
                            numel *= int(s)
                        smem_bytes += numel * (buf.dtype.bits // 8)
            if isinstance(node, For):
                num_stages = node.annotations.get("num_stages", 1)
                    
        stmt_functor.post_order_visit(self.prim_func.body, collect)
        return smem_bytes*num_stages
    
class BaseMicroKernel:
    def __init__(self):
        self.dsl_home = os.getenv("DSL_HOME", default=None)
        if self.dsl_home is None:
            raise EnvironmentError("The environment variable DSL_HOME is not set.")
        # prop = torch.cuda.get_device_properties(0)
        # str(prop.major) + str(prop.minor)
        self.base_path = self.dsl_home + "/autogen/" + get_arch() + "/"
        target_dir = Path(self.base_path)
        target_dir.mkdir(parents=True, exist_ok=True)

    def replace_header(self, text: str, src_target: str, num_split: int, dst_target: str) -> str:
        lines = text.splitlines(True)
        processed_lines = []
        target_count = 0
        matched_count = 0
            
        if num_split > 1:
            skip_count = 2
        else:
            skip_count = 1
            
        for line in lines:
            if src_target in line:
                matched_count += 1
                if matched_count == 1:
                    processed_lines.append("namespace kernel {\n")  # 第一次命中函数名时，加入命名空间的头
                if matched_count <= skip_count:
                    continue
                
                target_count += 1
                if num_split > 1:
                    processed_lines.append(dst_target.replace('<kernel_id>', str(target_count-1))) # 更换kernel_id号，从下标0开始
                else:
                    processed_lines.append(dst_target)
            else:
                processed_lines.append(line)
        code = "".join(processed_lines)
        return code + "\n} // kernel"

    def write_tuned_hparams_to_json(self, latency_hparams_list, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for latency, latency_ref, similarity, hparams, idx in latency_hparams_list:
                single_config = {
                    "latency": latency,
                    "latency_ref": latency_ref,
                    "similarity": similarity,
                    "hparams": hparams,
                    "idx": idx,
                }
                json_line = json.dumps(single_config, ensure_ascii=False, separators=(",", ":"))
                f.write(json_line + "\n")
        
        print(f"Save: {file_path}")

    def read_tuned_hparams_from_json(self, save_path):
        latency_hparams_list = []
        read_file_path = save_path+f"_atuned.json"
            
        try:
            with open(read_file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line:
                        continue
                    try:
                        json_data = json.loads(line)
                        tuple_item = (
                            json_data['latency'],
                            json_data['latency_ref'],
                            json_data['similarity'],
                            json_data['hparams'],
                            json_data['idx']
                        )
                        latency_hparams_list.append(tuple_item)
                    except json.JSONDecodeError as e:
                        print(f"Line {line_num}: Failed to parse JSON: {e}, content: {line}")
        except FileNotFoundError:
            print(f"Error: File {read_file_path} not found.")
        except Exception as e:
            print(f"Unknown error during file reading: {e}")
            
        # print(latency_hparams_list)
        return latency_hparams_list

    def _calc_sim(self, x, y):
        x, y = x.data.double(), y.data.double()
        denominator = (x * x + y * y).sum()
        if denominator == 0:
            return -1
        sim = 2 * (x * y).sum() / denominator
        return sim
        
    def _run_profile(self, kernel, strategy, hparams):
        warnup_iters = 500
        test_iters = 2000
        problem_cnt = 10
        
        test_data_list = []
        for _ in range(problem_cnt):
            test_data = strategy.gen_test_data(hparams)
            test_data_list.append(test_data)
        
        ref_func = strategy.get_torch_ref()
        def target_run(iter):
            test_data = test_data_list[iter % len(test_data_list)]
            return kernel(*test_data)
        def ref_run(iter):
            test_data = test_data_list[iter % len(test_data_list)]
            return ref_func(*test_data)
        
        target_result = target_run(0)
        ref_result = ref_run(0)
        if isinstance(target_result, list):
            sim = self._calc_sim(target_result[0], ref_result[0])
        else:
            sim = self._calc_sim(target_result, ref_result)
        
        perf_result_xop = xutil.perf_gemm(warmup_iters=warnup_iters, iters=test_iters, name="target", fn=target_run)
        perf_result_torch = xutil.perf_gemm(warmup_iters=warnup_iters, iters=test_iters, name="torch", fn=ref_run)
        # do_bench(lambda: ref_run(), warmup=warnup_iter*2, rep=test_iter*2, backend="event") # extra warnup
        latency = perf_result_xop.gemm_time_ms #do_bench(lambda: target_run(), warmup=warnup_iter, rep=test_iter, backend="event") # cupti
        latency_ref = perf_result_torch.gemm_time_ms #do_bench(lambda: ref_run(), warmup=warnup_iter, rep=test_iter, backend="event")
        return float(f"{latency:.5f}"), float(f"{latency_ref:.5f}"), float(f"{sim:.5f}")
        
    def run_tuning(self, strategy, save_path):
        tuned_file_path = save_path+f"_atuned.json"
        print(f"Start tuning with a total of {len(strategy.hparam_space)} schemes.")

        latency_hparams_list = []
        
        is_compile_parallel = True
        # # compile
        if (is_compile_parallel):
            num_workers = 8
            with concurrent.futures.ThreadPoolExecutor(num_workers, "tl-par-comp") as executor:
                futures = []
                future_map = {}
                for idx, hparams in enumerate(strategy.hparam_space):
                    future = executor.submit(strategy.get_kernel, selected_hparams=hparams)
                    future_map[future] = idx
                    futures.append(future)
                kernels = [... for _ in futures]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Parallel Compiling",
                ):
                    idx = future_map[future]
                    kernels[idx] = future.result()
    
        # profile
        for idx, hparams in enumerate(strategy.hparam_space):
            latency = None
            latency_ref = -1
            similarity = -1
            try:
                if (is_compile_parallel):
                    kernel = kernels[idx]
                else:
                    kernel = strategy.get_kernel(hparams)

                latency, latency_ref, similarity = self._run_profile(kernel, strategy, hparams)
                # profiler = kernel.get_profiler()
                # latency = round(profiler.do_bench(backend="cupti"), 5)
                status = "success"
                if (latency == 0):
                    status = "sth wrong with the latency"
            except Exception as e:
                status = f"{e}"
                
            if status == "success":
                latency_hparams_list.append((latency, latency_ref, similarity, hparams, idx))
            print(f">>>>> tuning({idx}-{status}): {latency} vs ref-{latency_ref} -> {hparams} // similarity: {similarity}")
            
        latency_hparams_list.sort(key=lambda x: x[0])
        self.write_tuned_hparams_to_json(latency_hparams_list, tuned_file_path)
        best_latency, _, _, selected_hparams, idx = latency_hparams_list[0]
        print(f"[Tuning] the best result: {best_latency} ms -> {selected_hparams}")
        
        return latency_hparams_list
    
    def auto_get_kernel(self, get_source_func, strategy, mode: HparamSelectMode):
        save_path = self.base_path+f"/{strategy.name}/{strategy.name}"
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path, exist_ok=True)
        
        if (mode == HparamSelectMode.TUNING):
            latency_hparams_list = self.run_tuning(strategy, save_path)
            # Save all tuned kernels.
            for i in range(len(latency_hparams_list)):
                latency, latency_ref, similarity, selected_hparams, idx = latency_hparams_list[i]
                kernel = strategy.get_kernel(selected_hparams)
                file_name = save_path+f"_top{i}.cuh"
                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(get_source_func(kernel, selected_hparams) + f"\n// latency: {latency} ms vs [ref-{latency_ref} sim-{similarity}], idx: {idx}")
            _, _, _, selected_hparams, selected_idx = latency_hparams_list[0]
        elif (mode == HparamSelectMode.TUNED):
            latency_hparams_list = self.read_tuned_hparams_from_json(save_path)
            _, _, _, selected_hparams, selected_idx = latency_hparams_list[0]
            print("[Tuned] selected_hparams: ", selected_hparams)
        elif (mode == HparamSelectMode.HEURISTIC):
            selected_idx = -1
            selected_hparams = strategy.get_heuristic_hparams()
            print("[Heuristic] selected_hparams: ", selected_hparams)
        elif (mode >= HparamSelectMode.SPECIFY):
            specified_idx = mode - HparamSelectMode.SPECIFY
            latency_hparams_list = self.read_tuned_hparams_from_json(save_path)
            _, _, _, selected_hparams, selected_idx = latency_hparams_list[specified_idx]
            # selected_hparams = strategy.hparam_space[id]
            print(f"[SPECIFY] selected_hparams[{specified_idx}]({latency}ms): {selected_hparams}")
            
        kernel = strategy.get_kernel(selected_hparams)
        kernel.config = selected_hparams
        if (is_enable_profiling()):
            latency, latency_ref, similarity = self._run_profile(kernel, strategy, selected_hparams)
        else:
            latency, latency_ref, similarity = 0,0,0
        # kernel.export_sources(kernel_path=save_path+f"_src.cuh")
        msg_suffix = f"latency: {latency} ms vs [ref-{latency_ref} sim-{similarity}], idx: {selected_idx}"
        with open(save_path+f".cuh", "w", encoding="utf-8") as f:
            f.write(get_source_func(kernel, selected_hparams) + f"\n// " + msg_suffix)
        print(f"selected: {selected_hparams}, " + msg_suffix)
        # print("0:", kernel.prim_func.attrs)
        # print("1:", kernel.adapter.params)
        # print("2:", kernel.adapter.func)
        # print("3:", kernel.config)
        # print("4:", kernel.prim_func)
        return kernel, save_path+f".cuh"