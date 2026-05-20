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
        self.base_path = self.dsl_home + "/demo/gen/" + get_arch() + "/"
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
        test_data = strategy.gen_test_data(hparams)
        ref_func = strategy.get_torch_ref()
        
        def target_run(iter):
            return kernel(*test_data)
        def ref_run(iter):
            return ref_func(*test_data)
        
        target_result = target_run()
        ref_result = ref_run()
        if isinstance(target_result, list):
            sim = self._calc_sim(target_result[0], ref_result[0])
        else:
            sim = self._calc_sim(target_result, ref_result)
        
        warnup_iters = 500
        test_iters = 100
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