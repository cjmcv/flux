
import argparse
import time
from typing import Optional, Any
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import xop
import xop.util as xutil

GEMM_COMM_ENABLE_CUDA_GRAPH = 1

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

def init_seed(seed=0):
    os.environ["NCCL_DEBUG"] = os.getenv("NCCL_DEBUG", "ERROR")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + seed)
    torch.cuda.manual_seed_all(3 + seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + seed)
    random.seed(3 + seed)

def matmul_int8(a, b):
    """
    torch._int_mm requires A.size(0) needs to be greater than 16
    """
    M, _ = a.shape
    if M <= 16:
        return torch._int_mm(torch.nn.functional.pad(a, (0, 0, 0, 32 - M)), b)[:M, :]
    return torch._int_mm(a, b)

def perf_torch(
    rank: int,
    group: ProcessGroup,
    inputs: list[torch.Tensor],
    weights: list[torch.Tensor],
    bias: Optional[torch.Tensor],
    warmup_iters: int,
    iters: int,
    problem_cnt: int,
    output_dtype: torch.dtype,
):
    m = inputs[0].size(0)
    n = weights[0].size(0)
    output = torch.empty([m, n], dtype=output_dtype, device=inputs[0].device, requires_grad=False)
        
    # nccl is not support cuda graph, we should use pynccl instead !
    mode = 2
    if mode == 0:
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph):
                problem_idx = 0
                torch.nn.functional.linear(inputs[problem_idx], weights[problem_idx], bias, out=output)#
                dist.all_reduce(output, group=group)
                
        def fn(iter_id):
            graph.replay()
            return output
    elif mode == 1:
        def fn(iter_id):
            problem_idx = iter_id%problem_cnt
            torch.nn.functional.linear(inputs[problem_idx], weights[problem_idx], bias, out=output)#
            dist.all_reduce(output, group=group)
            return output
    elif mode == 2:
        op = xop.GemmNormal(
            input_dtype=inputs[0].dtype,
            output_dtype=output_dtype,
            transpose_weight=False
        )
        def fn(iter_id):
            problem_idx = iter_id % problem_cnt
            op.forward(
                inputs[problem_idx],
                weights[problem_idx],
                output=output,
                bias=bias,
                input_scale=None,
                weight_scale=None,
                output_scale=None,
                tuning = None,
                fast_accum=False,
            )
            dist.all_reduce(output, group=group)
            return output

    return xutil.perf_gemm(warmup_iters, iters, "torch", fn)

def perf_xop(
    rank: int,
    group: ProcessGroup,
    inputs: list[torch.Tensor],
    weights: list[torch.Tensor],
    bias: Optional[torch.Tensor],
    inputs_scale: Optional[torch.Tensor],
    weights_scale: Optional[torch.Tensor],
    warmup_iters: int, 
    iters: int,
    problem_cnt: int,
    output_dtype: torch.dtype,
    fast_accum: bool,
):
    transpose_weight = False
    m = inputs[0].size(0)
    n = weights[0].size(0)

    output = torch.empty([m, n], dtype=output_dtype, device=inputs[0].device, requires_grad=False)
    op = xop.GemmCommRs(
        input_dtype=inputs[0].dtype,
        output_dtype=output_dtype,
        transpose_weight=transpose_weight,
        group=group,
        rank=rank,
    )
    
    if GEMM_COMM_ENABLE_CUDA_GRAPH:
        problem_idx = 0
        def forward_fn(problem_idx):
            op.forward(inputs[problem_idx],
                weights[problem_idx],
                output=output,
                bias=bias,
                input_scale=inputs_scale[problem_idx],
                weight_scale=weights_scale[problem_idx],
                output_scale=None,
                tuning = None,
                fast_accum=fast_accum,
            )
            
        # pre allocate workspace for cuda graph
        forward_fn(problem_idx)
        
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream): # , op.ar.capture():
            with torch.cuda.graph(graph):
                problem_idx = 0
                forward_fn(problem_idx)
                
        def fn(iter_id):
            graph.replay()
            return output
    else:       
        def fn(iter_id):
            problem_idx = iter_id % problem_cnt
            op.forward(
                inputs[problem_idx],
                weights[problem_idx],
                output=output,
                bias=bias,
                input_scale=inputs_scale[problem_idx],
                weight_scale=weights_scale[problem_idx],
                output_scale=None,
                tuning = None,
                fast_accum=fast_accum,
            )
            # print("bias:", bias, iter_id)
            # print("inputs[problem_idx]:", inputs[problem_idx])
            # print("weights[problem_idx]:", weights[problem_idx])
            # print("inputs_scale[problem_idx]:", inputs_scale[problem_idx])
            # print("weights_scale[problem_idx]:", weights_scale[problem_idx])
            # print("output:", output)
            return output
    return xutil.perf_gemm(warmup_iters, iters, "xop", fn)

# return atol, rtol
def get_allclose_threshold(args, k, world_size):
    # print("aaa", DTYPE_MAP[args.dtype], args.dtype, torch.float8_e4m3fn)
    if (args.quant_bits == 8):
        atol = 2e-1*np.sqrt(k)
        rtol = 2e-2
    elif (args.quant_bits == 4):
        atol = 2e-1*np.sqrt(k)
        rtol = 2e-2
    elif (args.dtype == "float8_e4m3fn" or args.dtype == "float8_e5m2"):
        atol = 2e-1*np.sqrt(k)
        rtol = 2e-2
    elif (args.output_dtype == "s8" or args.output_dtype == "s32"):
        atol = 0
        rtol = 0
    else:
        atol = 1e-2*np.sqrt(k)
        rtol = 2e-2
        
    return atol*world_size, rtol*world_size

THRESHOLD_MAP = {
    torch.float16: 10,  # 1e-1,
    torch.bfloat16: 2e-2,
    torch.float8_e4m3fn: 2e-2,
    torch.float8_e5m2: 2e-2,
    torch.int8: 0,
    torch.int32: 0,
}

def run(world_size, rank, M, args, xop_group, nccl_group, xop_perf, torch_perf):

    dtype = DTYPE_MAP[args.dtype]
    output_dtype = DTYPE_MAP[args.output_dtype]

    #
    N = args.N
    K = args.K
    cache_size = 100 * 1024 * 1024 # 100MB 
    total_bytes = (M*K + K*N) * torch.finfo(dtype).bits // 8 # + M*N

    problem_count = 5 # 1 + int((3 * cache_size) / total_bytes)
    # print("problem_count", problem_count, cache_size, total_bytes)
    #
    inputs = []
    weights = []
    inputs_scale = []
    weights_scale = []

    for i in range(problem_count):
        inputs.append(xutil.rand_tensor((M, K), dtype=dtype))
        weights.append(xutil.rand_tensor((N, K), dtype=dtype))
        # if (rank == 0):
        #     inputs.append(torch.zeros((M, K), dtype=dtype).cuda())
        #     weights.append(torch.zeros((N, K), dtype=dtype).cuda())
        # else:
        # inputs.append(torch.ones((M, K), dtype=dtype).cuda())
        # weights.append(torch.ones((N, K), dtype=dtype).cuda())
        inputs_scale.append(None)
        weights_scale.append(None)

    bias = None
    if args.has_bias:
        # bias = xutil.rand_tensor((N), output_dtype)
        bias = torch.ones((N), dtype=output_dtype).cuda() * 12

    perf_result_xop = perf_xop(
        rank, xop_group,
        inputs,
        weights,
        bias,
        inputs_scale,
        weights_scale,
        args.warmup_iters,
        args.iters,
        problem_count, 
        output_dtype,
        args.fast_accum,
    )
    
    perf_result_torch = perf_torch(
        rank, nccl_group,
        inputs,
        weights,
        bias,
        args.warmup_iters,
        args.iters,
        problem_count,
        output_dtype,
    )

    if args.show_ms:
        xop_perf.append(perf_result_xop.gemm_time_ms)
        torch_perf.append(perf_result_torch.gemm_time_ms)
    else:
        xop_perf.append(xutil.calculate_tflops(M,N,K, perf_result_xop.gemm_time_ms))
        torch_perf.append(xutil.calculate_tflops(M,N,K, perf_result_torch.gemm_time_ms))

    print(perf_result_torch)
    print(perf_result_xop)

    xop_output = perf_result_xop.output
    torch_output = perf_result_torch.output
    print(xop_output.dtype, torch_output.dtype)

    # torch.set_printoptions(threshold=float('inf'))
    print("xop_output: ", xop_output)
    print("torch_output: ", torch_output)
    # is_bitwise_match = xop.bitwise_check(xop_output, torch_output)
    # print("is bitwise match: ", is_bitwise_match)
    atol, rtol = get_allclose_threshold(args, K, world_size)
    # print(atol, rtol)
    xutil.torch_allclose(xop_output, torch_output, atol=atol, rtol=rtol)


def run_worker(world_size, rank, port, M, args, xop_perf, torch_perf):
    device = torch.device(f"cuda:{rank + xop.ALLREDUCE_GPUID_OFFSET}")
    torch.cuda.set_device(device)
    
    distributed_init_method = f"tcp://localhost:{port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    nccl_group = dist.group.WORLD
    xop_group = torch.distributed.new_group(list(range(world_size)), backend="gloo")
    
    exponent = args.M  # 65536: 17
    run(world_size, rank, 1, args, xop_group, nccl_group, xop_perf, torch_perf)
    for m in range(1, exponent):
        m = 2**m
        run(world_size, rank, m, args, xop_group, nccl_group, xop_perf, torch_perf)
        
    dist.barrier(group=nccl_group)
    dist.destroy_process_group(group=nccl_group)    
    
    #################  
    # plot
    plot_x_value = [1] + list(2**x for x in list(range(1, exponent)))
    plot_x = range(len(plot_x_value))
    plt.xticks(plot_x, plot_x_value, rotation=45)

    print(f"xop_perf_rank{rank}:{xop_perf}")
    print(f"torch_perf_rank{rank}:{torch_perf}")
    
    plt.plot(plot_x, xop_perf, label='xop_fused_gemmcomm', marker='o', markersize=3)
    plt.plot(plot_x, torch_perf, label='xop_gemm+nccl', marker='s', markersize=3)
    
    # plt.ylim(bottom=0)  # 

    title = f'perf-N{args.N}-K{args.K}-rank{rank}'
    plt.title(title)
    plt.xlabel('m_size')
    if args.show_ms:
        plt.ylabel('ms')
    else:
        plt.ylabel('tflops')

    plt.legend()
    plt.grid(True)

    # plt.xticks(plot_x)
    plt.savefig(title)
    plt.show()
    
    
def multi_process_parallel(
    world_size: int, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    port = 12355
    for i in range(world_size):
        proc_args = (world_size, i, port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.daemon = True
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert (
            procs[i].exitcode == 0
        ), f"Process {i} failed with exit code {procs[i].exitcode}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show_ms", default=False, action="store_true", help="whether to print time or tflops."
    )
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--quant_bits", default=-1, type=int, help="whether to use GemmQuant.")
    parser.add_argument("--step", default=5, type=int, help="m step")
    parser.add_argument("--warmup_iters", default=10, type=int, help="perf warmup iterations")
    parser.add_argument("--iters", default=20, type=int, help="perf iterations")
    parser.add_argument(
        "--dtype",
        default="bfloat16", # float16, float8_e4m3fn
        type=str,
        choices=list(DTYPE_MAP.keys()),
    )
    parser.add_argument(
        "--output_dtype",
        default="bfloat16", # float16
        type=str,
        help="allowed data type:: bfloat16,float16,s32.",
    )
    parser.add_argument(
        "--fast_accum", default=False, action="store_true", help="whether to use fp16 accum"
    )
    parser.add_argument(
        "--has_bias", default=False, action="store_true", help="whether to add bias"
    )
    return parser.parse_args()

# python3 tools/gemm/test_gemm_comm.py --has_bias 12 4096 4096 --show_ms
if __name__ == "__main__":
    init_seed()
    args = parse_args()

    xop_perf = []
    torch_perf = []
    
    rank = 0
    print(f"M: {args.M}, N: {args.N}, K: {args.K}")

    world_sizes = [2] # [2,4,8]
    for world_size in world_sizes:
        available_gpus = torch.cuda.device_count()
        if world_size > available_gpus:
            print(
                f"Skipping world_size={world_size}, requires {world_size} GPUs, found {available_gpus}"
            )
            continue

        print(f"Running test for world_size={world_size}")
        multi_process_parallel(
            world_size, run_worker, target_args=(args.M, args, xop_perf, torch_perf)
        )
        print(f"custom allreduce tp = {world_size}: OK")
        