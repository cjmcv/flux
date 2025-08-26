
import argparse
import time
from typing import Optional

import torch

import xop
import xop.util as xutil

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch.distributed as dist
from torch.distributed import ProcessGroup

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
    inputs: list[torch.Tensor],
    weights: list[torch.Tensor],
    bias: Optional[torch.Tensor],
    warmup_iters: int,
    iters: int,
    problem_cnt: int,
    output_dtype: torch.dtype,
):
    alpha_scale = 1.0
    def fn(iter_id):
        problem_idx = iter_id%problem_cnt
        output = alpha_scale * torch.nn.functional.linear(inputs[problem_idx], weights[problem_idx], bias)#
        return output

    return xutil.perf_gemm(warmup_iters, iters, "torch", fn)

def perf_xop(
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

    world_size = 1
    rank = 0
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # device = torch.cuda.current_device()
    distributed_init_method = f"tcp://localhost:{12345}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    new_group = torch.distributed.new_group(list(range(1)), backend="gloo")
    op = xop.GemmCommRs(
        input_dtype=inputs[0].dtype,
        output_dtype=output_dtype,
        transpose_weight=transpose_weight,
        group=new_group,
        rank=rank,
    )
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
def get_allclose_threshold(args, k):
    # print("aaa", DTYPE_MAP[args.dtype], args.dtype, torch.float8_e4m3fn)
    if (args.quant_bits == 8):
        return 2e-1*np.sqrt(k), 2e-2
    if (args.quant_bits == 4):
        return 2e-1*np.sqrt(k), 2e-2
    if (args.dtype == "float8_e4m3fn" or args.dtype == "float8_e5m2"):
        return 2e-1*np.sqrt(k), 2e-2
    if (args.output_dtype == "s8" or args.output_dtype == "s32"):
        return 0, 0

    return 2e-2, 2e-2
    
THRESHOLD_MAP = {
    torch.float16: 10,  # 1e-1,
    torch.bfloat16: 2e-2,
    torch.float8_e4m3fn: 2e-2,
    torch.float8_e5m2: 2e-2,
    torch.int8: 0,
    torch.int32: 0,
}

def run(M, args, xop_perf, torch_perf):
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
        # inputs.append(xutil.rand_tensor((M, K), dtype=dtype))
        # weights.append(xutil.rand_tensor((N, K), dtype=dtype))
        inputs.append(torch.ones((M, K), dtype=dtype).cuda())
        weights.append(torch.ones((N, K), dtype=dtype).cuda())
        inputs_scale.append(None)
        weights_scale.append(None)

    bias = None
    if args.has_bias:
        # bias = xutil.rand_tensor((N), output_dtype)
        bias = torch.ones((N), dtype=output_dtype).cuda() * 12

    perf_result_xop = perf_xop(
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

    print(xop_output)
    # is_bitwise_match = xop.bitwise_check(xop_output, torch_output)
    # print("is bitwise match: ", is_bitwise_match)
    atol, rtol = get_allclose_threshold(args, K)
    # print(atol, rtol)
    xutil.torch_allclose(xop_output, torch_output, atol=atol, rtol=rtol)

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
    parser.add_argument("--warmup_iters", default=0, type=int, help="perf warmup iterations")
    parser.add_argument("--iters", default=1, type=int, help="perf iterations")
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

# python3 tools/gemm/test_gemm_comm.py 14 4096 4096 --quant_bits=8 --dtype=float16 --output_dtype=float16
# python3 tools/gemm/test_gemm_comm.py 14 4096 4096 --quant_bits=8
# python3 tools/gemm/test_gemm_comm.py 14 4096 4096 --show_ms
# python3 tools/gemm/test_gemm_comm.py 14 4096 4096 --dtype=float16
# python3 tools/gemm/test_gemm_comm.py 14 4096 4096 --dtype=float16 --has_bias 
# python3 tools/gemm/test_gemm_comm.py 14 4096 4096 --dtype=float8_e4m3fn
# python3 tools/gemm/test_gemm_comm.py 14 4096 4096 --dtype=float8_e4m3fn --fast_accum
# python3 tools/gemm/test_gemm_comm.py 14 4096 4096 --dtype=float8_e4m3fn --output_dtype=float16 --fast_accum
if __name__ == "__main__":
    init_seed()
    args = parse_args()

    xop_perf = []
    torch_perf = []
    
    print(f"M: {args.M}, N: {args.N}, K: {args.K}")
    run(args.M, args, xop_perf, torch_perf)

    # print(f"M: {1}, N: {args.N}, K: {args.K}")
    # run(1, args, xop_perf, torch_perf)

    # if 0:
    #     for m in range(2, args.M, args.step):
    #         print(f"M: {m}, N: {args.N}, K: {args.K}")
    #         run(m, args, xop_perf, torch_perf)
    #     plot_x = [1] + list(range(2, args.M, args.step))
    # else:
    #     exponent = args.M # 65536: 17
    #     for m in range(1, exponent):
    #         m = 2**m
    #         print(f"M: {m}, N: {args.N}, K: {args.K}")
    #         run(m, args, xop_perf, torch_perf)
        
    #     plot_x_value = [1] + list(2**x for x in list(range(1, exponent)))
    #     plot_x = range(len(plot_x_value))
    #     plt.xticks(plot_x, plot_x_value, rotation=45)

    # print("xop_perf:", xop_perf)
    # plt.plot(plot_x, xop_perf, label='xop', marker='o', markersize=3)
    # plt.plot(plot_x, torch_perf, label='torch', marker='s', markersize=3)
    
    # # plt.ylim(bottom=0)  # 

    # plt.title(f'perf-N{args.N}-K{args.K}')
    # plt.xlabel('m_size')
    # if args.show_ms:
    #     plt.ylabel('ms')
    # else:
    #     plt.ylabel('tflops')

    # plt.legend()
    # plt.grid(True)

    # # plt.xticks(plot_x)
    # plt.savefig('perf-N-{0}-K-{1}.png'.format(args.N, args.K))
    # plt.show()
