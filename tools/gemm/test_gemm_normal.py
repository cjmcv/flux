
import argparse
import time
from typing import Optional

import torch

import xop
from xop.util import is_fp8_dtype

import os
import random
import numpy as np
import matplotlib.pyplot as plt

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

class PerfResult:
    def __init__(self, name: str, output: torch.Tensor, gemm_time_ms: float) -> None:
        self.name = name
        self.output = output
        self.gemm_time_ms = gemm_time_ms

    def __repr__(self) -> str:
        return f"{self.name}: gemm {self.gemm_time_ms:.3f} ms"

def perf_gemm(warmup_iters: int, iters: int, name: str, fn: callable):
    total_time = 0
    
    for i in range(warmup_iters + iters):
        if (i == warmup_iters):
            torch.cuda.synchronize()
            start = time.time()
        output = fn(i)

    torch.cuda.synchronize()
    end = time.time()
    total_time = end - start
    return PerfResult(name=name, output=output, gemm_time_ms=total_time / iters * 1000)

    # for i in range(warmup_iters):
    #     output = fn()
    # torch.cuda.synchronize()
    
    # start = time.time()
    # for i in range(iters):
    #     output = fn(i)
    # torch.cuda.synchronize()
    # end = time.time()
    # total_time = end - start
    # return PerfResult(name=name, output=output, gemm_time_ms=total_time / iters * 1000)


def perf_torch(
    inputs: list[torch.Tensor],
    weights: list[torch.Tensor],
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    is_fp8: bool,
    is_s8_dequant: bool,
    warmup_iters: int,
    iters: int,
    problem_cnt: int,
    output_dtype: torch.dtype,
):
    alpha_scale = 1.0
    if is_fp8:
        alpha_scale = input_scale * weight_scale
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(torch.bfloat16)
        for i in range(len(weights)):
            weights[i] = weights[i].to(torch.bfloat16)

    def fn(iter_id):
        problem_idx = iter_id%problem_cnt
        if is_s8_dequant:
            accum = matmul_int8(inputs[problem_idx], weights[problem_idx].t()).to(torch.float32)
            output = input_scale * weight_scale * accum
            output = output.to(torch.bfloat16)
            if bias is not None:
                output = output + bias
        elif inputs[problem_idx].dtype == torch.int8:
            output = matmul_int8(inputs[problem_idx], weights[problem_idx].t())
            if bias is not None:
                output = output + bias
        else:
            output = alpha_scale * torch.nn.functional.linear(inputs[problem_idx], weights[problem_idx], bias)
        return output

    return perf_gemm(warmup_iters, iters, "torch", fn)


def perf_xop(
    inputs: list[torch.Tensor],
    weights: list[torch.Tensor],
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    transpose_weight: bool,
    is_fp8: bool,
    is_s8_dequant: bool,
    warmup_iters: int, 
    iters: int,
    problem_cnt: int,
    output_dtype: torch.dtype,
):
    m = inputs[0].size(0)
    if transpose_weight:
        assert (
            is_fp8 == False and is_s8_dequant == False
        ), "FP8/S8 GEMM does not support transpose weight (RRR layout)"
        # weight = weight.t().contiguous()
        n = weights[0].size(1)
    else:
        n = weights[0].size(0)

    def _check_tensor_shape(tensor, shape):
        if not isinstance(tensor, torch.Tensor):
            return False
        if len(tensor.size()) != len(shape):
            return False
        for x, y in zip(list(tensor.size()), shape):
            if x != y:
                return False
        return True

    if is_s8_dequant:
        if not _check_tensor_shape(input_scale, (m, 1)):
            raise ValueError("input_scale's shape should be (m, 1) for S8 GEMM")
        if not _check_tensor_shape(weight_scale, (1, n)):
            raise ValueError("weight_scale's shape should be (1, n) for S8 GEMM")

    output = torch.empty([m, n], dtype=output_dtype, device=inputs[0].device, requires_grad=False)
    ## todo: remove below once moe fp8 gemm invoke get fixed
    op = xop.GemmNormal(
        input_dtype=inputs[0].dtype,
        output_dtype=output_dtype,
        transpose_weight=transpose_weight
    )
    def fn(iter_id):
        problem_idx = iter_id % problem_cnt
        op.forward(
            inputs[problem_idx],
            weights[problem_idx],
            output=output,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            tuning = None,
            fast_accum=False,
        )
        return output
    return perf_gemm(warmup_iters, iters, "xop", fn)


def rand_tensor(shape: list[int], dtype: torch.dtype):
    if dtype in [torch.int32, torch.int8]:
        return torch.randint(-127, 128, shape, dtype=dtype).cuda()
    elif is_fp8_dtype(dtype):
        data = torch.rand(shape, dtype=torch.bfloat16).cuda() / 10
        return data.to(dtype)
    else:
        return torch.rand(shape, dtype=dtype).cuda() / 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--step", default=5, type=int, help="m step")
    parser.add_argument("--warmup_iters", default=20, type=int, help="perf warmup iterations")
    parser.add_argument("--iters", default=500, type=int, help="perf iterations")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        type=str,
        choices=list(DTYPE_MAP.keys()),
    )
    parser.add_argument(
        "--output_dtype",
        default="",
        type=str,
        help="allowed data type:: bfloat16,float16,s32.",
    )
    parser.add_argument(
        "--has_bias", default=False, action="store_true", help="whether to add bias"
    )
    parser.add_argument(
        "--transpose_weight", default=False, action="store_true", help="whether to transpose weight"
    )

    return parser.parse_args()


THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
    torch.float8_e4m3fn: 2e-2,
    torch.float8_e5m2: 2e-2,
    torch.int8: 0,
    torch.int32: 0,
}

def run(M, args, xop_perf, torch_perf):
    #
    N = args.N
    K = args.K
    cache_size = 100 * 1024 * 1024 # 100MB 
    total_bytes = (M*K + K*N) * torch.finfo(dtype).bits // 8 # + M*N

    problem_count = 1 + int((3 * cache_size) / total_bytes)
    print("problem_count", problem_count, cache_size, total_bytes)
    #
    inputs = []
    weights = []
    if is_fp8:
        torch.use_deterministic_algorithms(False, warn_only=True)

    for i in range(problem_count):
        inputs.append(rand_tensor((M, K), dtype=dtype))
        weights.append(rand_tensor((N, K), dtype=dtype))

    input_scale = None
    weight_scale = None

    if is_fp8:
        input_scale = rand_tensor(1, dtype=torch.float32).cuda()
        weight_scale = rand_tensor(1, dtype=torch.float32).cuda()
    elif is_s8_dequant:
        input_scale = rand_tensor((M, 1), dtype=torch.float32)
        weight_scale = rand_tensor((1, N), dtype=torch.float32)

    bias = None
    if args.has_bias:
        bias_dtype = output_dtype
        # bias_shape = (1, N) if is_fp8 or is_s8_dequant else (M, N)
        bias_shape = (N)
        bias = rand_tensor(bias_shape, bias_dtype)

    perf_result_xop = perf_xop(
        inputs,
        weights,
        bias,
        input_scale,
        weight_scale,
        args.transpose_weight,
        is_fp8,
        is_s8_dequant,
        args.warmup_iters,
        args.iters,
        problem_count, 
        output_dtype,
    )
    xop_perf.append(perf_result_xop.gemm_time_ms)

    perf_result_torch = perf_torch(
        inputs,
        weights,
        bias,
        input_scale,
        weight_scale,
        is_fp8,
        is_s8_dequant,
        args.warmup_iters,
        args.iters,
        problem_count,
        output_dtype,
    )
    torch_perf.append(perf_result_torch.gemm_time_ms)

    print(perf_result_torch)
    print(perf_result_xop)

    xop_output = perf_result_xop.output
    torch_output = perf_result_torch.output

    # is_bitwise_match = xop.bitwise_check(xop_output, torch_output)
    # print("is bitwise match: ", is_bitwise_match)
    atol = THRESHOLD_MAP[xop_output.dtype]
    rtol = THRESHOLD_MAP[xop_output.dtype]
    xop.torch_allclose(xop_output, torch_output, atol=atol, rtol=rtol)

if __name__ == "__main__":
    init_seed()
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    is_fp8 = is_fp8_dtype(dtype)
    if args.output_dtype == "":
        output_dtype = torch.bfloat16 if is_fp8 or dtype == torch.int8 else dtype
    else:
        output_dtype = DTYPE_MAP[args.output_dtype]
    is_s8_dequant = dtype == torch.int8 and output_dtype == torch.bfloat16

    if is_s8_dequant:
        if args.transpose_weight:
            raise ValueError("s8 gemm with dequant must in RCR layout")

    xop_perf = []
    torch_perf = []
    print(f"M: {1}, N: {args.N}, K: {args.K}")
    run(1, args, xop_perf, torch_perf)
    for m in range(2, args.M, args.step):
        print(f"M: {m}, N: {args.N}, K: {args.K}")
        run(m, args, xop_perf, torch_perf)

    plot_x = [1] + list(range(2, args.M, args.step))
    plt.plot(plot_x, xop_perf, label='xop', marker='o', markersize=3)
    plt.plot(plot_x, torch_perf, label='torch', marker='s', markersize=3)

    plt.title(f'perf-N{args.N}-K{args.K}')
    plt.xlabel('m_size')
    plt.ylabel('ms')

    plt.legend()
    plt.grid(True)

    # plt.xticks(plot_x)
    plt.savefig('perf-N-{0}-K-{1}.png'.format(args.N, args.K))
    plt.show()