
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

    return xutil.perf_gemm(warmup_iters, iters, "torch", fn)

def perf_xop(
    inputs: list[torch.Tensor],
    weights: list[torch.Tensor],
    bias: Optional[torch.Tensor],
    inputs_scale: Optional[torch.Tensor],
    weights_scale: Optional[torch.Tensor],
    transpose_weight: bool,
    is_fp8: bool,
    is_s8_dequant: bool,
    warmup_iters: int, 
    iters: int,
    problem_cnt: int,
    output_dtype: torch.dtype,
    fast_accum: bool,
    quant_bits: int,
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
        if not _check_tensor_shape(inputs_scale, (m, 1)):
            raise ValueError("input_scale's shape should be (m, 1) for S8 GEMM")
        if not _check_tensor_shape(weights_scale, (1, n)):
            raise ValueError("weight_scale's shape should be (1, n) for S8 GEMM")

    output = torch.empty([m, n], dtype=output_dtype, device=inputs[0].device, requires_grad=False)
    if (quant_bits != -1):
        op = xop.GemmQuant(
            input_dtype=inputs[0].dtype,
            output_dtype=output_dtype,
            quant_bits=quant_bits
        )
        weights_fp8 = []
        weights_fp8_scale = []
        for i in range(problem_cnt):
            weight_fp8, weight_fp8_scale = op.weight_preprocess(weights[i], fast_accum)
            weights_fp8.append(weight_fp8)
            weights_fp8_scale.append(weight_fp8_scale)

        def fn(iter_id):
            problem_idx = iter_id % problem_cnt
            op.forward(
                inputs[problem_idx],
                weights_fp8[problem_idx],
                output=output,
                bias=bias,
                input_scale=None,
                weight_scale=weights_fp8_scale[problem_idx],
                output_scale=None,
                tuning = None,
                fast_accum=fast_accum,
            )
            return output
    else:
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
    if (args.quant_bits == 8):
        return 2e-1*np.sqrt(k), 2e-2
    if (args.quant_bits == 4):
        return 2e-1*np.sqrt(k), 2e-2
    if (args.output_dtype == torch.int8 or args.output_dtype == torch.int32):
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
    is_fp8 = xutil.is_fp8_dtype(dtype)
    if args.output_dtype == "":
        output_dtype = torch.bfloat16 if is_fp8 or dtype == torch.int8 else dtype
    else:
        output_dtype = DTYPE_MAP[args.output_dtype]
    is_s8_dequant = dtype == torch.int8 and output_dtype == torch.bfloat16

    if is_s8_dequant:
        if args.transpose_weight:
            raise ValueError("s8 gemm with dequant must in RCR layout")
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

    if is_s8_dequant:
        inputs_scale.append(xutil.rand_tensor((M, 1), dtype=torch.float32))
        weights_scale.append(xutil.rand_tensor((1, N), dtype=torch.float32))

    if is_fp8:
        fp8_org_inputs = []
        fp8_org_weights = []
        for i in range(problem_count):
            x = xutil.rand_tensor((M, K), dtype=output_dtype)
            y = xutil.rand_tensor((N, K), dtype=output_dtype)
            # x = torch.ones((M, K), device="cuda", dtype=output_dtype)
            # y = torch.ones((N, K), device="cuda", dtype=output_dtype)
            # x = torch.arange(1, M*K+1, dtype=output_dtype, device="cuda").reshape(M, K)
            # y = torch.arange(1, N*K+1, dtype=output_dtype, device="cuda").reshape(N, K)   
            x_fp8, x_scale = xutil.per_token_cast_to_fp8(x.clone(), args.fast_accum) # x_fp8[m, k], x_scale[m, k//128] => cutlass x_scale[m,k]
            y_fp8, y_scale = xutil.per_block_cast_to_fp8(y.clone(), args.fast_accum)
            # print("data_ptr: ", x_fp8.data_ptr(), y_fp8.data_ptr(), (x_fp8.data_ptr() % 128) == 0, (y_fp8.data_ptr() % 128) == 0)
            x_scale = xop.gemm_v2_blockscale_fp8_scale_a_preprocess(x_scale)

            fp8_org_inputs.append(x)
            fp8_org_weights.append(y)
            inputs.append(x_fp8)
            weights.append(y_fp8)

            inputs_scale.append(x_scale)
            weights_scale.append(y_scale.clone().contiguous())
            
            # for i in range(100):
            #     xop_gemm = xop.GemmNormal(
            #         input_dtype=torch.float8_e4m3fn,
            #         output_dtype=output_dtype,
            #         transpose_weight=False
            #     )
            #     out_xop = torch.empty((M, N), device="cuda", dtype=output_dtype)

            #     xop_gemm.forward(
            #         x_fp8.clone(),
            #         y_fp8.clone(),
            #         output=out_xop,
            #         bias=None,
            #         input_scale=xt_scale.clone(),
            #         weight_scale=yt_scale.clone(),
            #         output_scale=None,
            #         tuning = None,
            #         fast_accum=False,
            #     )
            #     print(out_xop)
    else:
        for i in range(problem_count):
            inputs.append(xutil.rand_tensor((M, K), dtype=dtype))
            weights.append(xutil.rand_tensor((N, K), dtype=dtype))
            inputs_scale.append(None)
            weights_scale.append(None)

    bias = None
    if args.has_bias:
        bias_dtype = output_dtype
        bias_shape = (N) # (M, N)
        bias = xutil.rand_tensor(bias_shape, bias_dtype)

    perf_result_xop = perf_xop(
        inputs,
        weights,
        bias,
        inputs_scale,
        weights_scale,
        args.transpose_weight,
        is_fp8,
        is_s8_dequant,
        args.warmup_iters,
        args.iters,
        problem_count, 
        output_dtype,
        args.fast_accum,
        args.quant_bits,
    )
    
    if not is_fp8:
        perf_result_torch = perf_torch(
            inputs,
            weights,
            bias,
            inputs_scale,
            weights_scale,
            is_fp8,
            is_s8_dequant,
            args.warmup_iters,
            args.iters,
            problem_count,
            output_dtype,
        )
    else:
        fp8_org_inputs[0] 
        output = torch.nn.functional.linear(fp8_org_inputs[0] , fp8_org_weights[0])
        perf_result_torch = xutil.PerfResult(name="torch.sim", output=output, gemm_time_ms=10000)

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

    # is_bitwise_match = xop.bitwise_check(xop_output, torch_output)
    # print("is bitwise match: ", is_bitwise_match)
    atol, rtol = get_allclose_threshold(args, K)
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
    parser.add_argument("--warmup_iters", default=10, type=int, help="perf warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
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
    parser.add_argument(
        "--transpose_weight", default=False, action="store_true", help="whether to transpose weight"
    )

    return parser.parse_args()

# python3 tools/gemm/test_gemm_normal.py 14 4096 4096 --quant_bits=8 --dtype=float16 --output_dtype=float16
# python3 tools/gemm/test_gemm_normal.py 14 4096 4096 --quant_bits=8
# python3 tools/gemm/test_gemm_normal.py 14 4096 4096 --show_ms
# python3 tools/gemm/test_gemm_normal.py 14 4096 4096 --dtype=float16
# python3 tools/gemm/test_gemm_normal.py 14 4096 4096 --dtype=float16 --has_bias 
# python3 tools/gemm/test_gemm_normal.py 14 4096 4096 --dtype=float8_e4m3fn
# python3 tools/gemm/test_gemm_normal.py 14 4096 4096 --dtype=float8_e4m3fn --fast_accum
# python3 tools/gemm/test_gemm_normal.py 14 4096 4096 --dtype=float8_e4m3fn --output_dtype=float16 --fast_accum
if __name__ == "__main__":
    init_seed()
    args = parse_args()

    xop_perf = []
    torch_perf = []
    print(f"M: {1}, N: {args.N}, K: {args.K}")
    run(1, args, xop_perf, torch_perf)

    if 0:
        for m in range(2, args.M, args.step):
            print(f"M: {m}, N: {args.N}, K: {args.K}")
            run(m, args, xop_perf, torch_perf)
        plot_x = [1] + list(range(2, args.M, args.step))
    else:
        exponent = args.M # 65536: 17
        for m in range(1, exponent):
            m = 2**m
            print(f"M: {m}, N: {args.N}, K: {args.K}")
            run(m, args, xop_perf, torch_perf)
        
        plot_x_value = [1] + list(2**x for x in list(range(1, exponent)))
        plot_x = range(len(plot_x_value))
        plt.xticks(plot_x, plot_x_value, rotation=45)

    print("xop_perf:", xop_perf)
    plt.plot(plot_x, xop_perf, label='xop', marker='o', markersize=3)
    plt.plot(plot_x, torch_perf, label='torch', marker='s', markersize=3)
    
    # plt.ylim(bottom=0)  # 

    plt.title(f'perf-N{args.N}-K{args.K}')
    plt.xlabel('m_size')
    if args.show_ms:
        plt.ylabel('ms')
    else:
        plt.ylabel('tflops')

    plt.legend()
    plt.grid(True)

    # plt.xticks(plot_x)
    plt.savefig('perf-N-{0}-K-{1}.png'.format(args.N, args.K))
    plt.show()


# # The usage within torch.compile of vllm.
# # vllm/model_executor/layers/linear.py
# # vllm/v1/attention/backends/flash_attn.py

# from vllm.utils import direct_register_custom_op
# from torch.library import Library
# xop_lib = Library("xop", "FRAGMENT")
# import xop

# xop_gemm = xop.GemmNormal(
#     input_dtype=torch.float16,
#     output_dtype=torch.float16,
#     transpose_weight=False
# )
# def xop_gemm_normal(
#     input_tensor: torch.Tensor,
#     weight_tensor: torch.Tensor,
#     output_tensor: torch.Tensor,
#     bias_tensor: Optional[torch.Tensor] = None,
#     input_scale: Optional[torch.Tensor] = None,
#     weight_scale: Optional[torch.Tensor] = None,
#     output_scale: Optional[torch.Tensor] = None,
#     tuning: Optional[torch.Tensor] = None,
#     fast_accum: bool = False) -> None:
#     xop_gemm.forward(
#             input_tensor,
#             weight_tensor,
#             output_tensor,
#             #None, 
#             bias_tensor,
#             input_scale,
#             weight_scale,
#             output_scale,
#             tuning,
#             fast_accum,
#         )
#     # if bias_tensor is not None:
#     #     output_tensor += bias_tensor # output_tensor = output_tensor + bias_tensor is not allow in vllm torch compile
# def xop_gemm_normal_fake(
#     input_tensor: torch.Tensor,
#     weight_tensor: torch.Tensor,
#     output_tensor: torch.Tensor,
#     bias_tensor: Optional[torch.Tensor] = None,
#     input_scale: Optional[torch.Tensor] = None,
#     weight_scale: Optional[torch.Tensor] = None,
#     output_scale: Optional[torch.Tensor] = None,
#     tuning: Optional[torch.Tensor] = None,
#     fast_accum: bool = False) -> None:
#     pass
# direct_register_custom_op(
#     op_name="xop_gemm_normal",
#     op_func=xop_gemm_normal,
#     mutates_args=["output_tensor"],
#     fake_impl=xop_gemm_normal_fake,
#     target_lib=xop_lib,
# )

# # Call
# torch.ops.xop.xop_gemm_normal(
#                 input_tensor = x,
#                 weight_tensor = layer.weight,
#                 output_tensor = output,
#                 bias_tensor = bias,
#                 input_scale = None,
#                 weight_scale = None,
#                 output_scale = None,
#                 tuning = None,
#                 fast_accum=False,
#             )
