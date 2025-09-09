
import argparse
import time

import torch
import xop

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

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

# python3 tools/gemm/test_gemm_mini.py --has_bias 12 4096 4096
if __name__ == "__main__":
    args = parse_args()

    xop_perf = []
    torch_perf = []
    
    rank = 0
    print(f"M: {args.M}, N: {args.N}, K: {args.K}")

    M = args.M
    N = args.N
    K = args.K
    dtype = DTYPE_MAP[args.dtype]
    output_dtype = DTYPE_MAP[args.output_dtype]
    input = torch.ones((M, K), dtype=dtype).cuda()
    weight = torch.ones((N, K), dtype=dtype).cuda()

    bias = None
    if args.has_bias:
        # bias = xutil.rand_tensor((N), output_dtype)
        bias = torch.ones((N), dtype=output_dtype).cuda() * 12
        

    output = torch.empty([M, N], dtype=output_dtype, device=input.device, requires_grad=False)
    op = xop.GemmCommRs(
        input_dtype=input.dtype,
        output_dtype=output_dtype,
        transpose_weight=False,
        group=None,
        rank=0,
    )
    
    # 
    op.forward(
        input,
        weight,
        output=output,
        bias=bias,
        input_scale=None,
        weight_scale=None,
        output_scale=None,
        tuning = None,
        fast_accum=False,
    )
    torch.set_printoptions(threshold=float('inf'))
    print(output)
    all_ones = output.sub(1).abs().max() < 1e-6
    print("all_ones", all_ones)
    torch.set_printoptions(threshold=1000)
    
    torch.cuda.synchronize()
    start = time.time()
    for i in range(5):
        op.forward(
            input,
            weight,
            output=output,
            bias=bias,
            input_scale=None,
            weight_scale=None,
            output_scale=None,
            tuning = None,
            fast_accum=False,
        )
        # print(output)
        print("all_ones for", output.sub(1).abs().max() < 1e-6)
    torch.cuda.synchronize()
    end = time.time()
    total_time = end - start

    print("total time:", total_time * 1000)