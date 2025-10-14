
import sys
import torch
import time
import math
from typing import Tuple

def get_arch():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    major = properties.major
    minor = properties.minor
    return major * 10 + minor

def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype.itemsize == 1 and dtype.is_floating_point

def rand_tensor(shape: list[int], dtype: torch.dtype):
    if dtype in [torch.int32, torch.int8]:
        return torch.randint(-127, 128, shape, dtype=dtype).cuda()
    elif is_fp8_dtype(dtype):
        data = torch.rand(shape, dtype=torch.bfloat16).cuda() / 10
        return data.to(dtype)
    else:
        return torch.rand(shape, dtype=dtype).cuda() * 2 - 1
    
def calculate_tflops(m, k, n, time_ms):
    total_flops = m * n * k * 2
    time_s = time_ms / 1000.0
    tflops = total_flops / (10**12 * time_s)
    return tflops

class PerfResult:
    def __init__(self, name: str, output: torch.Tensor, gemm_time_ms: float) -> None:
        self.name = name
        self.output = output
        self.gemm_time_ms = gemm_time_ms

    def __repr__(self) -> str:
        return f"{self.name}: gemm {self.gemm_time_ms:.3f} ms"

def perf_gemm(warmup_iters: int, iters: int, name: str, fn: callable):
    if (warmup_iters + iters == 0):
        output = fn(0)
        torch.cuda.synchronize()
        return PerfResult(name=name, output=output, gemm_time_ms=1 / 1 * 1000)
    
    total_time = 0
    for i in range(warmup_iters + iters):
        if (i == warmup_iters):
            torch.cuda.synchronize()
            start = time.time()
        output = fn(i)

    torch.cuda.synchronize()
    end = time.time()
    total_time = end - start

    output.zero_()
    output = fn(0)
    # print(output)
    torch.cuda.synchronize()
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

# scale = max(abs(BF16)) / 448
# FP8 = clamp(round(BF16 / scale), -448, 448)
# If using per-token quantization, the max is taken over the current token; 
# if using per-block quantization, the max is taken over the entire block.
def per_token_cast_to_fp8(x: torch.Tensor, fast_accum: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    max_value = 448.0
    if (fast_accum):
        max_value = 22.0 # ? Experimental data
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (max_value / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n
    ), (x_amax / max_value).view(m, -1)

def per_block_cast_to_fp8(x: torch.Tensor, fast_accum: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    max_value = 448.0
    if (fast_accum):
        max_value = 22.0 # ? Experimental data
    m, n = x.shape
    x_padded = torch.zeros(
        (math.ceil(m/128) * 128, math.ceil(n/128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (max_value / x_amax)).to(torch.float8_e4m3fn)
    # x_scaled = (x_view).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / max_value).view(
        x_view.size(0), x_view.size(2)
    )

def torch_allclose(x, y, rtol, atol, print_prefix="", verbose=True):
    if not torch.allclose(x, y, rtol=rtol, atol=atol):
        print(f"shape of x: {x.shape}")
        print(f"shape of y: {y.shape}")

        print("x:", file=sys.stderr)
        print(x, file=sys.stderr)
        print("y:", file=sys.stderr)
        print(y, file=sys.stderr)
        print("x-y", x - y, file=sys.stderr)
        diff_loc = torch.isclose(x, y, rtol=rtol, atol=atol) == False
        # print("x diff:", file=sys.stderr)
        print(x[diff_loc], file=sys.stderr)
        # print("y diff:", file=sys.stderr)
        print(y[diff_loc], file=sys.stderr)
        num_diff = torch.sum(diff_loc)

        if len(y.shape) == 1:
            diff_rate = num_diff / y.shape[0]
        else:
            diff_rate = num_diff / (y.shape[0] * y.shape[1])
        # print(f"diff count: {num_diff} ({diff_rate*100:.3f}%), {list(y.shape)}", file=sys.stderr)
        max_diff = torch.max(torch.abs(x - y))
        rtol_abs = rtol * torch.min(torch.abs(y))
        # print(f"diff max: {max_diff}, atol: {atol}, rtol_abs: {rtol_abs}", file=sys.stderr)
        diff_indices = (diff_loc == True).nonzero(as_tuple=False)
        # print(f"diff locations:\n{diff_indices}", file=sys.stderr)
        print("--------------------------------------------------------------\n", file=sys.stderr)
        raise RuntimeError

    if verbose:
        print(print_prefix + " all close!")

# __all__ = [
#     "is_fp8_dtype",
#     "get_arch",
#     "torch_allclose"
# ]
