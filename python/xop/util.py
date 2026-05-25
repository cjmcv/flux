
import sys
import torch
import time
import math
import numpy as np
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
    
def create_matrix_arange_row(M, N, dtype=torch.bfloat16, device='cuda'):
    row_indices = torch.arange(M, dtype=dtype, device=device)
    matrix = row_indices.unsqueeze(1).expand(M, N).contiguous()  # contiguous is very important!
    return matrix

def create_matrix_arange_col(M, N, dtype=torch.bfloat16, device='cuda'):
    col_indices = torch.arange(N, dtype=dtype, device=device)
    matrix = col_indices.unsqueeze(0).expand(M, N).contiguous()
    return matrix
        
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

def torch_profile(func):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        func(0)
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    
    prof.export_chrome_trace("trace.json") # chrome://tracing/
    
def assert_similar(x, y, eps=1e-2, name="tensor", assert_=False, print_=True):
    def print_red_warning(msg):
        print(f"\033[91m{msg}\033[0m")

    def calc_sim(x, y, name="tensor"):
        x, y = x.data.double(), y.data.double()
        denominator = (x * x + y * y).sum()
        if denominator == 0:
            print_red_warning(f"{name} all zero")
            return 1
        sim = 2 * (x * y).sum() / denominator
        return sim
    
    sim = calc_sim(x, y, name)
    diff = 1.0 - sim
    if not (0 <= diff <= eps):
        print_red_warning(f"{name} Error: {diff}")
        if assert_:
            raise AssertionError(f"{name} Error: {diff}")
        return False
    else:
        if print_:
            print(f"passed: {name} diff={diff}")
        return True
            
def check_allclose_ret(target_run, torch_run, iters, print_mode):
    if (print_mode==2):
        torch.set_printoptions(threshold=float('inf'))
    torch.cuda.synchronize()
    
    # print("inner: ", torch_out, torch_out.data_ptr())
    for i in range(iters):
        target_result = target_run(i)
        torch_result = torch_run(i)
        torch.cuda.synchronize()
        
        total_num = torch_result.numel()
        if (torch.allclose(target_result, torch_result, rtol=1e-2, atol=0)):
            print("allclose: True")
        else:
            if (print_mode >= 1):
                print("target_out:", target_result.shape, "\n", target_result)
                print("torch_out:", torch_result.shape, "\n", torch_result)
                print("diff: ", target_result - torch_result)
            
            radio = abs((target_result - torch_result)/torch_result)
            
            threshold = [0.05, 0.10]
            count0 = (radio > threshold[0]).sum().item()
            count1 = (radio > threshold[1]).sum().item()
            print("radio > ", threshold[0], ": ", count0, "-", count0/total_num, " / ", threshold[1], ": ", count1, "-", count1/total_num)
        assert_similar(target_result, torch_result, name="similar")    

def torch_profile(func):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        func(0)
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    prof.export_chrome_trace("trace.json") # chrome://tracing/     
                    
def perf_gemm(warmup_iters: int, iters: int, name: str, fn: callable, enable_torch_prof: bool = False):
    if (warmup_iters + iters == 0):
        output = fn(0)
        torch.cuda.synchronize()
        return PerfResult(name=name, output=output, gemm_time_ms=1 / 1 * 1000)
    
    my_stream = torch.cuda.Stream()          # 也可传外部流
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    with my_stream:
        for i in range(warmup_iters):
            output = fn(i)
            
    with my_stream:
        start_event.record()
        for i in range(iters):
            fn(i)
        end_event.record()
        my_stream.synchronize()
            
    my_stream.record_event(end_event)
    my_stream.synchronize()
    total_time = start_event.elapsed_time(end_event) * 1e-3

    if enable_torch_prof:
        torch_profile(fn)
        
    # Clean up and run it once to confirm the results are correct.
    if isinstance(output, torch.Tensor):
        output.zero_()
    else:
        for o in output:
            o.zero_()
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

def profile(target_func: callable, torch_ref_func: callable):
    check_allclose_ret(target_func, torch_ref_func, iters=5, print_mode=0)
    
    torch_profile(target_func)
    torch_profile(torch_ref_func)
    
    perf_result_xop = perf_gemm(warmup_iters=100, iters=500, name="target", fn=target_func)
    perf_result_torch = perf_gemm(warmup_iters=100, iters=500, name="torch", fn=torch_ref_func)
    print(f"Latency: {perf_result_xop.gemm_time_ms:0.5}ms vs {perf_result_torch.gemm_time_ms:0.5}(torch) ms")
    
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

# return atol, rtol
def get_allclose_threshold(k, dtype, quant_bits=0):
    # print("aaa", DTYPE_MAP[args.dtype], args.dtype, torch.float8_e4m3fn)
    if (quant_bits == 8):
        return 2e-1*np.sqrt(k), 2e-2
    if (quant_bits == 4 or quant_bits == 44):
        return 2e-1*np.sqrt(k), 2e-2
    if (dtype == "float8_e4m3fn" or dtype == "float8_e5m2"):
        return 2e-1*np.sqrt(k), 2e-2
    # if (args.output_dtype == "s8" or args.output_dtype == "s32"):
    #     return 0, 0
    return 5e-3*np.sqrt(k), 2e-2

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
