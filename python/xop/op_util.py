
import sys
import time
import math
from typing import Tuple

import torch
import triton
import triton.language as tl

import xop

def pad_row_to_alignment(x, align):
    row = x.size(0)
    pad_rows = (align - (row % align)) % align
    if pad_rows == 0:
        return x
  
    padding = torch.zeros(pad_rows, x.size(1), dtype=x.dtype, device="cuda")
    return torch.cat([x, padding], dim=0)

# scale_a[m, k//128]
# x and x_scale are internally read in multiples of 4, and use predicate tensors handle boundary cases.
# But x_scale is transposed before input, if the m-dimension is not padded to a multiple of 4, the transposed data will be mismatched.
# org: [[1,1]] =>transpose [[1],[1]], in memory, they are the same, like [1,1]
# org: [[1,1]] =>pad [[1,1], [0,0], [0,0], [0,0]] =>transpose [[1,0,0,0], [1,0,0,0]] => it looks like [1,0,0,0,1,0,0,0]
def gemm_v2_blockscale_fp8_scale_a_preprocess(scale_a):
    torch.set_printoptions(precision=8)
    chunk_size = 16384
    if (scale_a.size(0) <= chunk_size):
        return pad_row_to_alignment(scale_a, 4).t().contiguous()
        
    chunks = torch.split(scale_a, chunk_size, dim=0)
    if len(chunks) >= 2 and chunks[-1].size(0) < chunk_size:
        last_two = torch.cat([chunks[-2], chunks[-1]], dim=0)
        last_two = pad_row_to_alignment(last_two, 4).contiguous()
        chunks = chunks[:-2] + (last_two,)

    # print("origin: ", scale_a.shape)
    # for i, chunk in enumerate(chunks):
    #     print(f"chunk {i+1} shape: {chunk.shape}")
    # chunks_t = [chunk.t() for chunk in chunks]
    # for i, chunk in enumerate(chunks_t):
    #     print(f"chunk_t {i+1} shape: {chunk.shape}")

    chunks_t_flat = [chunk.t().reshape(1, -1) for chunk in chunks]
    # for i, chunk in enumerate(chunks_t_flat):
    #     print(f"chunks_t_flat {i+1} shape: {chunk.shape}")
    
    flattened_chunks = []
    for i, chunk in enumerate(chunks_t_flat):
        flattened_chunks.append(chunk)

    res = torch.cat(flattened_chunks, dim=1).contiguous()
    # print("res: ", res.shape)
    return res


# The triton version of per_token_cast_to_fp8
@triton.jit
def _per_token_fp8_cast_kernel(
    x_ptr,            # [m, n]
    y_ptr,            # [m, n]  -> fp8
    scale_ptr,        # [m, n//128] -> per-128 scale
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_scale_m,
    M: tl.constexpr,
    N: tl.constexpr,
    MAX_VAL: tl.constexpr,
    BLOCK_M: tl.constexpr = 1,
    BLOCK_N: tl.constexpr = 128,
):
    pid_m = tl.program_id(0)   # row-id
    pid_n = tl.program_id(1)   # 128-element tile-id in this row

    # offsets for this tile
    offs_m = pid_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # load 128 contiguous elements
    mask = (offs_m < M) & (offs_n < N)
    x = tl.load(x_ptr + offs_m * stride_xm + offs_n * stride_xn, mask=mask, other=0.0)

    # compute amax
    amax = tl.max(tl.abs(x))
    # ensure at least 1e-4
    amax = tl.maximum(amax, 1e-4)
    # scale factor
    scale = MAX_VAL / amax

    # cast to fp8 (e4m3fn) via inline ptx
    fp8 = scale * x
    # clamp to fp8 range, then reinterpret bits (Triton 2.2+ has to_fp8 helper)
    fp8 = tl.clamp(fp8, -MAX_VAL, MAX_VAL)
    fp8 = fp8.to(tl.float8e4nv)   # float8e4nv = e4m3fn

    # store
    tl.store(y_ptr + offs_m * stride_ym + offs_n * stride_yn, fp8, mask=mask)
    # only store one scale per 128-element tile
    if pid_n * BLOCK_N < N:
        tl.store(scale_ptr + offs_m * stride_scale_m + pid_n, amax / MAX_VAL)


def triton_per_token_cast_to_fp8(x: torch.Tensor, fast_accum: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    max_value = 22.0 if fast_accum else 448.0
    m, n = x.shape
    x = x.contiguous()

    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((m, n // 128), dtype=torch.float32, device=x.device)

    grid = lambda META: (m, triton.cdiv(n, META['BLOCK_N']))

    _per_token_fp8_cast_kernel[grid](
        x, y, scale,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        scale.stride(0),
        M=m, N=n, MAX_VAL=max_value,
        BLOCK_N=128,
    )
    return y, scale

# The triton version of per_block_cast_to_fp8
@triton.jit
def _per_block_fp8_cast_kernel(
    x_ptr,            # [M, N]
    y_ptr,            # [M, N] -> fp8
    scale_ptr,        # [M//128, N//128] -> per-tile scale
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_scale_m,
    M, N,
    MAX_VAL: tl.constexpr,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # load 128 contiguous elements
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, mask=mask, other=0.0)

    amax = tl.max(tl.abs(x))
    amax = tl.maximum(amax, 1e-4)
    scale = MAX_VAL / amax

    # scale + cast to fp8
    x_fp8 = scale * x
    x_fp8 = tl.clamp(x_fp8, -MAX_VAL, MAX_VAL)
    x_fp8 = x_fp8.to(tl.float8e4nv)

    # store fp8
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, x_fp8, mask=mask)
    tl.store(scale_ptr + pid_m * stride_scale_m + pid_n, amax / MAX_VAL)

def triton_per_block_cast_to_fp8(x: torch.Tensor, fast_accum: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    max_value = 22.0 if fast_accum else 448.0

    M, N = x.shape
    pad_M = int(math.ceil(M / 128) * 128)
    pad_N = int(math.ceil(N / 128) * 128)

    x_padded = torch.zeros((pad_M, pad_N), dtype=x.dtype, device=x.device)
    x_padded[:M, :N] = x

    y_padded = torch.empty_like(x_padded, dtype=torch.float8_e4m3fn)
    scales = torch.empty((pad_M // 128, pad_N // 128), dtype=torch.float32, device=x.device)

    grid = lambda META: (triton.cdiv(pad_M, META['BLOCK_M']),
                         triton.cdiv(pad_N, META['BLOCK_N']))

    _per_block_fp8_cast_kernel[grid](
        x_padded, y_padded, scales,
        x_padded.stride(0), x_padded.stride(1),
        y_padded.stride(0), y_padded.stride(1),
        scales.stride(0),
        M=pad_M, N=pad_N, MAX_VAL=max_value,
        BLOCK_M=128, BLOCK_N=128,
    )

    # crop back to original shape
    return y_padded[:M, :N].contiguous(), scales[: (M + 127) // 128, : (N + 127) // 128]


def symmetric_group_w4a16_pack_bf16(w_bf16: torch.Tensor,
                                    group_size: int = 128):
    # Perform group-wise symmetric quantization on the bfloat16 weights, 
    # and pack the int4 weights and bfloat16 scales in the format required by CUTLASS W4A16.
    # <input>
    # w_bf16 : torch.Tensor [N, K], dtype=torch.bfloat16
    # group_size : int, The number of channels per group must be divisible by K.
    # <return>
    # packed_w : torch.Tensor [N, K//2], dtype=torch.int8
    # scale : torch.Tensor [N, K//group_size], dtype=torch.bfloat16
    
    assert w_bf16.dim() == 2, "only support 2-D weight"
    N, K = w_bf16.shape
    assert K % group_size == 0, "K must be divisible by group_size"
    num_groups = K // group_size
    
    w = w_bf16.contiguous()   # [N, K]
    # torch.set_printoptions(threshold=float('inf'), linewidth=200, precision=4)

    # Calculate the scale by group (symmetric quantization, zero = 0)
    w_groups = w.reshape(N, num_groups, group_size)        # [N, G, GS]
    w_max = w_groups.abs().amax(dim=-1, keepdim=True)      # [N, G, 1]
    # print("w_max", w_max)
    scale_ = w_max / 8.0                                   # int4 [-7,7]
    scale_ = scale_.clamp(min=1e-12)
    # print("scale_", scale_)
    # Quantize & Round to integers
    w_int4 = torch.round(w_groups / scale_).clamp(-8, 7).to(torch.int8)  # [N, G, GS]
    # Keep the scale as bfloat16 with a dimension of [N, G].
    scale_bf16 = scale_.squeeze(-1).contiguous().to(torch.bfloat16)   # [N, G]
    
    # temp = w_int4 * scale_
    # w_bf16 = temp.reshape(N, K)
    # print("w_bf16", w_bf16, w_bf16.shape, temp.shape, scale_.shape)
    
    # Merge two int4s into one int8.
    # reshape it into the dimension [N, K], then pairwise pack by columns.
    w_int4 = w_int4.reshape(N, K)
    assert K % 2 == 0, "K must be even for packing"
    w_even = w_int4[:, 1::2]
    w_odd  = w_int4[:, 0::2]
    
    # pack
    w_even_uint8 = (w_even & 0x0F).to(torch.uint8)
    w_odd_uint8 = (w_odd & 0x0F).to(torch.uint8)
    packed_w = (w_even_uint8 << 4) | w_odd_uint8
    packed_w = packed_w.to(torch.int8).contiguous()   # [N, K//2]
    
    return packed_w, scale_bf16.t().contiguous()

def symmetric_group_w4a16_pack_bf16_reorder(w_bf16: torch.Tensor, group_size: int):
    q_int4, s_int4 = symmetric_group_w4a16_pack_bf16(w_bf16, group_size)
    xop.gemm_w4a16_sm90_reorder_weight(q_int4)
    return q_int4, s_int4
            