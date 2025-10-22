
from typing import Optional, List, Tuple

import torch
import torch.nn as nn

import numpy as np
import xop


# Precompute permutations for Marlin weight and scale shuffling 
def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single

_perm, _scale_perm, _scale_perm_single = _get_perms()

def pack2int4(k,n,groupsize, fp16_w, scales):
    # if fp16_w.dtype != torch.half:
    #     raise ValueError('Only `torch.half` weights are supported.')
    tile = 16
    maxq = 2 ** 4 - 1
    s = scales
    w = fp16_w
    if groupsize != k:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
        s = s.reshape((1, -1))
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    if groupsize != k:
        w = w.reshape((groupsize, -1, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((k, n)).contiguous()
        s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
    else:
        s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
    s = s.reshape((-1, n)).contiguous()
    w = w.reshape((k // tile, tile, n // tile, tile))
    w = w.permute((0, 2, 1, 3))
    w = w.reshape((k // tile, n * tile))
    res = w
    res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
    q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res = res.cpu().numpy().astype(np.uint32)
    for i in range(8):
        q |= res[:, i::8] << 4 * i
    q = torch.from_numpy(q.astype(np.int32)).to(w.device)
    return q, s

def marlin_quant_int4(w, groupsize=-1):
    w_fp = w
    m = w.shape[0]
    n = w.shape[1]

    maxq = 2 ** 4 - 1
    # such as groupsize=2: w[8,4] => w[4,g=2,4] => w[g=2,4,4] => w[g=2,16]£¬
    # then you can compute the scale row-wise.
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    # s[1, n], the maximum absolute value of each row.
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    # maxq = 15, In symmetric quantization, only the range [-8, 7] is actually used. 
    # The effective "half-span" is 8, so maxq_half = (maxq + 1) // 2 = 8
    # a / 8 = a / ((maxq + 1) / 2) = a x 2 / (maxq + 1), maxq is taken as 15, omitting the "+1"
    # so: a x 2 / maxq
    s *= 2 / maxq
    s = s.reshape((-1, n)).contiguous()

    if groupsize == -1:
        groupsize = m
    qo, so = pack2int4(m,n,groupsize, w_fp, s)
    return w_fp, qo, so

class GemmQuant:
    def __init__(
        self,
        input_dtype: torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
        quant_bits: int = 8,
    ): 
        self.quant_bits = quant_bits
        if (self.quant_bits == 8):
            self.gemm_normal = xop.GemmNormal(
                input_dtype=torch.float8_e4m3fn,
                output_dtype=output_dtype,
                transpose_weight=False
            )

    def weight_preprocess(self, weight: torch.Tensor, fast_accum: bool = False):
        if (self.quant_bits == 8):
            y_fp8, y_scale = xop.triton_per_block_cast_to_fp8(weight, fast_accum)
            return y_fp8, y_scale
        else:
            w_fp, q_int4, s_int4 = marlin_quant_int4(weight.t())
        return q_int4, s_int4
    
    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        tuning: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
    ) -> int: 
        # x_fp8, x_scale = xutil.per_token_cast_to_fp8(input, fast_accum) # x_fp8[m, k], x_scale[m, k//128] => cutlass x_scale[m,k]
        # y_fp8, y_scale = xutil.per_block_cast_to_fp8(weight, fast_accum)

        if (self.quant_bits == 8):
            if (weight_scale is None):
                y_fp8, y_scale = xop.triton_per_block_cast_to_fp8(weight, fast_accum)
            else:
                y_fp8, y_scale = weight, weight_scale
            x_fp8, x_scale = xop.triton_per_token_cast_to_fp8(input, fast_accum)
            x_scale = xop.gemm_v2_blockscale_fp8_scale_a_preprocess(x_scale)

            return self.gemm_normal.forward(
                x_fp8,
                y_fp8,
                output=output,
                bias=bias,
                input_scale=x_scale,
                weight_scale=y_scale,
                output_scale=None,
                tuning = tuning,
                fast_accum=fast_accum,
            )
        else:
            m = output.shape[0]
            n = output.shape[1]
            workspace = torch.zeros(n // 128 * 16, device=input.device)
            thread_k, thread_n = -1, -1 # 64, 256
            xop.marlin_fp16xint4_matmul(input, weight, output, weight_scale, workspace, thread_k, thread_n, -1, 16)
            return 0
