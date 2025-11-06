
from typing import Optional

import torch
import torch.nn as nn

import xop

class GemmQuant:
    def __init__(
        self,
        input_dtype: torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
        quant_bits: int = 8,
        num_groups: int = -1,
    ): 
        self.num_groups = num_groups
        if (num_groups == -1):
            self.num_groups = 128 # default
            
        self.quant_bits = quant_bits
        if (self.quant_bits == 8):
            self.gemm_normal = xop.GemmNormal(
                input_dtype=torch.float8_e4m3fn,
                output_dtype=output_dtype,
                transpose_weight=False
            )
        elif (self.quant_bits == 44):
            self.gemm_normal = xop.GemmNormal(
                input_dtype=input_dtype,
                output_dtype=output_dtype,
                transpose_weight=False
            )

    def weight_preprocess(self, weight: torch.Tensor, fast_accum: bool = False):
        if (self.quant_bits == 8):
            y_fp8, y_scale = xop.triton_per_block_cast_to_fp8(weight, fast_accum)
            return y_fp8, y_scale.t().contiguous()
        elif (self.quant_bits == 4):
            n = weight.shape[0]
            self.workspace = torch.zeros(n // 128 * 16, device=weight.device)
            w_fp, q_int4, s_int4 = xop.marlin_quant_int4(weight, -1) # self.num_groups: Can not support 128 on h20? 
            return q_int4, s_int4
        else: # 44
            q_int4, s_int4 = xop.symmetric_group_w4a16_pack_bf16_reorder(weight, self.num_groups)
            print(weight.shape, q_int4.shape, q_int4.dtype, s_int4.shape, s_int4.dtype)
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
            chunk_size = 16384
            if (tuning is not None and tuning[0] == 2):
                chunk_size = tuning[1]

            if (weight_scale is None):
                y_fp8, y_scale = xop.triton_per_block_cast_to_fp8(weight, fast_accum)
            else:
                y_fp8, y_scale = weight, weight_scale
            x_fp8, x_scale = xop.triton_per_token_cast_to_fp8(input, fast_accum)
            x_scale = xop.gemm_blockscale_fp8_scale_a_preprocess(x_scale, chunk_size)

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
        elif (self.quant_bits == 4):
            thread_k, thread_n = -1, -1 # 64, 256
            xop.marlin_fp16xint4_matmul(input, weight, output, weight_scale, self.workspace, thread_k, thread_n, -1, 16)
            return 0
        else: # 44
            # out = torch.matmul(input, weight.t())
            # print("out", out)            
            # return 0
            return self.gemm_normal.forward(
                input,
                weight,
                output=output,
                bias=bias,
                input_scale=None,
                weight_scale=weight_scale,
                output_scale=None,
                tuning = tuning,
                fast_accum=fast_accum,
            )
