
from typing import Optional, List, Tuple

import torch
import torch.nn as nn

import xop

class GemmQuant:
    def __init__(
        self,
        input_dtype: torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
        transpose_weight: bool = False,
    ): 
        self.gemm_normal = xop.GemmNormal(
            input_dtype=torch.float8_e4m3fn,
            output_dtype=output_dtype,
            transpose_weight=transpose_weight
        )

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

        x_fp8, x_scale = xop.triton_per_token_cast_to_fp8(input, fast_accum)
        y_fp8, y_scale = xop.triton_per_block_cast_to_fp8(weight, fast_accum)

        x_scale = xop.gemm_v2_blockscale_fp8_scale_preprocess(x_scale)

        ret = self.gemm_normal.forward(
            x_fp8,
            y_fp8,
            output=output,
            bias=bias,
            input_scale=x_scale,
            weight_scale=y_scale,
            output_scale=None,
            tuning = None,
            fast_accum=fast_accum,
        )
        return ret
