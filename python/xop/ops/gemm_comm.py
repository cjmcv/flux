
from typing import Optional, List, Tuple

import torch
from torch.distributed import ProcessGroup

import numpy as np
import xop
from xop.ops.custom_all_reduce import CustomAllreduce

class GemmCommRs:
    def __init__(
        self,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        transpose_weight: bool,
        group: ProcessGroup,
        rank: int,
    ): 
        self.gemm_comm = xop.GemmComm(
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            transpose_weight=transpose_weight
        )

        device = torch.device(f"cuda:{rank}")
        self.ar = CustomAllreduce(group, device)
        self.rank = rank
        
    def capture(self):
        self.ar.capture()
        
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
        gemm_out = torch.empty_like(output)
        self.gemm_comm.forward(
            input,
            weight,
            output=gemm_out,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=output_scale,
            tuning = tuning,
            fast_accum=fast_accum,
        )
        self.ar.custom_all_reduce_t(gemm_out, output)
