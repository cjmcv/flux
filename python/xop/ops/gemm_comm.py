import numpy as np
from typing import Optional, List, Tuple

import torch
from torch.distributed import ProcessGroup
from contextlib import contextmanager

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
        
    def ar(self):
        return self.ar
        
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
        fa, reg_buffer, reg_buffer_sz_bytes = self.ar.address()
        if self.ar.is_capturing():
            if torch.cuda.is_current_stream_capturing():
                self.gemm_comm.forward(
                    input,
                    weight,
                    output=output,
                    bias=bias,
                    input_scale=input_scale,
                    weight_scale=weight_scale,
                    output_scale=output_scale,
                    tuning = tuning,
                    fast_accum=fast_accum,
                    registered=True,
                    fa=fa, reg_buffer=0, reg_buffer_sz_bytes=0,
                )
        else:
            self.gemm_comm.forward(
                input,
                weight,
                output=output,
                bias=bias,
                input_scale=input_scale,
                weight_scale=weight_scale,
                output_scale=output_scale,
                tuning = tuning,
                fast_accum=fast_accum,
                registered=False,
                fa=fa, reg_buffer=reg_buffer, reg_buffer_sz_bytes=reg_buffer_sz_bytes,
            )