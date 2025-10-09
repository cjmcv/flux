import numpy as np
from typing import Optional, List, Tuple
import ctypes

import torch
from torch.distributed import ProcessGroup

import xop

# from xop.ops.custom_all_reduce import CustomAllreduce
from xop.ops.cuda_ipc_manager import CudaIpcManager

ENABLE_ALLREDUCE = 1
ALLREDUCE_GPUID_OFFSET = 0

def split_rows(x: torch.Tensor, stride: int = 1024):
    M = x.size(0)
    n_full, rem = divmod(M, stride)
    if rem == 0:
        return [x[i*stride : (i+1)*stride] for i in range(n_full)]
    
    # Remainder > 0: the second-to-last chunk takes 1024 rows, and the last chunk takes 1024 + rem rows.
    chunks = [x[i*stride : (i+1)*stride] for i in range(n_full - 1)]
    chunks.append(x[(n_full - 1)*stride : ]) 
    return chunks

def split_rows(x: torch.Tensor, stride=1024):
    n_full, rem = divmod(x.size(0), stride)
    chunks = [x[i*stride : (i+1)*stride] for i in range(n_full)]
    if rem:
        chunks.append(x[n_full*stride : ])
    return chunks

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

        if ENABLE_ALLREDUCE:
            device = torch.device(f"cuda:{rank + ALLREDUCE_GPUID_OFFSET}")
            # self.ar = CustomAllreduce(group, device)
            self.ar = CudaIpcManager(group, device)
            self.rank = rank
  
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
        if ENABLE_ALLREDUCE:
            fa, reg_buffer, reg_buffer_sz_bytes = self.ar.address()
        else:
            fa, reg_buffer_sz_bytes = 0,0
            reg_buffer = torch.zeros_like(output).cuda().data_ptr()
        return self.gemm_comm.forward(input, weight, output, bias=bias,
                                    input_scale=input_scale, weight_scale=weight_scale, output_scale=output_scale,
                                    tuning = tuning, fast_accum=fast_accum,
                                    registered=True, fa=fa, reg_buffer=reg_buffer, reg_buffer_sz_bytes=reg_buffer_sz_bytes)