import numpy as np
from typing import Optional, List, Tuple
import ctypes

import torch
from torch.distributed import ProcessGroup

import xop
from xop.ops.custom_all_reduce import CustomAllreduce

ENABLE_ALLREDUCE = 0
ALLREDUCE_GPUID_OFFSET = 5
IS_SPLIT_M = 0

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
            self.ar = CustomAllreduce(group, device)
            self.rank = rank
        
    def ar(self):
        return self.ar

    def forward_inner(
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
            
            if self.ar.is_capturing():
                if torch.cuda.is_current_stream_capturing():
                    return self.gemm_comm.forward(input, weight, output, bias=bias,
                        input_scale=input_scale, weight_scale=weight_scale, output_scale=output_scale,
                        tuning = tuning, fast_accum=fast_accum,
                        registered=True, fa=fa, reg_buffer=0, reg_buffer_sz_bytes=0)
            else:
                return self.gemm_comm.forward(input, weight, output, bias=bias,
                    input_scale=input_scale, weight_scale=weight_scale, output_scale=output_scale,
                    tuning = tuning, fast_accum=fast_accum,
                    registered=False, fa=fa, reg_buffer=reg_buffer, reg_buffer_sz_bytes=reg_buffer_sz_bytes,
                )
        else:
            fa, reg_buffer, reg_buffer_sz_bytes = 0,0,0
            reg_buffer = torch.zeros((output.shape[0], output.shape[1]), dtype=torch.int32).cuda()
            
            self.gemm_comm.forward(input, weight, output, bias=bias,
                input_scale=input_scale, weight_scale=weight_scale, output_scale=output_scale,
                tuning = tuning, fast_accum=fast_accum,
                registered=False, fa=fa, reg_buffer=reg_buffer.data_ptr(), reg_buffer_sz_bytes=reg_buffer_sz_bytes,
            )
            # print(reg_buffer)
            # print(reg_buffer.flatten()[:32*32])
            
            # for i, v in enumerate(reg_buffer.flatten()[:32*32+1]):
            #     print(f'{i:3d}  {v.item()}  {v.item()%16+1}')
                    
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
        if (input.size(0) > 2048 and IS_SPLIT_M == True):
            # TODO: 普通gemm正常，serial也正常，不使用cuda graph也正常。但是使用cudagraph且非serial则不正常。使用4096测试
            input_chunks = split_rows(input, 1024)
            output_chunks = split_rows(output, 1024)
            for i in range(len(input_chunks)):
                self.forward_inner(input_chunks[i], weight, output_chunks[i], bias=bias,
                                  input_scale=input_scale, weight_scale=weight_scale, output_scale=output_scale,
                                  tuning = tuning, fast_accum=fast_accum)
            return 0
        else:
            return self.forward_inner(input, weight, output, bias=bias,
                                    input_scale=input_scale, weight_scale=weight_scale, output_scale=output_scale,
                                    tuning = tuning, fast_accum=fast_accum)