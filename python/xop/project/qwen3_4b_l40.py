
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F

import numpy as np
import xop

ENABLE_QUANT_8 = 1
ENABLE_QUANT_4 = 1

class XopGemmSpecify:
    def __init__(
        self,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        fast_accum: bool,
        weight: Optional[torch.Tensor] = None,
    ): 
        self.hparam = torch.zeros(2, dtype=torch.int16, device='cpu')
        self.hparam[0] = 2
        
        self.weight = weight
        self.fast_accum = fast_accum
        
        self.gemm_normal = xop.GemmNormal(
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            transpose_weight=False
        )
            
        if (ENABLE_QUANT_8):
            self.gemm_quant8 = xop.GemmQuant(
                input_dtype=input_dtype,
                output_dtype=output_dtype,
                quant_bits=8
            )
            self.q8_y, self.q8_y_scale = self.gemm_quant8.weight_preprocess(weight, self.fast_accum)
        
        if (ENABLE_QUANT_4):
            self.gemm_quant4 = xop.GemmQuant(
                input_dtype=input_dtype,
                output_dtype=output_dtype,
                quant_bits=4
            )
            self.q4_y, self.q4_y_scale = self.gemm_quant4.weight_preprocess(weight, self.fast_accum)

    def get_run_mode(
        self,
        input: torch.Tensor,
        output: torch.Tensor
    ) -> int:
        
        M = input.shape[0]
        N = output.shape[1]
        K = input.shape[1]
        
        #######################################
        mode = 0
        # if (M >= 32):
        #     mode = 1
        #     if (M >= 1024 and ENABLE_QUANT_8 == 1):
        #         mode = 8 
        # else:
        #     if (ENABLE_QUANT_4):
        #         mode = 4
        #######################################
        
        # N K => max_m
        if mode == 1:
            self.hparam[1] = 2048
        elif mode == 8:
            self.hparam[1] = 512
            
        return mode
    
    def forward(
        self,
        run_mode,
        input: torch.Tensor,
        output: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> int: 
        if (run_mode == 0):
            assert(0)
        elif (run_mode == 1):
            return self.gemm_normal.forward(input, self.weight, output, bias, 
                                            None, None, None, 
                                            self.hparam, self.fast_accum)
        elif (run_mode == 8):
            return self.gemm_quant8.forward(input, self.q8_y, output, bias, 
                                           None, self.q8_y_scale, None, 
                                           self.hparam, self.fast_accum)
        elif (run_mode == 4):
            return self.gemm_quant4.forward(input, self.q4_y, output, bias, 
                                           None, self.q4_y_scale, None, 
                                           None, self.fast_accum)