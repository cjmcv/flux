
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F

import numpy as np
import xop
from xop.common import Meta

ENABLE_QUANT_8 = 1
ENABLE_QUANT_4 = 1

# tuning

class XopGemmSpecify:
    def __init__(
        self,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        fast_accum: bool,
    ): 
        self.hparam = torch.zeros(3, dtype=torch.int16, device='cpu')
        self.hparam[0] = 2
        self.hparam[2] = -1 # Meta.Sm80
        
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
            self.weight_quant8_ready = False
        
        if (ENABLE_QUANT_4):
            self.gemm_quant4 = xop.GemmQuant(
                input_dtype=input_dtype,
                output_dtype=output_dtype,
                quant_bits=4
            )
            self.weight_quant4_ready = False

    def get_run_mode(self, M, N, K) -> int:
        
        #######################################
        mode = 0
        if (M >= 128):
            mode = 1
            if (M >= 2048 and ENABLE_QUANT_8 == 1):
                mode = 8 
        elif (M <= 32):
            if (ENABLE_QUANT_4):
                mode = 4
        #######################################
        
        # N K => max_m
        if mode == 1:
            if (N == 2560 and K == 9728) or (N == 19456 and K == 2560):
                self.hparam[1] = 512
            else:  # if (N == 2560 and K == 4096) or (N == 6144 and K == 2560)
                self.hparam[1] = 1024
        elif mode == 8:
            if (N == 2560 and K == 4096) or (N == 6144 and K == 2560):
                self.hparam[1] = 1024
            elif (N == 2560 and K == 9728):
                self.hparam[1] = 2048
            else:
                self.hparam[1] = 512
            
        return mode
    
    def forward(
        self,
        run_mode,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> int: 
        output = torch.empty(input.shape[0], weight.shape[0], dtype=input.dtype, device="cuda")
        
        if (run_mode == 0):
            assert(0)
        elif (run_mode == 1):
            print("xop_noquant")
            self.gemm_normal.forward(input, weight, output, bias, 
                                    None, None, None, 
                                    self.hparam, self.fast_accum)
        elif (run_mode == 8):
            print("xop_quant8")
            if (self.weight_quant8_ready == False):
                self.q8_y, self.q8_y_scale = self.gemm_quant8.weight_preprocess(weight, self.fast_accum)
                self.weight_quant8_ready = True
                
            self.gemm_quant8.forward(input, self.q8_y, output, bias, 
                                    None, self.q8_y_scale, None, 
                                    self.hparam, self.fast_accum)
        elif (run_mode == 4):
            print("xop_quant4")
            if (self.weight_quant4_ready == False):
                self.q4_y, self.q4_y_scale = self.gemm_quant4.weight_preprocess(weight, self.fast_accum)
                self.weight_quant4_ready = True
                
            self.gemm_quant4.forward(input, self.q4_y, output, bias, 
                                    None, self.q4_y_scale, None, 
                                    None, self.fast_accum)
        
        return output