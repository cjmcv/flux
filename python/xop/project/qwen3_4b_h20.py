
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F

import numpy as np
import xop
from xop.common import Meta

from torch.library import Library
from .common import direct_register_custom_op

ENABLE_QUANT_8 = 1
ENABLE_QUANT_4 = 1

# tuning

class XopGemmSpecify:
    def __init__(
        self,
        weight,
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
            self.q8_y, self.q8_y_scale = self.gemm_quant8.weight_preprocess(weight, self.fast_accum)
        
        if (ENABLE_QUANT_4):
            self.gemm_quant4 = xop.GemmQuant(
                input_dtype=input_dtype,
                output_dtype=output_dtype,
                quant_bits=4
            )
            self.q4_y, self.q4_y_scale = self.gemm_quant4.weight_preprocess(weight, self.fast_accum)

    def get_run_mode(self, M, N, K) -> int:
        
        self.hparam[1] = 8192
        self.hparam[2] = -1
        mode = 0
        #######################################
        
        # if ((M <= 64) or 
        #     (M <= 256 and N == 2560 and K == 4096) or 
        #     (M <= 128 and N == 2560 and K == 9728) or
        #     (M <= 256 and N == 6144 and K == 2560)):

        if (M <= 32):
            mode = 4
            # self.hparam[2] = Meta.Sm80

        # if ((M >= 512 and N == 19456 and K == 2560) or 
        #     (M >= 2048 and N == 6144 and K == 2560) or 
        #     (M >= 2048 and N == 2560 and K == 9728) or 
        #     (M >= 4096 and N == 2560 and K == 4096)):
        #     mode = 8
            
        return mode
    
    def forward(
        self,
        run_mode,
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> int: 
        if (run_mode == 0):
            assert(0)
        elif (run_mode == 1):
            print("xop_noquant")
            self.gemm_normal.forward(input, weight, output, bias, 
                                    None, None, None, 
                                    self.hparam, self.fast_accum)
        elif (run_mode == 8):
            print("xop_quant8")
            self.gemm_quant8.forward(input, self.q8_y, output, bias, 
                                    None, self.q8_y_scale, None, 
                                    self.hparam, self.fast_accum)
        elif (run_mode == 4):
            print("xop_quant4")
            self.gemm_quant4.forward(input, self.q4_y, output, bias, 
                                    None, self.q4_y_scale, None, 
                                    None, self.fast_accum)
