
from typing import Optional, Any, List, Tuple

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



def register_xop_gemm(xop_gemm: Any,
                    is_quant4: bool,
                    op_name: str = "xop_gemm_normal",
                    target_lib: Any = None) -> None:
    
    def xop_gemm_normal(
        input: torch.Tensor,
        weight: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        tuning: Optional[torch.Tensor] = None,
        fast_accum: bool = False) -> torch.Tensor:  
        return xop_gemm.forward(input, weight, None, bias, input_scale, weight_scale, None, tuning, fast_accum)
    
    # Question:
    #  File "/tmp/torchinductor_cjmcv/ji/cjidr76cmg3e7dwi2qf5tvisjb76c4taqaeyknitn7tv745mpfru.py", line 48, in call
    #  assert_size_stride(buf1, (123, 4096), (4096, 1), 'torch.ops.xop.gemm_quant4_forward.default')
    #  AssertionError: expected size 256==123, stride 4096==4096 at dim=0
    # Answer:
    #  You need to delete "torchinductor_cjmcv" and then generate/run it again.
    def xop_gemm_normal_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        tuning: Optional[torch.Tensor] = None,
        fast_accum: bool = False) -> torch.Tensor: 
        if (is_quant4):
            return torch.empty(
                (input.shape[0], weight.shape[1]//2),
                dtype=input.dtype,
                device=input.device
            )
        else:
            return torch.empty(
                (input.shape[0], weight.shape[0]),
                dtype=input.dtype,
                device=input.device
            )
    
    direct_register_custom_op(
        op_name=op_name,
        op_func=xop_gemm_normal,
        mutates_args=[], # "output"
        fake_impl=xop_gemm_normal_fake,
        target_lib=target_lib,
    )

g_gemm_normal = xop.GemmNormal(
    input_dtype=torch.bfloat16,
    output_dtype=torch.bfloat16,
    transpose_weight=False
)

if (ENABLE_QUANT_8):
    g_gemm_quant8 = xop.GemmQuant(
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        quant_bits=8
    )

if (ENABLE_QUANT_4):
    g_gemm_quant4 = xop.GemmQuant(
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        quant_bits=4
    )

xop_lib = Library("xop", "FRAGMENT")
register_xop_gemm(g_gemm_normal, False, op_name="gemm_normal_forward", target_lib=xop_lib)
register_xop_gemm(g_gemm_quant8, False, op_name="gemm_quant8_forward", target_lib=xop_lib)
register_xop_gemm(g_gemm_quant4, True, op_name="gemm_quant4_forward", target_lib=xop_lib)

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
        
        if (ENABLE_QUANT_8):
            self.q8_y, self.q8_y_scale = g_gemm_quant8.weight_preprocess(weight, self.fast_accum)
        
        if (ENABLE_QUANT_4):
            self.q4_y, self.q4_y_scale = g_gemm_quant4.weight_preprocess(weight, self.fast_accum)

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
        bias: Optional[torch.Tensor] = None
    ) -> int: 
        if (run_mode == 0):
            assert(0)
        elif (run_mode == 1):
            # print("xop_noquant")
            return torch.ops.xop.gemm_normal_forward(input, weight, None, bias, 
                                    None, None, None, 
                                    self.hparam, self.fast_accum)
        elif (run_mode == 8):
            # print("xop_quant8")
            return torch.ops.xop.gemm_quant8_forward(input, self.q8_y, None, bias, 
                                    None, self.q8_y_scale, None, 
                                    self.hparam, self.fast_accum)
        elif (run_mode == 4):
            # print("xop_quant4", input.shape[0], self.q4_y.shape[1]//2)
            return torch.ops.xop.gemm_quant4_forward(input, self.q4_y, None, bias, 
                                    None, self.q4_y_scale, None, 
                                    None, self.fast_accum)
            