
import torch
import torch.nn.functional as F

import xop
from xop.project.qwen3_4b_l40 import XopGemmSpecify
    
# python tools/test_torch_compile.py
def main():
    M = 1
    N = 4096
    K = 4096
    A = torch.randn((M, K), device='cuda', dtype=torch.bfloat16, requires_grad=False)
    B = torch.randn((N, K), device='cuda', dtype=torch.bfloat16, requires_grad=False)
    C = torch.randn((M, N), device='cuda', dtype=torch.bfloat16, requires_grad=False)
    
    D = F.linear(A, B, None)
    print("torch: ", D, D.shape)
    
    xop_gemm_normal = XopGemmSpecify(input_dtype=torch.bfloat16, output_dtype=torch.bfloat16, 
                                     fast_accum=False, weight=B)
    mode = xop_gemm_normal.get_run_mode(input=A, output=C)
    if (mode != 0):
        xop_gemm_normal.forward(mode, input=A, output=C)
    else:
        C = F.linear(A, B, None)
    print("Xop : ", C, C.shape)
    
if __name__ == '__main__':
    main()