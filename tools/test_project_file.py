
import torch
import torch.nn.functional as F

import xop
from xop.project.qwen3_4b_l40 import XopGemmSpecify
    
# python tools/test_torch_compile.py
def main():
    M = 256
    N = 4096
    K = 4096
    A = torch.randn((M, K), device='cuda', dtype=torch.bfloat16, requires_grad=False)
    B = torch.randn((N, K), device='cuda', dtype=torch.bfloat16, requires_grad=False)
    C = torch.randn((M, N), device='cuda', dtype=torch.bfloat16, requires_grad=False)
    
    D = F.linear(A, B, None)
    print("torch: ", D, D.shape)
    
    xop_gemm_normal = XopGemmSpecify(input_dtype=torch.bfloat16, output_dtype=torch.bfloat16, fast_accum=False)
    mode = xop_gemm_normal.get_run_mode(A.shape[0], B.shape[0], A.shape[1])
    if (mode != 0):
        C = xop_gemm_normal.forward(mode, input=A, weight=B)
        print("Xop1 : ", C, C.shape)
    else:
        C = F.linear(A, B, None)
        print("Xop0 : ", C, C.shape)
    
if __name__ == '__main__':
    main()