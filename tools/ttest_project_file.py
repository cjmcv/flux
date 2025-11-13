
import torch
import torch.nn.functional as F

import xop
from xop.project.qwen3_4b_h20 import XopGemmSpecify
    
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
    
    xop_gemm_normal = XopGemmSpecify(B, input_dtype=torch.bfloat16, output_dtype=torch.bfloat16, fast_accum=False)

    def forward_fn():
        mode = xop_gemm_normal.get_run_mode(A.shape[0], B.shape[0], A.shape[1])
        mode = 4
        if (mode != 0):
            C = xop_gemm_normal.forward(mode, input=A, weight=B)
            print("Xop1 : ", C.shape)
        else:
            C = F.linear(A, B, None)
            print("Xop0 : ", C.shape)
        return C
    
    print("Capturing...")
    stream = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g):
            C = forward_fn()
            
    print("Cuda graph replay: ", g.pool())
    g.replay()
    print(C)
    print(g)
    
if __name__ == '__main__':
    main()