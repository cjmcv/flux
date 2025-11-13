
import torch
import torch.nn.functional as F

import xop
from xop.project.qwen3_4b_h20 import XopGemmSpecify
    
###########################################
from torch.library import Library
from xop.project.qwen3_4b_h20 import register_xop_gemm

xop_lib = Library("xop", "FRAGMENT")


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
    
    xop_gemm = XopGemmSpecify(B, input_dtype=torch.bfloat16, output_dtype=torch.bfloat16, fast_accum=False)

    register_xop_gemm(xop_gemm, op_name="xop_gemm_normal", target_lib=xop_lib)

    run_mode = xop_gemm.get_run_mode(A.shape[0], B.shape[0], A.shape[1])
    run_mode = 4
    # if (run_mode != 0):
    #     C = torch.ops.xop.xop_gemm_normal(run_mode=run_mode,input=A,weight=B)
    #     print("Xop1 : ", C.shape)
    # else:
    #     C = F.linear(A, B, None)
    #     print("Xop0 : ", C.shape)
            
    
    print("Compiling...")
    compiled_func = torch.compile(torch.ops.xop.xop_gemm_normal, backend="inductor")
    G = compiled_func(run_mode, A, B)
    print("G", G)
    
    # print("Warm-up compile...")
    stream = torch.cuda.Stream()
    # with torch.cuda.stream(stream):
    #     compiled_func(run_mode, A, B)
    # torch.cuda.synchronize() 
    
    print("Capturing...")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g):
            E = compiled_func(run_mode, A, B)
            # E = torch.ops.xop.xop_gemm_normal(run_mode=run_mode,input=A,weight=B)
            
    print("Cuda graph replay: ", g.pool())
    
    C.zero_()
    g.replay()
    print(E)
    # print(g)
    
    
if __name__ == '__main__':
    main()