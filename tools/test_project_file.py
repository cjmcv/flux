
import torch
import torch.nn.functional as F

import xop
from xop.project.qwen3_4b_h20 import XopGemmSpecify
    
###########################################
from torch.library import Library
from xop.project.qwen3_4b_h20 import direct_register_custom_op

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

    def xop_gemm_normal(
        run_mode: int,
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor) -> None:
        xop_gemm.forward(run_mode=run_mode, input=input, weight=weight, output=output)
        
    def xop_gemm_normal_fake(
        run_mode: int,
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor) -> None:
        pass
    
    direct_register_custom_op(
        op_name="xop_gemm_normal",
        op_func=xop_gemm_normal,
        mutates_args=["output"],
        fake_impl=xop_gemm_normal_fake,
        target_lib=xop_lib,
    )
    
    # def forward_fn():
    #     mode = xop_gemm.get_run_mode(A.shape[0], B.shape[0], A.shape[1])
    #     mode = 0
    #     if (mode != 0):
    #         C = xop_gemm.forward(mode, input=A, weight=B)
    #         print("Xop1 : ", C.shape)
    #     else:
    #         C = F.linear(A, B, None)
    #         print("Xop0 : ", C.shape)
    #     return C

    run_mode = xop_gemm.get_run_mode(A.shape[0], B.shape[0], A.shape[1])
    run_mode = 1
    if (run_mode != 0):
        torch.ops.xop.xop_gemm_normal(run_mode=run_mode,input=A,weight=B,output=C)
        print("Xop1 : ", C.shape)
    else:
        C = F.linear(A, B, None)
        print("Xop0 : ", C.shape)
            
    
    print("Compiling...")
    compiled_func = torch.compile(torch.ops.xop.xop_gemm_normal, backend="inductor")
    compiled_func(run_mode, A,B,C)
    
    print("Capturing...")
    stream = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g):
            compiled_func(run_mode, A,B,C)
            # torch.ops.xop.xop_gemm_normal(run_mode=run_mode,input=A,weight=B,output=C)
            
    print("Cuda graph replay: ", g.pool())
    g.replay()
    print(C)
    # print(g)
    
    
if __name__ == '__main__':
    main()