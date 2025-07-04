
import torch

import xop

op = xop.GemmNormal(
    input_dtype=torch.bfloat16,
    output_dtype=torch.bfloat16,
    transpose_weight=False
)

def my_func(A, B, C):
    return op.forward(
            A,
            B,
            output=C,
            bias=None,
            input_scale=None,
            weight_scale=None,
            output_scale=None,
            tuning = None,
            fast_accum=False,
        )

def main():
    M = 1000
    N = 1000
    K = 1000
    A = torch.randn((M, K), device='cuda', dtype=torch.bfloat16, requires_grad=False)
    B = torch.randn((N, K), device='cuda', dtype=torch.bfloat16, requires_grad=False)
    C = torch.randn((N, K), device='cuda', dtype=torch.bfloat16, requires_grad=False)
    compiled_func = torch.compile(my_func, backend="inductor")
    compiled_func(A, B, C)
    print(C)

if __name__ == '__main__':
    main()