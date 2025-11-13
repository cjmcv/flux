
import torch
import torch.nn.functional as F
import xop

print(torch.__version__)

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
    
def test_torch_compile(A, B, C):
    print("Start testing torch compile")
    compiled_func = torch.compile(my_func, backend="inductor")
    compiled_func(A, B, C)
    print(C)
    
def test_cuda_graph(A, B, C):
    print("Start testing cuda graph")
    
    def forward_fn():
        op.forward(
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
        
    # pre allocate for cuda graph   
    # forward_fn()
    
    stream = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g):
            forward_fn()
            
    print("Cuda graph replay: ", g.pool())
    g.replay()
    print(C)
    print(g)
    
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        g.replay()
    prof.export_chrome_trace("trace.json")
    
# python tools/test_torch_compile.py
def main():
    M = 1
    N = 4096
    K = 4096
    A = torch.randn((M, K), device='cuda', dtype=torch.bfloat16, requires_grad=False)
    B = torch.randn((N, K), device='cuda', dtype=torch.bfloat16, requires_grad=False)
    C = torch.randn((N, K), device='cuda', dtype=torch.bfloat16, requires_grad=False)
    
    test_cuda_graph(A, B, C)
    print("\n#################\n")
    test_torch_compile(A, B, C)

if __name__ == '__main__':
    main()