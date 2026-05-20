import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from typing import List,Any

from xop.project.qwen3_4b_h20_compile import XopGemmSpecify
# from xop.project.dsl_kernels import splitk_gemv_vectorized_tvm # , fused_gemv_gemv
from models.modeling_qwen3 import Qwen3ForCausalLM

import torch.cuda.nvtx as nvtx       # for torch profiler

from torch._inductor import config   # for debuging torch compile
config.trace.enabled = True          
config.trace.log_autotuning_results = True
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_LOGS"] = "fusion,graph,graph_breaks"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

get_data = torch.randn
# get_data = torch.ones

ENABLE_XOP = 1
ENABLE_CUDAGRAPH = 1
ENABLE_TORCHCOMPILE = 0
ENABLE_MEASURE_OP = 0
ENABLE_TORCH_PROFILER = 1

WARNUP_ROUNDS = 200
TEST_ROUNDS = 5000
# in => k, out => n

class LinearLayer(nn.Module):
    """A custom linear layer implementation using functional linear"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with random values and convert to bfloat16
        self.weight = nn.Parameter(get_data(out_features, in_features, dtype=torch.bfloat16, device="cuda") * 0.1)
        self.bias = nn.Parameter(get_data(out_features, dtype=torch.bfloat16, device="cuda") * 0.1)
        
        if ENABLE_XOP:
            print(self.weight.device)
            self.xop_gemm = XopGemmSpecify(self.weight, input_dtype=torch.bfloat16, output_dtype=torch.bfloat16, fast_accum=False)
            # self.tilelang_gemm = splitk_gemv_vectorized_tvm(self.weight.shape[0], self.weight.shape[1], 2, 32)

    def forward(self, x):
        # Use functional linear for inference computation
        if ENABLE_XOP:
            # run_mode = self.xop_gemm.get_run_mode(x.shape[0], self.weight.shape[0], x.shape[1])
            run_mode = 1
            return self.xop_gemm.forward(run_mode, x, self.weight)
            # return self.tilelang_gemm(x.squeeze(0), self.weight).unsqueeze(0)
        else:
            return F.linear(x, self.weight) #, self.bias

class MultiLinearModel(nn.Module):
    """Model containing multiple linear layers"""
    
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Create multiple linear layers based on layer_sizes
        for i in range(len(layer_sizes) - 1):
            self.layers.append(LinearLayer(layer_sizes[i], layer_sizes[i + 1]))

    @torch.no_grad()
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if ENABLE_MEASURE_OP:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                
            x = layer(x)
            
            if ENABLE_MEASURE_OP:
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
                print(f"Layer {i} forward time: {elapsed:.3f} ms")
        
            # # Apply ReLU activation for all but the last layer
            # if i < len(self.layers) - 1:
            #     x = x[:, :x.shape[1]/2]
            #     # x = F.relu(x)
        return x

# @nvtx.annotate("Graph Replay warmup", color="green")

def profile_one_config(batch_size, layer_sizes, record_prof):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device to CUDA for GPU acceleration
    device = torch.device('cuda')
    
    # Enable mixed precision training for bfloat16
    # torch.set_float32_matmul_precision('high')
    
    # Define model architecture: input_size -> 512 -> 256 -> 128 -> output_size
    model = MultiLinearModel(layer_sizes).to(device, dtype=torch.bfloat16)
    
    # Print model dtype information
    print(f"Model parameters dtype: {next(model.parameters()).dtype}")
    
    # Compile the model using torch.compile for performance optimization
    print("Compiling model with torch.compile...")
    if ENABLE_TORCHCOMPILE:
        # torch.compile(model, backend="eager") # eager / aot_eager / inductor
        compiled_model = torch.compile(model, backend="inductor")
    else:
        compiled_model = model 
    
    # Create sample input data in bfloat16
    input_tensor = get_data(batch_size, layer_sizes[0], device=device, dtype=torch.bfloat16)
    print(f"Input tensor dtype: {input_tensor.dtype}")
    
    # Warm up the model (important for CUDA graphs)
    with nvtx.range("Graph Replay warmup", color="green"):
        print("Warming up model...")
        for _ in range(WARNUP_ROUNDS):
            _ = compiled_model(input_tensor)
        
    if ENABLE_TORCH_PROFILER:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            compiled_model(input_tensor)
        print(prof.key_averages().table(sort_by="cuda_time_total"))
        prof.export_chrome_trace("trace.json") # chrome://tracing/
        
    if ENABLE_MEASURE_OP:
        exit()
    
    # Create and use CUDA graph for optimized execution
    print("Creating CUDA graph...")
    graph = torch.cuda.CUDAGraph()
    
    # Set model to evaluation mode for inference
    compiled_model.eval()

    # Capture the computation in CUDA graph
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            output = compiled_model(input_tensor)
    
    # Execute the captured graph
    print("Executing CUDA graph...")
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    # print(output)
    
    # Print output statistics
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)  
    
    TEST_ROUNDS = 500
    with nvtx.range("Graph Replay testing", color="red"):
        if ENABLE_CUDAGRAPH:    
            with stream:
                start_event.record()
                for _ in range(TEST_ROUNDS):
                    graph.replay()
                end_event.record()
            stream.synchronize()
        else:
            with stream:
                start_event.record()
                for _ in range(TEST_ROUNDS):
                    compiled_model(input_tensor)
                end_event.record()
            stream.synchronize()
        
    total_elapsed_ms = start_event.elapsed_time(end_event)
    avg_elapsed_ms = total_elapsed_ms / TEST_ROUNDS
    print(f"total_elapsed_ms: {total_elapsed_ms:.3f} ms")
    print(f"avg_elapsed_ms: {avg_elapsed_ms:.6f} ms")
    record_prof.append(avg_elapsed_ms)
    # # Verify the model contains multiple linear layers
    # print(f"\nModel has {len(model.layers)} linear layers")
    # for i, layer in enumerate(model.layers):
    #     print(f"Layer {i}: {layer.in_features} -> {layer.out_features}")
    #     print(f"  Weight dtype: {layer.weight.dtype}, Bias dtype: {layer.bias.dtype}")
    
if __name__ == "__main__":
    with torch.device("cuda"):
        model_name = "/home/cjmcv/project/llm_models/Qwen/Qwen3-0.6B"
        model = Qwen3ForCausalLM.from_pretrained(model_name, world_size=1, max_num_pages=16, page_size=4096).to("cuda")
        
    batch_sizes = [1]#,2,4,8,16,32,64,128,256,5121024,2048,4096,8192
    # layer_sizes_list = [[4096, 4096, 4096], [4096, 128]]
    # layer_sizes_list = [[9728, 2560, 6144], [4096, 2560, 19456]]
    # layer_sizes_list = [[9728, 2560], [2560, 6144], [4096, 2560], [2560, 19456]]
    layer_sizes_list = [[2560, 19456], [9728, 2560]]
    # layer_sizes_list = [[2560, 19456, 2560]]

    fc = lambda tflops_list: [round(num, 3) for num in tflops_list]
    record_prof: List[List[Any]] = []
    for layer_sizes in layer_sizes_list:
        prof_one_group: List[Any] = []
        for batch_size in batch_sizes:
            profile_one_config(batch_size, layer_sizes, prof_one_group)
        record_prof.append(prof_one_group)
    
    for idx, layer_sizes in enumerate(layer_sizes_list):
        print(layer_sizes, ":", fc(record_prof[idx]))

    ######################################################################
        
    # def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    #     d = x.shape[-1] // 2
    #     return torch.nn.functional.silu(x[..., :d]) * x[..., d:]
    
    # def test_torch_mlp2(x, w_gatedup, w_down_proj):
    #     import torch.nn.functional as F
    #     O1 = F.linear(x, w_gatedup)
    #     D = silu_and_mul(O1)
    #     return F.linear(D, w_down_proj)

    # batch_size = 1
    # hidden_size = 2560
    # intermediate_size = 9728
    # x_torch = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    # w_gatedup_torch = torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    # w_down_proj_torch = torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")

    # warnup_iter = 20
    # test_iter = 100
    # for _ in range(warnup_iter):
    #     test_torch_mlp2(x_torch, w_gatedup_torch, w_down_proj_torch)
        
    # starter = torch.cuda.Event(enable_timing=True)
    # ender = torch.cuda.Event(enable_timing=True)
    # starter.record()
    # for _ in range(test_iter):
    #     test_torch_mlp2(x_torch, w_gatedup_torch, w_down_proj_torch)
    # ender.record()
    # torch.cuda.synchronize()
    # run_time = starter.elapsed_time(ender)
    # print("torch run time (ms): ", run_time / test_iter)
    
# 0
# [9728, 2560] : [0.266]
# [2560, 6144] : [0.168]
# [4096, 2560] : [0.117]
# [2560, 19456] : [0.53]
# 1
# [4096, 2560, 19456] : [0.651] tile
# [4096, 2560, 19456] : [0.654]

# [9728, 2560] : [0.266]
# [2560, 6144] : [0.169]
# [4096, 2560] : [0.114]
# [2560, 19456] : [0.527]
