import torch
import torch.nn as nn
import torch.nn.functional as F
from xop.project.qwen3_4b_h20_compile import XopGemmSpecify

ENABLE_XOP = 1

class LinearLayer(nn.Module):
    """A custom linear layer implementation using functional linear"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with random values and convert to bfloat16
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16, device="cuda") * 0.1)
        self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.bfloat16, device="cuda") * 0.1)
        
        if ENABLE_XOP:
            print(self.weight.device)
            self.xop_gemm = XopGemmSpecify(self.weight, input_dtype=torch.bfloat16, output_dtype=torch.bfloat16, fast_accum=False)

            
    def forward(self, x):
        # Use functional linear for inference computation
        if ENABLE_XOP:
            run_mode = self.xop_gemm.get_run_mode(x.shape[0], self.weight.shape[0], x.shape[1])
            run_mode = 1
            return self.xop_gemm.forward(run_mode, x, self.weight)
        else:
            return F.linear(x, self.weight, self.bias)

class MultiLinearModel(nn.Module):
    """Model containing multiple linear layers"""
    
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Create multiple linear layers based on layer_sizes
        for i in range(len(layer_sizes) - 1):
            self.layers.append(LinearLayer(layer_sizes[i], layer_sizes[i + 1]))
            
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ReLU activation for all but the last layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device to CUDA for GPU acceleration
    device = torch.device('cuda')
    
    # Enable mixed precision training for bfloat16
    torch.set_float32_matmul_precision('high')
    
    # Define model architecture: input_size -> 512 -> 256 -> 128 -> output_size
    layer_sizes = [4096, 4096, 4096, 4096, 128]
    model = MultiLinearModel(layer_sizes).to(device, dtype=torch.bfloat16)
    
    # Print model dtype information
    print(f"Model parameters dtype: {next(model.parameters()).dtype}")
    
    # Compile the model using torch.compile for performance optimization
    print("Compiling model with torch.compile...")
    model = torch.compile(model)
    
    # Create sample input data in bfloat16
    batch_size = 16
    input_tensor = torch.randn(batch_size, layer_sizes[0], device=device, dtype=torch.bfloat16)
    print(f"Input tensor dtype: {input_tensor.dtype}")
    
    # Warm up the model (important for CUDA graphs)
    print("Warming up model...")
    for _ in range(10):
        _ = model(input_tensor)
    
    # Create and use CUDA graph for optimized execution
    print("Creating CUDA graph...")
    graph = torch.cuda.CUDAGraph()
    
    # Set model to evaluation mode for inference
    model.eval()
    
    # Capture the computation in CUDA graph
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            output = model(input_tensor)
    
    # Execute the captured graph
    print("Executing CUDA graph...")
    graph.replay()
    
    # Print output statistics
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)  
      
    TEST_ROUNDS = 2000
    with stream:
        start_event.record()
        for _ in range(TEST_ROUNDS):
            graph.replay()
        end_event.record()
    stream.synchronize()
    total_elapsed_ms = start_event.elapsed_time(end_event)
    avg_elapsed_ms = total_elapsed_ms / TEST_ROUNDS
    print(f"total_elapsed_ms: {total_elapsed_ms:.3f} ms")
    print(f"avg_elapsed_ms: {avg_elapsed_ms:.6f} ms")

    # # Verify the model contains multiple linear layers
    # print(f"\nModel has {len(model.layers)} linear layers")
    # for i, layer in enumerate(model.layers):
    #     print(f"Layer {i}: {layer.in_features} -> {layer.out_features}")
    #     print(f"  Weight dtype: {layer.weight.dtype}, Bias dtype: {layer.bias.dtype}")