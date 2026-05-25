import os
import sys
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import megakernel as mi
from xop.ops.dsl.pkt_util import TorchRef, PerfReporter, Qwen3Info
from xop.ops.dsl.mk_layers import MkLayers
import xop.util as xutil

ENABLE_PREFETCH = False

if __name__ == "__main__":
    max_batch_size = 1
    batch_size = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.getenv("XOP_HOME", default=None)+"/megakernel/gen", help="Output files directory")
    parser.add_argument("--trace-name", default="qwen3", help="Perfetto trace output name")
    parser.add_argument("--profiling", action="store_true", help="Use Profiler to generate trace")
    parser.add_argument("--nc", action="store_true", help="no-compile: Use the specified compiled library instead of recompiling it")

    args = parser.parse_args()
    world_size = 1
    rank = 0

    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    # model_name = args.model
    torch.set_default_dtype(torch.bfloat16)

    model_tag = "qwen3_06b"
    layers = MkLayers(model_tag, 0, 1, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mk = layers.get_mk()
    layout = layers.get_layout()
    reporter = PerfReporter() 
    # reporter.memory_footprint_simulation(rank)
    
    hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, num_hidden_layers \
        = Qwen3Info.get_basic_params(model_tag)
    splitk = 1 # 8
    
    x_torch = torch.randn((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_rms_norm_torch = torch.randn((1, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_gatedup_torch = torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_down_proj_torch = torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    x = mk.attach_input(torch_tensor=x_torch, name="in")
    w_rms_norm = mk.attach_input(torch_tensor=w_rms_norm_torch, name="w_norm")
    w_gatedup = mk.attach_input(torch_tensor=w_gatedup_torch, name="w_gatedup")
    w_down_proj = mk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj")
    mlp_out = mk.attach_input(torch_tensor=out_torch, name="mlp_out")
    input_tensors = [x_torch, w_rms_norm_torch, w_gatedup_torch, w_down_proj_torch, out_torch]
    
    x_residual = x

    rms_out = mk.new_tensor(dims=(max_batch_size, hidden_size), dtype=mi.bfloat16, name="rms_out", io_category="cuda_tensor")
    
    if ENABLE_PREFETCH:
        prefetch_weight = w_gatedup
        prefetch_layout = (layout.linear1_layout[0][0], 0, 0) # 即rms_norm后还剩下多少的sm可用于塞入预取
        fused_layout = tuple(a + b for a, b in zip(layout.rmsnorm_layout[0], prefetch_layout)), layout.rmsnorm_layout[1]
        mk.rmsnorm_layer(
            input=x,
            weight=w_rms_norm,
            output=rms_out,
            sync_mode=(0, 0, 0),
            layout=fused_layout,
            fused_params=[99, 11, *prefetch_layout],
            fused_tensor=prefetch_weight,
        )
    else:
        mk.rmsnorm_layer(
            input=x,
            weight=w_rms_norm,
            output=rms_out,
            sync_mode=(0, 0, 0),
            layout=layout.rmsnorm_layout,
        )

    # mlp_mid_torch = torch.zeros((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    # mlp_mid = mk.attach_input(torch_tensor=mlp_mid_torch, name="mlp_mid")
    mlp_mid = mk.new_tensor(dims=(max_batch_size, intermediate_size*2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
    mk.linear_layer(
        input=rms_out,
        weight=w_gatedup,
        output=mlp_mid,
        sync_mode=(0, layout.rmsnorm_layout[0][0], 0) if ENABLE_PREFETCH else (0, 0, 0),
        layout=layout.linear1_layout,
    )
    
    # silu_mul_out_torch = torch.zeros((max_batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    # silu_mul_out = mk.attach_input(torch_tensor=silu_mul_out_torch, name="silu_mul_out")
    # mlp_out_torch = silu_mul_out_torch
    silu_mul_out = mk.new_tensor(dims=(max_batch_size, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
    if ENABLE_PREFETCH:
        prefetch_weight = w_down_proj
        prefetch_layout = (layout.linear2_layout[0][0], 0, 0)
        fused_layout = tuple(a + b for a, b in zip(layout.silu_mul_layout[0], prefetch_layout)), layout.silu_mul_layout[1]
        print("fused_layout", fused_layout)
        mk.silu_mul_layer(
            input=mlp_mid,
            output=silu_mul_out,
            sync_mode=(0, 0, 0),
            layout=fused_layout,
            fused_params=[99, 11, *prefetch_layout],
            fused_tensor=prefetch_weight,
        )
    else:
        mk.silu_mul_layer(
            input=mlp_mid,
            output=silu_mul_out,
            sync_mode=(0, 0, 0), # (2, 0, 0)
            layout=layout.silu_mul_layout,
        )
        
    mk.linear_with_residual_layer(
        input=silu_mul_out,
        weight=w_down_proj,
        residual=x_residual,
        output=mlp_out,
        sync_mode=(0, layout.silu_mul_layout[0][0], 0) if ENABLE_PREFETCH else (0, 0, 0),
        layout=layout.linear2_layout,
    )

    layers.compile_load(input_tensors=input_tensors, enable_prefetch=ENABLE_PREFETCH, is_no_compile=args.nc, output_dir=args.output_dir)
    
    ###
    def ref_run():
        return TorchRef.norm_mlp(x_torch[:batch_size], w_rms_norm_torch, w_gatedup_torch, w_down_proj_torch) + x_torch[:batch_size]

    graph, ref_output = TorchRef.compile_capture(ref_run, is_compile=False)
    
    ##
    def torch_ref(iter):
        graph.replay()
        return ref_output

    def target_func(iter):
        mk(batch_size)
        return out_torch[:batch_size]
    
    if not args.profiling:
        xutil.profile(target_func, torch_ref)
    else:
        for i in range(100):
            torch_ref()
        target_func()