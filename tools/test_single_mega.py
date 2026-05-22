import os
import sys
import torch
import argparse
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import megakernel as mi

from xop.ops.dsl.pkt_util import TorchRef, PerfReporter, Qwen3Info
from xop.ops.dsl.mk_layers import MkLayers

from xop.ops.dsl.micro_base import HparamSelectMode
from xop.ops.dsl.micro_autogen import MicroAutoGen

def test_parallel_rms_norm(mk, layout, max_batch_size, batch_size, hidden_size):
    M = 16
    M2 = 8
    head_dim = 128
    x_torch = torch.randn((M+M2, head_dim), dtype=torch.bfloat16, device="cuda")
    w_rms_norm_torch = torch.randn((2, head_dim), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.empty((M+M2, head_dim), dtype=torch.bfloat16, device="cuda")
    
    x = mk.attach_input(torch_tensor=x_torch, name="in")
    w_rms_norm = mk.attach_input(torch_tensor=w_rms_norm_torch, name="w_norm")
    rms_out = mk.attach_input(torch_tensor=out_torch, name="rms_out")
    input_tensors = [x_torch, w_rms_norm_torch, out_torch]
    
    mk.rmsnorm_layer(
        input=x,
        weight=w_rms_norm,
        output=rms_out,
        sync_mode=(0, 0, 0),
        layout=layout.merge_q_k_norm_layout,
    )
    
    layers.compile_load(input_tensors=input_tensors, is_no_compile=args.nc, output_dir=args.output_dir)

    def torch_ref():
        return TorchRef.rms_norm(x_torch[:batch_size], w_rms_norm_torch)
    
    a = x_torch
    b = w_rms_norm_torch
    a1 = a[ : M, :]
    a2 = a[M : M+M2, :]
    b1 = b[0 : 1, :]
    b2 = b[1 : 2, :]
    def torch_ref():
        c1 = TorchRef.rms_norm(a1, b1) 
        c2 = TorchRef.rms_norm(a2, b2) 
        return torch.cat([c1, c2], dim=0)
    def target_func():
        mk(batch_size)
        return out_torch
    
    reporter.generate_report(target_func, torch_ref,
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=1)
    
def test_rms_norm(mk, layout, max_batch_size, batch_size, hidden_size):
    x_torch = torch.randn((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_rms_norm_torch = torch.randn((1, hidden_size), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.randn((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    x = mk.attach_input(torch_tensor=x_torch, name="in")
    w_rms_norm = mk.attach_input(torch_tensor=w_rms_norm_torch, name="w_norm")
    rms_out = mk.attach_input(torch_tensor=out_torch, name="rms_out")
    input_tensors = [x_torch, w_rms_norm_torch, out_torch]
    mk.rmsnorm_layer(
        input=x,
        weight=w_rms_norm,
        output=rms_out,
        sync_mode=(0, 0, 0),
        layout=layout.rmsnorm_layout,
    )
    layers.compile_load(input_tensors=input_tensors, is_no_compile=args.nc, output_dir=args.output_dir)

    def torch_ref():
        return TorchRef.rms_norm(x_torch[:batch_size], w_rms_norm_torch)

    def target_func():
        mk(batch_size)
        return out_torch[:batch_size]
    
    reporter.generate_report(target_func, torch_ref,
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=1)

def test_linear(mk, max_batch_size, batch_size, N, K, spec_layout):
    x_torch = torch.randn((max_batch_size, K), dtype=torch.bfloat16, device="cuda")
    # w_torch = w_gatedup_torch 
    w_torch = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, N), dtype=torch.bfloat16, device="cuda")
    print("x: ", x_torch.data_ptr(), "w: ", w_torch.data_ptr(), "o: ", out_torch.data_ptr())
    
    x = mk.attach_input(torch_tensor=x_torch, name="in")
    w = mk.attach_input(torch_tensor=w_torch, name="w")
    linear_out = mk.attach_input(torch_tensor=out_torch, name="linear_out")
    input_tensors = [x_torch, w_torch, out_torch]
    
    # grid_dim, block_dim(实际是tile_dim), thread_num(实际是block_dim, 固定为threadIdx.x==128或256, 其他维度为1)
    
    mk.linear_layer(
        input=x,
        weight=w,
        output=linear_out,
        sync_mode=(0, 0, 0),
        layout=spec_layout,
    )
    layers.compile_load(input_tensors=input_tensors, is_no_compile=args.nc, output_dir=args.output_dir)
    
    def torch_ref():
        return TorchRef.linear(x_torch[:batch_size], w_torch)
    
    def target_func():
        mk(batch_size)
        return out_torch[:batch_size]
        
    # target_output = target_func()
    # ref_output = torch_ref()
    reporter.generate_report(target_func, torch_ref, 
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=1)

def test_silu_mul(mk, layout, max_batch_size, batch_size, intermediate_size):
    x_torch = torch.randn((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    
    x = mk.attach_input(torch_tensor=x_torch, name="in")
    silu_mul_out = mk.attach_input(torch_tensor=out_torch, name="silu_mul_out")
    input_tensors = [x_torch, out_torch]
    
    mk.silu_mul_layer(
        input=x,
        output=silu_mul_out,
        sync_mode=(0, 0, 0), # (2, 0, 0)
        layout=layout.silu_mul_layout,
    )
    layers.compile_load(input_tensors=input_tensors, is_no_compile=args.nc, output_dir=args.output_dir)

    def torch_ref():
        return TorchRef.silu_and_mul(x_torch[:batch_size])
    
    def target_func():
        mk(batch_size)
        return out_torch[:batch_size]
        
    # target_output = target_func()    
    # ref_output = torch_ref()
    reporter.generate_report(target_func, torch_ref, 
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=1)
    
def test_linear_residual(mk, max_batch_size, batch_size, N, K, spec_layout):
    x_residual_torch = torch.randn((max_batch_size, N), dtype=torch.bfloat16, device="cuda")
    x_torch = torch.randn((max_batch_size, K), dtype=torch.bfloat16, device="cuda")
    w_down_proj_torch = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, N), dtype=torch.bfloat16, device="cuda")

    x_residual = mk.attach_input(torch_tensor=x_residual_torch, name="res")
    x = mk.attach_input(torch_tensor=x_torch, name="in")
    w_down_proj = mk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj")
    mlp_out = mk.attach_input(torch_tensor=out_torch, name="mlp_out")
    input_tensors = [x_residual_torch, x_torch, w_down_proj_torch, out_torch]
    
    mk.linear_with_residual_layer(
        input=x,
        weight=w_down_proj,
        residual=x_residual,
        output=mlp_out,
        sync_mode=(0, 0, 0),
        layout=spec_layout,
    )
    layers.compile_load(input_tensors=input_tensors, is_no_compile=args.nc, output_dir=args.output_dir)
    
    def torch_ref():
        return TorchRef.linear(x_torch[:batch_size], w_down_proj_torch) + x_residual_torch

    def target_func():
        mk(batch_size)
        return out_torch[:batch_size]
        
    target_output = target_func()    
    ref_output = torch_ref()
    reporter.generate_report(target_func, torch_ref, 
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=1)
   
def test_rope(mk, layout, max_batch_size, batch, num_heads, num_kv_heads, head_dim):
    seqlen = 1
    q_torch = torch.randn(batch, seqlen, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen=1, H=num_heads,  D=head_dim]
    k_torch = torch.randn(batch, seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)   # [B, N=seqlen=1, H=num_kv_heads, D=head_dim]
    cos_half = torch.randn(batch, seqlen, head_dim//2, device="cuda", dtype=torch.bfloat16)
    sin_half = torch.randn(batch, seqlen, head_dim//2, device="cuda", dtype=torch.bfloat16)
    cos_torch = torch.cat((cos_half, cos_half), dim=-1)
    sin_torch = torch.cat((sin_half, sin_half), dim=-1)
    q_out_torch = torch.empty(batch, seqlen, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k_out_torch = torch.empty(batch, seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    
    # print("torch: ", q_torch.data_ptr(), k_torch.data_ptr(), v_torch.data_ptr(), mask_torch.data_ptr(), out_torch.data_ptr())
    q = mk.attach_input(torch_tensor=q_torch, name="q")
    k = mk.attach_input(torch_tensor=k_torch, name="k")
    cos = mk.attach_input(torch_tensor=cos_torch, name="cos")
    sin = mk.attach_input(torch_tensor=sin_torch, name="sin")
    q_out = mk.attach_input(torch_tensor=q_out_torch, name="q_out")
    k_out = mk.attach_input(torch_tensor=k_out_torch, name="k_out")
    input_tensors = [q_torch, k_torch, cos_torch, sin_torch, q_out_torch, k_out_torch]
    
    mk.rope_layer(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        q_embed=q_out,
        k_embed=k_out,
        sync_mode=(0, 0, 0),
        layout=layout.rope_layout,
    )
    layers.compile_load(input_tensors=input_tensors, is_no_compile=args.nc, output_dir=args.output_dir)
    
    def target_func():
        mk(batch_size)
        return torch.cat((q_out_torch, k_out_torch), dim=-2)
    
    def torch_ref():
        q_emb, k_emb = TorchRef.apply_rotary_pos_emb_triton(q_torch, k_torch, cos_torch, sin_torch, position_ids=None, unsqueeze_dim=2)
        return torch.cat((q_emb, k_emb), dim=-2)
    
    target_output = target_func()    
    ref_output = torch_ref()
    print("target_output", target_output)
    print("ref_output", ref_output)
    
    print("target_output", target_func())
    print("ref_output", torch_ref())
    # print("target_output", target_func())
    # print("ref_output", torch_ref())
    # print("target_output", target_func())
    # print("ref_output", torch_ref())
    # if (torch.allclose(out_torch, ref_output, rtol=1e-2, atol=0)):
    #     print("allclose: True")
    
    reporter.generate_report(target_func, torch_ref, 
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=1)
     
def test_rope_fused(mk, layout, max_batch_size, batch, num_heads, num_kv_heads, head_dim):
    seqlen = 1
    q_torch = torch.randn(batch, seqlen, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen=1, H=num_heads,  D=head_dim]
    k_torch = torch.randn(batch, seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)   # [B, N=seqlen=1, H=num_kv_heads, D=head_dim]
    cos_half = torch.randn(batch, seqlen, head_dim//2, device="cuda", dtype=torch.bfloat16)
    sin_half = torch.randn(batch, seqlen, head_dim//2, device="cuda", dtype=torch.bfloat16)
    cos_torch = torch.cat((cos_half, cos_half), dim=-1)
    sin_torch = torch.cat((sin_half, sin_half), dim=-1)
    q_out_torch = torch.empty(batch, seqlen, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k_out_torch = torch.empty(batch, seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    
    # print("torch: ", q_torch.data_ptr(), k_torch.data_ptr(), v_torch.data_ptr(), mask_torch.data_ptr(), out_torch.data_ptr())
    q = mk.attach_input(torch_tensor=q_torch, name="q")
    k = mk.attach_input(torch_tensor=k_torch, name="k")
    cos = mk.attach_input(torch_tensor=cos_torch, name="cos")
    sin = mk.attach_input(torch_tensor=sin_torch, name="sin")
    q_out = mk.attach_input(torch_tensor=q_out_torch, name="q_out")
    k_out = mk.attach_input(torch_tensor=k_out_torch, name="k_out")
    input_tensors = [q_torch, k_torch, cos_torch, sin_torch, q_out_torch, k_out_torch]
    
    layer_num = 10
    max_kv_seqlen = 8192
    key_cache_torch = torch.zeros(layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen_kv,  H=groups, D=dim]
    value_cache_torch = torch.zeros(layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    # key_cache_curstep_torch = torch.randn(batch, seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen_kv,  H=groups, D=dim]
    value_cache_curstep_torch = torch.randn(batch, seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    
    layer_id = 5
    step = 100
    extra_layout = (2, 0, 0)
    fused_layout = tuple(a + b for a, b in zip(layout.rope_layout[0], extra_layout)), layout.rope_layout[1]
    mk.rope_layer(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        q_embed=q_out,
        k_embed=k_out,
        sync_mode=(0, 0, 0),
        layout=fused_layout,
        fused_params=[99, 0, *extra_layout, layer_id],
    )
    
    # print("hello", num_kv_heads*head_dim, key_cache_torch.data_ptr(), value_cache_torch.data_ptr())
    edge_torch = torch.empty(10, device="cuda", dtype=torch.int32)
    edge_torch[0].fill_(step) # step, 动态更新复用
    edge_torch[1].fill_(num_kv_heads*head_dim) # kvcache onestep_size
    edge_torch[2].fill_(batch*max_kv_seqlen*num_kv_heads*head_dim) # kvcache onelayer_size
    meta = [edge_torch, key_cache_torch, value_cache_torch, k_torch, value_cache_curstep_torch]
    layers.compile_load(input_tensors=input_tensors, meta_tensors=meta, is_no_compile=args.nc, output_dir=args.output_dir)
    
    print("ptr1", key_cache_torch[layer_id,:,step,:,:].data_ptr(), 
          value_cache_torch[layer_id,:,step,:,:].data_ptr(),
          value_cache_curstep_torch.data_ptr())
    
    inner_k_out_torch = key_cache_torch[layer_id,:,step,:,:].view(batch, seqlen, num_kv_heads, head_dim)
    print("shape:", inner_k_out_torch.shape)
    def target_func():
        mk(batch_size)
        return torch.cat((q_out_torch, inner_k_out_torch), dim=-2)
    
    def torch_ref():
        q_emb, k_emb = TorchRef.apply_rotary_pos_emb_triton(q_torch, k_torch, cos_torch, sin_torch, position_ids=None, unsqueeze_dim=2)
        return torch.cat((q_emb, k_emb), dim=-2)
    
    target_output = target_func()    
    ref_output = torch_ref()
    print("target_output", target_output)
    print("ref_output", ref_output)
    print("allclose", torch.allclose(value_cache_torch[layer_id, 0, step, :, :], value_cache_curstep_torch, rtol=0.01, atol=0.01))
    # print("target_output", target_func())
    # print("ref_output", torch_ref())
    
    reporter.generate_report(target_func, torch_ref, 
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=0)

def test_gqa_decode(mk, layout, max_batch_size, batch, num_heads, num_kv_heads, seqlen_kv, head_dim):
    # 1）两个图切换，以适配两个gqa配置？
    # 2）两个gqa布局，按多的配置，部分block可空跑。！！优先尝试
    #    生成代码时，添加优先主动退出条件(如有生成的时split的kernel则进入，不split的blockz直接退出。)
    # 封装attach_input.
    # 3) (暂停，一轮推理内，step不用改，可以两次推理间在外面修改) 
    #    添加 step, 搜 RuntimeConfig的int *step， 将该指针指向 edge 的内存，搜 kernel::gqa_decode_kernel，将input转为 写死的RuntimeConfig的step。
    #    对应 global_runtime_config[kernel_id].infer_cnt 和 global_runtime_config[kernel_id].batch_size
    #    可以在调用mk推理后，外面打印edge的内容来检查step的递增情况。step的递增在static_persistent_kernel里进行，在等待前置信号时确定
    split = 8 # TODO 自动配置
    glse_torch = torch.empty(batch, num_heads, split, device="cuda", dtype=torch.bfloat16)
    out_partial_torch = torch.empty(batch, num_heads, split, head_dim, device="cuda", dtype=torch.bfloat16)
    
    step = 63
    q_torch = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)              # [B, N=seqlen_q=1, H=num_heads,  D=head_dim]
    k_torch = torch.randn(batch, seqlen_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen_kv,  H=num_kv_heads, D=head_dim]
    v_torch = torch.randn(batch, seqlen_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    edge_torch = torch.empty(10, device="cuda", dtype=torch.int32)
    edge_torch[0].fill_(step)
    mask_torch = torch.ones(batch, seqlen_kv, num_kv_heads, device="cuda", dtype=torch.uint8)
    out_torch = torch.empty(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    
    # print("torch: ", q_torch.data_ptr(), k_torch.data_ptr(), v_torch.data_ptr(), mask_torch.data_ptr(), out_torch.data_ptr())
    q = mk.attach_input(torch_tensor=q_torch, name="q")
    k = mk.attach_input(torch_tensor=k_torch, name="k")
    v = mk.attach_input(torch_tensor=v_torch, name="v")
    edge = mk.attach_input(torch_tensor=edge_torch, name="edge")
    mask = mk.attach_input(torch_tensor=mask_torch, name="mask")
    glse = mk.attach_input(torch_tensor=glse_torch, name="glse")
    out_partial = mk.attach_input(torch_tensor=out_partial_torch, name="out_partial")
    attn_out = mk.attach_input(torch_tensor=out_torch, name="attn_out")
    input_tensors = [q_torch, k_torch, v_torch, edge_torch, mask_torch, glse_torch, out_partial_torch, out_torch]
    
    mk.gqa_decode_layer(
        q=q,
        k_cache=k,
        v_cache=v,
        edge=edge,
        mask=mask,
        glse=glse,
        out_partial=out_partial,
        output=attn_out,
        sync_mode=(0, 0, 0),
        layout=layout.gqa_decode_layout_shortkv,
        # layout=((1, 8, 8), (64, 64, 8), (16, 1, 1), (64, 64, 8))
    )
    layers.compile_load(input_tensors=input_tensors, meta_tensors=[edge_torch], is_no_compile=args.nc, output_dir=args.output_dir)
    
    def target_func():
        mk(batch_size)
        return out_torch
    
    k_slice = k_torch[:, :step+1, :, :]
    v_slice = v_torch[:, :step+1, :, :]
    def torch_ref():
        return TorchRef.attention_sdpa(q_torch, k_slice, v_slice, False)
    
    # target_output = target_func()
    # ref_output = torch_ref()
    # print("target_output", target_output)
    # print("ref_output", ref_output)
    
    # print("target_output", target_func())
    # print("ref_output", torch_ref())
    print("step", edge_torch[0])
    # print("target_output", target_func())
    # print("ref_output", torch_ref())
    # print("target_output", target_func())
    # print("ref_output", torch_ref())
    # if (torch.allclose(out_torch, ref_output, rtol=1e-2, atol=0)):
    #     print("allclose: True")
    
    reporter.generate_report(target_func, torch_ref, 
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=0)

def test_gqa_decode_multi_instance(mk, layout, max_batch_size, batch, num_heads, num_kv_heads, seqlen_kv, head_dim):
    # TODO: 添加input_tensors后，未兼容
    split = 8 # TODO 自动配置
    glse_torch = torch.empty(batch, num_heads, split, device="cuda", dtype=torch.bfloat16)
    out_partial_torch = torch.empty(batch, num_heads, split, head_dim, device="cuda", dtype=torch.bfloat16)
    
    q_torch = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)              # [B, N=seqlen_q=1, H=num_heads,  D=head_dim]
    k_torch = torch.randn(batch, seqlen_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen_kv,  H=num_kv_heads, D=head_dim]
    v_torch = torch.randn(batch, seqlen_kv, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    edge_torch = torch.empty(10, device="cuda", dtype=torch.int32)
    mask_torch = torch.ones(batch, seqlen_kv, num_kv_heads, device="cuda", dtype=torch.uint8)
    out_torch = torch.empty(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    
    ##############################################
    # print("torch: ", q_torch.data_ptr(), k_torch.data_ptr(), v_torch.data_ptr(), mask_torch.data_ptr(), out_torch.data_ptr())
    q = mk.attach_input(torch_tensor=q_torch, name="q")
    k = mk.attach_input(torch_tensor=k_torch, name="k")
    v = mk.attach_input(torch_tensor=v_torch, name="v")
    edge = mk.attach_input(torch_tensor=edge_torch, name="edge")
    mask = mk.attach_input(torch_tensor=mask_torch, name="mask")
    glse = mk.attach_input(torch_tensor=glse_torch, name="glse")
    out_partial = mk.attach_input(torch_tensor=out_partial_torch, name="out_partial")
    attn_out = mk.attach_input(torch_tensor=out_torch, name="attn_out")
    input_tensors = [q_torch, k_torch, v_torch, edge_torch, mask_torch, glse_torch, out_partial_torch, out_torch]
    
    mk.gqa_decode_layer(
        q=q,
        k_cache=k,
        v_cache=v,
        edge=edge,
        mask=mask,
        glse=glse,
        out_partial=out_partial,
        output=attn_out,
        sync_mode=(0, 0, 0),
        layout=layout.gqa_decode_layout_shortkv,
        # layout=((1, 8, 8), (64, 64, 8), (16, 1, 1), (64, 64, 8))
    )
    layers.compile_load(input_tensors=input_tensors, meta_tensors=[edge_torch], is_no_compile=args.nc, output_dir=args.output_dir)
    
    ###########################################################################
    
    layers2 = MkLayers("qwen3_06b", 1, 1, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mk2 = layers2.get_mk()
    q = mk2.attach_input(torch_tensor=q_torch, name="q")
    k = mk2.attach_input(torch_tensor=k_torch, name="k")
    v = mk2.attach_input(torch_tensor=v_torch, name="v")
    edge = mk2.attach_input(torch_tensor=edge_torch, name="edge")
    mask = mk2.attach_input(torch_tensor=mask_torch, name="mask")
    glse = mk2.attach_input(torch_tensor=glse_torch, name="glse")
    out_partial = mk2.attach_input(torch_tensor=out_partial_torch, name="out_partial")
    attn_out = mk2.attach_input(torch_tensor=out_torch, name="attn_out")

    mk2.gqa_decode_layer(
        q=q,
        k_cache=k,
        v_cache=v,
        edge=edge,
        mask=mask,
        glse=glse,
        out_partial=out_partial,
        output=attn_out,
        sync_mode=(0, 0, 0),
        layout=layout.gqa_decode_layout_shortkv,
        # layout=((1, 8, 8), (64, 64, 8), (16, 1, 1), (64, 64, 8))
    )
    layers2.compile_load(input_tensors=input_tensors, meta_tensors=[edge_torch], is_no_compile=args.nc, output_dir=args.output_dir+"2")
    
    valid_kv_seqlen1 = 64
    valid_kv_seqlen2 = 1024
    def target_func():
        edge_torch[0].fill_(valid_kv_seqlen1)
        mk(batch_size)
        edge_torch[0].fill_(valid_kv_seqlen2)
        mk2(batch_size)
        return out_torch
    
    k_slice1 = k_torch[:, :valid_kv_seqlen1, :, :]
    v_slice1 = v_torch[:, :valid_kv_seqlen1, :, :]
    k_slice2 = k_torch[:, :valid_kv_seqlen2, :, :]
    v_slice2 = v_torch[:, :valid_kv_seqlen2, :, :]
    def torch_ref():
        TorchRef.attention_sdpa(q_torch, k_slice1, v_slice1, False)
        return TorchRef.attention_sdpa(q_torch, k_slice2, v_slice2, False)
        
    reporter.generate_report(target_func, torch_ref, 
                            warnup_iter=100, test_iter=100, 
                            allclose_iter=5, print_mode=1)
   
def test_replace_weight(mk, max_batch_size, batch_size, N, K, spec_layout):
    # TODO: 添加input_tensors后，未兼容
    x_torch = torch.randn((max_batch_size, K), dtype=torch.bfloat16, device="cuda")
    
    w_torch = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    w_torch2 = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    
    out_torch = torch.zeros((max_batch_size, N), dtype=torch.bfloat16, device="cuda")
    print("x: ", x_torch.data_ptr(), "w: ", w_torch.data_ptr(), "o: ", out_torch.data_ptr())
    
    x = mk.attach_input(torch_tensor=x_torch, name="in")
    w = mk.attach_input(torch_tensor=w_torch, name="w1")
    w2 = mk.attach_input(torch_tensor=w_torch2, name="w2")
    linear_out = mk.attach_input(torch_tensor=out_torch, name="linear_out")

    mk.linear_layer(
        input=x,
        weight=w,
        output=linear_out,
        sync_mode=(0, 0, 0),
        layout=spec_layout,
    )
    
    layers.append_repl_weight_pair(1, "w1", "w2")
    layers.append_repl_weight_pair(1, "w3", "w4")
    layers.append_repl_weight_pair(2, "w1", "w7")
    layers.append_repl_weight_pair(2, "w3", "w8")
    layers.compile_load(is_no_compile=args.nc, output_dir=args.output_dir)
    
    def torch_ref():
        return TorchRef.linear(x_torch[:batch_size], w_torch)
    def torch_ref2():
        return TorchRef.linear(x_torch[:batch_size], w_torch2)
        
    def target_func():
        mk(batch_size, 0)
        return out_torch[:batch_size]
    def target_func2():
        mk(batch_size, 1)
        return out_torch[:batch_size]   
     
    print("mk1: ", target_func())
    print("torch1: ", torch_ref())
    
    print("mk2: ", target_func2())
    print("torch2: ", torch_ref2())
    
    print("mk1: ", target_func())
    print("torch1: ", torch_ref())
    # reporter.generate_report(target_func, torch_ref, 
    #                         warnup_iter=100, test_iter=100, 
    #                         allclose_iter=5, print_mode=1)
 
def test_prefetch_weight(mk, max_batch_size, batch_size, N, K, spec_layout):
    ENABLE_PREFETCH = False
    
    x_torch = torch.randn((max_batch_size, K), dtype=torch.bfloat16, device="cuda")
    w_rms_norm_torch = torch.randn((1, K), dtype=torch.bfloat16, device="cuda")
    rms_out_torch = torch.randn((max_batch_size, K), dtype=torch.bfloat16, device="cuda")
    
    w_torch = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, N), dtype=torch.bfloat16, device="cuda")
    print("x: ", x_torch.data_ptr(), "w: ", w_torch.data_ptr(), "o: ", out_torch.data_ptr())
    
    x = mk.attach_input(torch_tensor=x_torch, name="in")
    w_rms_norm = mk.attach_input(torch_tensor=w_rms_norm_torch, name="w_norm")
    rms_out = mk.attach_input(torch_tensor=rms_out_torch, name="rms_out")
    w_linear = mk.attach_input(torch_tensor=w_torch, name="w")
    linear_out = mk.attach_input(torch_tensor=out_torch, name="linear_out")
    input_tensors = [x_torch, w_rms_norm_torch, rms_out_torch, w_torch, out_torch]
    
    if ENABLE_PREFETCH:
        prefetch_weight = w_linear
        prefetch_layout = (layout.linear1_layout[0][0], 0, 0) # 169 即rms_norm后还剩下多少的sm可用于塞入预取
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
        
    mk.linear_layer(
        input=rms_out,
        weight=w_linear,
        output=linear_out,
        # sync_mode=(0, 1, 0), # y轴式，表示producer只有1个，对应前置算子的block数量
        sync_mode=(0, layout.rmsnorm_layout[0][0], 0) if ENABLE_PREFETCH else (0, 0, 0),
        layout=spec_layout,
    )
    layers.compile_load(input_tensors=input_tensors, enable_prefetch=ENABLE_PREFETCH, is_no_compile=args.nc, output_dir=args.output_dir)
    
    ##
    def ref_core():
        O1 = TorchRef.rms_norm(x_torch[:batch_size], w_rms_norm_torch)
        O2 = TorchRef.linear(O1, w_torch)
        return O2
    ref_graph, ref_output = TorchRef.compile_capture(ref_core, is_compile=False)
    def torch_ref():
        ref_graph.replay()
        return ref_output
    #
    def target_core():
        mk(batch_size)
        return out_torch[:batch_size] 
    graph2, ref_output2 = TorchRef.compile_capture(target_core, is_compile=False)
    def target_func():
        graph2.replay()
        return ref_output2
    ref_output2.zero_()    
    # 
    
    reporter.generate_report(target_core, torch_ref, 
                            warnup_iter=200, test_iter=200, 
                            allclose_iter=5, print_mode=1)
    print("w_torch.data_ptr: ", w_torch.data_ptr())
    
def test_prefetch_weight_residual(mk, max_batch_size, batch_size, N, K, spec_layout):
    ENABLE_PREFETCH = True
    
    x_torch = torch.randn((max_batch_size, K*2), dtype=torch.bfloat16, device="cuda")
    silu_mul_out_torch = torch.zeros((max_batch_size, K), dtype=torch.bfloat16, device="cuda")
    x_residual_torch = torch.randn((max_batch_size, N), dtype=torch.bfloat16, device="cuda")    
    w_down_proj_torch = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, N), dtype=torch.bfloat16, device="cuda")
        
    x = mk.attach_input(torch_tensor=x_torch, name="in")
    silu_mul_out = mk.attach_input(torch_tensor=silu_mul_out_torch, name="silu_mul_out")
    x_residual = mk.attach_input(torch_tensor=x_residual_torch, name="res")
    w_down_proj = mk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj")
    mlp_out = mk.attach_input(torch_tensor=out_torch, name="mlp_out")
    
    input_tensors = [x_torch, silu_mul_out_torch, x_residual_torch, w_down_proj_torch, out_torch]
    
    if ENABLE_PREFETCH:
        prefetch_weight = w_down_proj
        prefetch_layout = (spec_layout[0][0], 0, 0) # 169 即rms_norm后还剩下多少的sm可用于塞入预取
        fused_layout = tuple(a + b for a, b in zip(layout.silu_mul_layout[0], prefetch_layout)), layout.silu_mul_layout[1]
        mk.silu_mul_layer(
            input=x,
            output=silu_mul_out,
            sync_mode=(0, 0, 0), # (2, 0, 0)
            layout=fused_layout,
            fused_params=[99, 11, *prefetch_layout],
            fused_tensor=prefetch_weight,
        )
    else:
        mk.silu_mul_layer(
            input=x,
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
        layout=spec_layout,
    )
    layers.compile_load(input_tensors=input_tensors, enable_prefetch=ENABLE_PREFETCH, is_no_compile=args.nc, output_dir=args.output_dir)
    
    def ref_core():
        O1 = TorchRef.silu_and_mul(x_torch[:batch_size])
        O2 = TorchRef.linear(O1, w_down_proj_torch) + x_residual_torch
        return O2
    ref_graph, ref_output = TorchRef.compile_capture(ref_core, is_compile=False)
    def torch_ref():
        ref_graph.replay()
        return ref_output
    #
    def target_core():
        mk(batch_size)
        return out_torch[:batch_size] 
    graph2, ref_output2 = TorchRef.compile_capture(target_core, is_compile=False)
    def target_func():
        graph2.replay()
        return ref_output2
    ref_output2.zero_()    
    # 
     
    # target_output = target_func()    
    # ref_output = torch_ref()
    reporter.generate_report(target_core, torch_ref, 
                            warnup_iter=200, test_iter=200, 
                            allclose_iter=5, print_mode=1)
   
   
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

    # # 13/16/20/21/22/30/45: 0.118, 24-29
    # micro = MicroAutoGen(1, 1024, 3072)
    # micro.gen_qwen3_mlp(1, HparamSelectMode.TUNED) # HEURISTIC, TUNING, TUNED
       
    reporter = PerfReporter() 
    # model, tokenizer = reporter.memory_footprint_simulation(rank)
    # w_rms_torch, w_gatedup_torch, w_down_proj_torch = reporter.get_weight_qwen3_mlp(layer_id=0)
    
    kernel_num = 1
    model_tag = "qwen3_06b"
    layers = MkLayers(model_tag, 0, kernel_num, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mk = layers.get_mk()
    layout = layers.get_layout()
    
    hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, num_hidden_layers \
        = Qwen3Info.get_basic_params(model_tag)
    seqlen_kv=2048
    
    # test_rms_norm(mk, layout, max_batch_size, batch_size, hidden_size)
    test_linear(mk, max_batch_size, batch_size, intermediate_size*2, hidden_size, layout.linear1_layout)
    # test_silu_mul(mk, layout, max_batch_size, batch_size, intermediate_size) # 5us vs 2us，需要加速
    # test_linear_residual(mk, max_batch_size, batch_size, hidden_size, intermediate_size, layout.linear2_layout)
    
    # test_linear(mk, max_batch_size, batch_size, (num_heads+2*num_kv_heads)*head_dim, hidden_size, layout.qkv_proj_layout)
    # test_rope(mk, layout, max_batch_size=1, batch=1, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
    # test_gqa_decode(mk, layout, max_batch_size=1, batch=1, num_heads=num_heads, num_kv_heads=num_kv_heads, seqlen_kv=seqlen_kv, head_dim=head_dim)
    # test_linear_residual(mk, max_batch_size, batch_size, hidden_size, num_heads*head_dim, layout.o_proj_layout)
    
    #######################################
    # test_replace_weight(mk, max_batch_size, batch_size, intermediate_size*2, hidden_size, layout.linear1_layout) # TODO
    # test_gqa_decode_multi_instance(mk, layout, max_batch_size=1, batch=1, num_heads=num_heads, num_kv_heads=num_kv_heads, seqlen_kv=seqlen_kv, head_dim=head_dim) # TODO
    # test_rope_fused(mk, layout, max_batch_size=1, batch=1, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim) # TODO
    
    # test_parallel_rms_norm(mk, layout, max_batch_size, batch_size, hidden_size)
    # test_prefetch_weight(mk, max_batch_size, batch_size, intermediate_size*2, hidden_size, layout.linear1_layout)
    # test_prefetch_weight_residual(mk, max_batch_size, batch_size, hidden_size, intermediate_size, layout.linear2_layout)
    # print("Test single_mega completed.")
    # ncu --set full --section "SpeedOfLight_RooflineChart" -k "kernel" -o my_profile python demo/single_linear.py --nc
    # ncu --set full --section "SpeedOfLight_RooflineChart" -k "persistent_kernel" -o my_profile python demo/single_linear.py