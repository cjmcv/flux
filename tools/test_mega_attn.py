import os
import sys
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import megakernel as mi
from xop.ops.dsl.pkt_util import TorchRef, PerfReporter, Qwen3Info
from xop.ops.dsl.mk_layers import MkLayers
import xop.util as xutil

# w_cos_torch, w_sin_torch针对step，每个step一份，所有层共享
def ref_run(step, key_cache, value_cache, x_torch, w_layernorm_torch, w_qkv_proj_torch, w_q_norm_torch, w_k_norm_torch, w_cos_torch, w_sin_torch, w_o_proj_torch):
    o1 = TorchRef.rms_norm(x_torch, w_layernorm_torch)
    qkv_out = TorchRef.linear(o1, w_qkv_proj_torch)
    # print("qkv_out", qkv_out)
    # print("qkv_proj", x_torch.size(), w_qkv_proj_torch.size()) # qkv_proj torch.Size([1, 1024]) torch.Size([4096, 1024])
    
    q_dim = num_heads*head_dim
    kv_dim = num_kv_heads*head_dim
    query_states = qkv_out[:, :q_dim].view(batch, q_seqlen, num_heads, head_dim) 
    key_states = qkv_out[:, q_dim:q_dim+kv_dim].view(batch, q_seqlen, num_kv_heads, head_dim) 
    value_states = qkv_out[:, q_dim+kv_dim:].view(batch, q_seqlen, num_kv_heads, head_dim) 
    
    query_states = TorchRef.rms_norm(query_states, w_q_norm_torch)
    key_states = TorchRef.rms_norm(key_states, w_k_norm_torch)
    query_states, key_states = TorchRef.apply_rotary_pos_emb_triton(query_states, key_states, w_cos_torch, w_sin_torch, unsqueeze_dim=2)
    
    key_cache[0, step, :, :] = key_states
    value_cache[0, step, :, :] = value_states
    k_slice = key_cache[:, :step+1, :, :]
    v_slice = value_cache[:, :step+1, :, :]
    # k_slice.zero_()
    # v_slice.zero_()
    attn_output = TorchRef.attention_sdpa(query_states, k_slice, v_slice, False)
    attn_output = attn_output.reshape(batch*q_seqlen, q_dim)
    final_output = TorchRef.linear(attn_output, w_o_proj_torch) + x_torch # res
    # print("torch", key_states, value_states, final_output)
    # print("torch:", query_states, "\n", key_states, "\n", value_states, "\n", attn_output, "\n", final_output)
    # print("o_proj", attn_output.size(), w_o_proj_torch.size()) # o_proj torch.Size([1, 1024]) torch.Size([1024, 2048])
    return final_output
    
if __name__ == "__main__":
    max_batch_size = 1
    batch = 1
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
    reporter = PerfReporter() 
    # reporter.memory_footprint_simulation(rank)
    # q_proj: torch.Size([2048, 1024])
    # k_proj: torch.Size([1024, 1024])
    # v_proj: torch.Size([1024, 1024])
    # o_proj: torch.Size([1024, 2048])
    
    hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, num_hidden_layers \
        = Qwen3Info.get_basic_params(model_tag)

    q_seqlen = 1
    max_kv_seqlen = 8192
    x_torch = torch.randn((batch, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_layernorm_torch = torch.randn((1, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    w_qkv_proj_torch = torch.randn(((num_heads+2*num_kv_heads)*head_dim, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_q_norm_torch = torch.randn((1, head_dim), dtype=torch.bfloat16, device="cuda")
    w_k_norm_torch = torch.randn((1, head_dim), dtype=torch.bfloat16, device="cuda")
    w_qk_norm_torch = torch.cat([w_q_norm_torch, w_k_norm_torch], dim=0).contiguous()
    
    cos_half = torch.randn((batch, q_seqlen, head_dim//2), dtype=torch.bfloat16, device="cuda")
    sin_half = torch.randn((batch, q_seqlen, head_dim//2), dtype=torch.bfloat16, device="cuda")
    w_cos_torch = torch.cat((cos_half, cos_half), dim=-1).contiguous()
    w_sin_torch = torch.cat((sin_half, sin_half), dim=-1).contiguous()
    w_o_proj_torch = torch.randn((hidden_size, num_heads*head_dim), dtype=torch.bfloat16, device="cuda")
    
    ###
    layer_num = 10
    key_cache_5dim_torch = torch.zeros(layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen_kv,  H=groups, D=dim]
    value_cache_5dim_torch = torch.zeros(layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    
    # 
    q_dim = num_heads*head_dim
    kv_dim = num_kv_heads*head_dim    
    qkv_proj_out_torch = torch.zeros(max_batch_size, q_dim+2*kv_dim, dtype=torch.bfloat16, device="cuda")
    qk_torch = {
        "2d": qkv_proj_out_torch[:, :q_dim+kv_dim].view(batch*q_seqlen*(num_heads+num_kv_heads), head_dim) 
    }
    q_torch = {
        "2d": qkv_proj_out_torch[:, :q_dim].view(batch*q_seqlen*num_heads, head_dim),
        "3d": qkv_proj_out_torch[:, :q_dim].view(batch*q_seqlen, num_heads, head_dim),
        "4d": qkv_proj_out_torch[:, :q_dim].view(batch, q_seqlen, num_heads, head_dim),
    }
    k_torch = {
        "2d": qkv_proj_out_torch[:, q_dim:q_dim+kv_dim].view(batch*q_seqlen*num_kv_heads, head_dim),
        "4d": qkv_proj_out_torch[:, q_dim:q_dim+kv_dim].view(batch, q_seqlen, num_kv_heads, head_dim),
    }
    v_torch = {
        "4d": qkv_proj_out_torch[:, q_dim+kv_dim:].view(batch, q_seqlen, num_kv_heads, head_dim),
    }
    
    max_attn_split = 8
    edge_torch = torch.empty(10, device="cuda", dtype=torch.int32)
    glse_torch = torch.empty(batch, num_heads, max_attn_split, device="cuda", dtype=torch.bfloat16)
    out_partial_torch = torch.empty(batch, num_heads, max_attn_split, head_dim, device="cuda", dtype=torch.bfloat16)
    mask_torch = torch.ones(batch, max_kv_seqlen, num_kv_heads, device="cuda", dtype=torch.uint8)
    
    attn_out_3dim_torch = torch.empty(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    attn_out_2dim_torch = attn_out_3dim_torch.view(batch*q_seqlen, q_dim)
    
    out_torch = torch.zeros((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")    
    
    #########################################
    
    layer_id = 5
    layers = MkLayers(model_tag, 0, 1, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mk = layers.get_mk()
    layout = layers.get_layout()

    x = mk.attach_input(torch_tensor=x_torch, name="in")
    #    rmsnorm (x) -> linear (qkv proj) -> rmsnorm (q/k) -> rope (q/k) -> attn -> linear_res
    # -> update cos/sin (step)                             -> update kvcache (step)
    w_layernorm = mk.attach_input(torch_tensor=w_layernorm_torch, name="w_layernorm")
    layernorm_out = mk.new_tensor(dims=(max_batch_size, hidden_size), dtype=mi.bfloat16, name="layernorm_out", io_category="cuda_tensor")
    # qkv proj
    w_qkv_proj = mk.attach_input(torch_tensor=w_qkv_proj_torch, name="w_qkv_proj")
    qkv_proj_out = mk.attach_input(torch_tensor=qkv_proj_out_torch, name="qkv_proj_out")
    # qk norm
    if 1:
        w_qk_norm = mk.attach_input(torch_tensor=w_qk_norm_torch, name="w_qk_norm")
        qk_states = mk.attach_input(torch_tensor=qk_torch["2d"], name="qk_states")
    else:
        w_q_norm = mk.attach_input(torch_tensor=w_q_norm_torch, name="w_q_norm")
        w_k_norm = mk.attach_input(torch_tensor=w_k_norm_torch, name="w_k_norm")
        query_states = mk.attach_input(torch_tensor=q_torch["2d"], name="query_states")
        key_states = mk.attach_input(torch_tensor=k_torch["2d"], name="key_states")
    # rope
    rope_in_q = mk.attach_input(torch_tensor=q_torch["4d"], name="rope_in_q")
    rope_in_k = mk.attach_input(torch_tensor=k_torch["4d"], name="rope_in_k")
    w_cos = mk.attach_input(torch_tensor=w_cos_torch, name="cos")
    w_sin = mk.attach_input(torch_tensor=w_sin_torch, name="sin")
    # attn
    attn_in_q = mk.attach_input(torch_tensor=q_torch["3d"], name="attn_in_q")
    attn_in_kcache = mk.attach_input(torch_tensor=key_cache_5dim_torch, name="attn_in_kcache")
    attn_in_vcache = mk.attach_input(torch_tensor=value_cache_5dim_torch, name="attn_in_vcache")
    edge = mk.attach_input(torch_tensor=edge_torch, name="edge")
    attn_in_mask = mk.attach_input(torch_tensor=mask_torch, name="attn_in_mask")
    attn_in_glse = mk.attach_input(torch_tensor=glse_torch, name="attn_in_glse")
    attn_out_partial = mk.attach_input(torch_tensor=out_partial_torch, name="attn_out_partial")
    attn_out_3dim = mk.attach_input(torch_tensor=attn_out_3dim_torch, name="attn_out_3dim") # 
    attn_out_2dim = mk.attach_input(torch_tensor=attn_out_2dim_torch, name="attn_out_2dim")
    # linear_res
    w_o_proj = mk.attach_input(torch_tensor=w_o_proj_torch, name="w_o_proj")
    o_proj_res_out = mk.attach_input(torch_tensor=out_torch, name="o_proj_res_out")
    
    input_tensors = [x_torch, w_layernorm_torch, w_qkv_proj_torch, qkv_proj_out_torch, w_qk_norm_torch, qk_torch["2d"],
                     q_torch["4d"], k_torch["4d"], w_cos_torch, w_sin_torch,
                     q_torch["3d"], key_cache_5dim_torch, value_cache_5dim_torch, edge_torch,
                     mask_torch, glse_torch, out_partial_torch, attn_out_3dim_torch, attn_out_2dim_torch, 
                     w_o_proj_torch, out_torch]
    #########################################
    
    mk.rmsnorm_layer(
        input=x,
        weight=w_layernorm,
        output=layernorm_out,
        sync_mode=(0, 0, 0),
        layout=layout.rmsnorm_layout,
    )
    mk.linear_layer(
        input=layernorm_out,
        weight=w_qkv_proj,
        output=qkv_proj_out,
        sync_mode=(0, 0, 0),
        layout=layout.qkv_proj_layout,
    )
    
    if 1:
        mk.rmsnorm_layer(
            input=qk_states,
            weight=w_qk_norm,
            output=qk_states,
            sync_mode=(0, 0, 0),
            layout=layout.merge_q_k_norm_layout,
        )
    else:
        mk.rmsnorm_layer(
            input=query_states,
            weight=w_q_norm,
            output=query_states,
            sync_mode=(0, 0, 0),
            layout=layout.q_norm_layout,
        )
        mk.rmsnorm_layer(
            input=key_states,
            weight=w_k_norm,
            output=key_states,
            sync_mode=(0, 0, 0),
            layout=layout.k_norm_layout,
        )
    
    # rope
    extra_layout = (2, 0, 0)
    fused_layout = tuple(a + b for a, b in zip(layout.rope_layout[0], extra_layout)), layout.rope_layout[1]
    mk.rope_layer(
        q=rope_in_q,
        k=rope_in_k,
        cos=w_cos,
        sin=w_sin,
        q_embed=rope_in_q,
        k_embed=rope_in_k,
        sync_mode=(0, 0, 0),
        layout=fused_layout,
        fused_params=[99, 1, *extra_layout],
    )
    
    # attn    
    mk.gqa_decode_layer(
        q=attn_in_q,
        k_cache=attn_in_kcache,
        v_cache=attn_in_vcache,
        edge=edge,
        mask=attn_in_mask,
        glse=attn_in_glse,
        out_partial=attn_out_partial,
        output=attn_out_3dim,
        sync_mode=(0, 0, 0),
        layout=layout.gqa_decode_layout_shortkv,
        fused_params=[99, 0],
    )
    mk.linear_with_residual_layer(
        input=attn_out_2dim,
        weight=w_o_proj,
        residual=x,
        output=o_proj_res_out,
        sync_mode=(0, 0, 0),
        layout=layout.linear2_layout,
    )
    
    step = 15
    edge_torch[0].fill_(step) # kv_seqlen
    edge_torch[1].fill_(num_kv_heads*head_dim) # kvcache onestep_size
    edge_torch[2].fill_(batch*max_kv_seqlen*num_kv_heads*head_dim) # kvcache onelayer_size
    meta = [edge_torch, key_cache_5dim_torch, value_cache_5dim_torch, k_torch["4d"], v_torch["4d"]]
    layers.compile_load(input_tensors=input_tensors, meta_tensors=meta, is_no_compile=args.nc, output_dir=args.output_dir)

    print(key_cache_5dim_torch[layer_id, 0, step, :, :].data_ptr(), value_cache_5dim_torch[layer_id, 0, step, :, :].data_ptr())
    mk(batch)

    edge_torch[0].fill_(step)
    def mk_run(iter):
        mk(batch)
        return out_torch
    
    key_cache = torch.zeros(batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)  # [B, N=seqlen_kv,  H=groups, D=dim]
    value_cache = torch.zeros(batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    def torch_ref_tmp():
        return ref_run(step, key_cache, value_cache, x_torch, 
                       w_layernorm_torch, w_qkv_proj_torch, w_q_norm_torch, w_k_norm_torch, w_cos_torch, w_sin_torch, w_o_proj_torch)

    graph, ref_output = TorchRef.compile_capture(torch_ref_tmp, is_compile=True)
    def torch_ref(iter):
        graph.replay()
        return ref_output
    
    for i in range(100):
        torch_ref(0)
        
    mk_run(0)
    xutil.profile(mk_run, torch_ref)
