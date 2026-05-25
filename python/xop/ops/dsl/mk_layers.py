
import torch
from types import SimpleNamespace
import megakernel as mi
from xop.ops.dsl.pkt_util import Qwen3Info
from xop.ops.dsl.micro_config import get_arch

# model_tag:  "qwen3_06b" / "qwen3_4b"
class MkLayers:
    def __init__(self, model_tag, instance_id, kernel_num, world_size, rank, max_batch_size, trace_name, profiling):
        self.profiler_tensor = None
        if profiling:
            self.profiler_tensor = torch.zeros(3000 * 128, dtype=torch.uint64, device="cuda").contiguous()
            
        # int num_sms_to_use = global_runtime_config[kernel_id].num_workers + num_schedulers / 4;
        num_workers, num_schedulers = mi.get_static_configurations_from_gpu(rank) # n, (sm-n)*4
        print("num_workers: ", num_workers)
        print("num_schedulers: ", num_schedulers)
        
        self.mk = mi.PersistentKernel(
            instance_id=instance_id,
            kernel_num=kernel_num,
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            meta_tensors={}, #  meta_tensors={"qo_indptr_buffer": self.qo_indptr_buffer,},
            profiler_tensor=self.profiler_tensor,
            trace_name=trace_name,
            model_tag=model_tag,
        )
        self.max_batch_size = max_batch_size
        self.w_qkv_proj_torch = []
        self.w_qk_norm_torch = []
        self.w_mlp_gateup_proj = []
        
        self.Qwen3MegaConfig = None
        if (model_tag == "qwen3_4b"):
            if get_arch() == "sm_89":
                from xop.ops.dsl.autogen.sm_89.qwen3_4b_mega_config import Qwen3MegaConfig4b
            elif get_arch() == "sm_120":
                from xop.ops.dsl.autogen.sm_120.qwen3_4b_mega_config import Qwen3MegaConfig4b
            self.Qwen3MegaConfig = Qwen3MegaConfig4b
        elif (model_tag == "qwen3_06b"):
            if get_arch() == "sm_89":
                from xop.ops.dsl.autogen.sm_89.qwen3_06b_mega_config import Qwen3MegaConfig06b
            elif get_arch() == "sm_120":
                from xop.ops.dsl.autogen.sm_120.qwen3_06b_mega_config import Qwen3MegaConfig06b
            self.Qwen3MegaConfig = Qwen3MegaConfig06b

    def get_mk(self):
        return self.mk
    def get_layout(self):
        return self.Qwen3MegaConfig
    
    def append_repl_weight_pair(self, kernel_id, base_weight, target_weight):
        if kernel_id not in self.mk.repl_weight_mapping:
            self.mk.repl_weight_mapping[kernel_id] = []
        self.mk.repl_weight_mapping[kernel_id].append((base_weight, target_weight))
        
    def compile_load(self, input_tensors=list(), meta_tensors=list(), enable_prefetch=False, is_no_compile=False, output_dir="./gen"):
        if is_no_compile is True:
            module_path = output_dir + "/test.cpython-38-x86_64-linux-gnu.so"
            self.mk.load_module(module_path, input_tensors, meta_tensors)
        else:
            module_path = self.mk.compile(enable_prefetch=enable_prefetch, output_dir=output_dir)
            print("module_path: ", module_path)
            self.mk.load_module(module_path, input_tensors, meta_tensors)

    @staticmethod
    def qwen3_alloc_torch_buffer(model_tag, layer_num, batch, q_seqlen, max_kv_seqlen=2048):
        hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, num_hidden_layers \
            = Qwen3Info.get_basic_params(model_tag)
        
        params = SimpleNamespace(
            batch = batch,
            q_seqlen = q_seqlen,
            max_kv_seqlen = max_kv_seqlen,
            hidden_size = hidden_size,
            intermediate_size = intermediate_size,
            num_heads = num_heads,
            num_kv_heads = num_kv_heads,
            head_dim = head_dim,
        )
        # 所有层共享
        public_pt = SimpleNamespace(
            # 在rope中完成更新, layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim
            key_cache_5d = torch.zeros(layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16),  # [B, N=seqlen_kv,  H=groups, D=dim]
            value_cache_5d = torch.zeros(layer_num, batch, max_kv_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16),
            # 在推理前，根据step拷贝进对应的值
            cos = torch.empty((batch, q_seqlen, params.head_dim), dtype=torch.bfloat16, device="cuda"),
            sin = torch.empty((batch, q_seqlen, params.head_dim), dtype=torch.bfloat16, device="cuda"),
        )
        
        #############################################################################        
        # attn layer
        # #    rmsnorm (x) -> linear (qkv proj) -> rmsnorm (q/k) -> rope (q/k) -> attn -> linear_res
        # # -> update cos/sin (step)                             -> update kvcache (step)   
        
        ##################
        # torch tensor
        q_dim = num_heads*head_dim
        kv_dim = num_kv_heads*head_dim   
        max_attn_split = 8 
        qkv_proj_out_torch = torch.zeros(batch, q_dim+2*kv_dim, dtype=torch.bfloat16, device="cuda")
        attn_out_3dim_torch = torch.empty(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        attn_out_2dim_torch = attn_out_3dim_torch.view(batch*q_seqlen, q_dim)
        io_pt = SimpleNamespace(
            x_torch = torch.empty((batch, hidden_size), dtype=torch.bfloat16, device="cuda"),
            layernorm_out_torch = torch.empty((batch, hidden_size), dtype=torch.bfloat16, device="cuda"),
            qkv_proj_out_torch = qkv_proj_out_torch,
            qk_torch = {
                "2d": qkv_proj_out_torch[:, :q_dim+kv_dim].view(batch*q_seqlen*(num_heads+num_kv_heads), head_dim) 
            },
            q_torch = {
                "2d": qkv_proj_out_torch[:, :q_dim].view(batch*q_seqlen*num_heads, head_dim),
                "3d": qkv_proj_out_torch[:, :q_dim].view(batch*q_seqlen, num_heads, head_dim),
                "4d": qkv_proj_out_torch[:, :q_dim].view(batch, q_seqlen, num_heads, head_dim),
            },
            k_torch = {
                "2d": qkv_proj_out_torch[:, q_dim:q_dim+kv_dim].view(batch*q_seqlen*num_kv_heads, head_dim),
                "4d": qkv_proj_out_torch[:, q_dim:q_dim+kv_dim].view(batch, q_seqlen, num_kv_heads, head_dim),
            },
            v_torch = {
                "4d": qkv_proj_out_torch[:, q_dim+kv_dim:].view(batch, q_seqlen, num_kv_heads, head_dim),
            },
            edge_torch = torch.empty(10, device="cuda", dtype=torch.int32),
            glse_torch = torch.empty(batch, num_heads, max_attn_split, device="cuda", dtype=torch.bfloat16),
            out_partial_torch = torch.empty(batch, num_heads, max_attn_split, head_dim, device="cuda", dtype=torch.bfloat16),
            mask_torch = torch.ones(batch, max_kv_seqlen, num_kv_heads, device="cuda", dtype=torch.uint8),
            attn_out_torch = {
                "3d": attn_out_3dim_torch,
                "2d": attn_out_2dim_torch
            },
            o_proj_res_out_torch = torch.zeros((batch, hidden_size), dtype=torch.bfloat16, device="cuda"), # x_torch

            # mlp
            mlp_rms_out_torch = torch.empty((batch, hidden_size), dtype=torch.bfloat16, device="cuda"),
            mlp_mid_torch = torch.empty((batch, intermediate_size*2), dtype=torch.bfloat16, device="cuda"),
            silu_mul_out_torch = torch.empty((batch, intermediate_size), dtype=torch.bfloat16, device="cuda"),
        )
        return params, io_pt, public_pt
        
    # def sync_kvcache_from_model(public_pt, ):
    #     # [layer_num, B, N=seqlen_kv,  H=groups, D=dim]
    #     public_pt.key_cache_5d
    #     public_pt.value_cache_5d
        
    def qwen3_alloc_io_buffer(self, params, io_pt, public_pt):
        self.params = params
        self.io_pt = io_pt
        self.public_pt = public_pt
        
        # mk tensor
        self.attn_layer_io = SimpleNamespace(
            layer_in = SimpleNamespace(pt=self.io_pt.x_torch, mk=self.mk.attach_input(torch_tensor=self.io_pt.x_torch, name="attn_layer_in")),
            layernorm_out = SimpleNamespace(pt=self.io_pt.layernorm_out_torch, mk=self.mk.attach_input(torch_tensor=self.io_pt.layernorm_out_torch, name="layernorm_out")),
            qkv_proj_out = SimpleNamespace(pt=self.io_pt.qkv_proj_out_torch, mk=self.mk.attach_input(torch_tensor=self.io_pt.qkv_proj_out_torch, name="qkv_proj_out")),
            qk_norm_states = SimpleNamespace(pt=None, mk=self.mk.attach_input(torch_tensor=self.io_pt.qk_torch["2d"], name="qk_norm_states")),
            rope_io = SimpleNamespace(
                q = SimpleNamespace(pt=self.io_pt.q_torch["4d"], mk=self.mk.attach_input(torch_tensor=self.io_pt.q_torch["4d"], name="rope_io_q")),
                k = SimpleNamespace(pt=self.io_pt.k_torch["4d"], mk=self.mk.attach_input(torch_tensor=self.io_pt.k_torch["4d"], name="rope_io_k"))
            ),
            kv_curstep = SimpleNamespace(
                k = SimpleNamespace(pt=self.io_pt.k_torch["4d"], mk=self.mk.attach_input(torch_tensor=self.io_pt.k_torch["4d"], name="k_curstep")),
                v = SimpleNamespace(pt=self.io_pt.v_torch["4d"], mk=self.mk.attach_input(torch_tensor=self.io_pt.v_torch["4d"], name="v_curstep")),
            ),
            attn_in = SimpleNamespace(
                q = SimpleNamespace(pt=self.io_pt.q_torch["3d"], mk=self.mk.attach_input(torch_tensor=self.io_pt.q_torch["3d"], name="attn_in_q")),
                kcache = SimpleNamespace(pt=self.public_pt.key_cache_5d, mk=self.mk.attach_input(torch_tensor=self.public_pt.key_cache_5d, name="attn_in_kcache")),
                vcache = SimpleNamespace(pt=self.public_pt.value_cache_5d, mk=self.mk.attach_input(torch_tensor=self.public_pt.value_cache_5d, name="attn_in_vcache")),
                edge = SimpleNamespace(pt=self.io_pt.edge_torch, mk=self.mk.attach_input(torch_tensor=self.io_pt.edge_torch, name="edge")),
                mask = SimpleNamespace(pt=self.io_pt.mask_torch, mk=self.mk.attach_input(torch_tensor=self.io_pt.mask_torch, name="attn_in_mask")),
                glse = SimpleNamespace(pt=self.io_pt.glse_torch, mk=self.mk.attach_input(torch_tensor=self.io_pt.glse_torch, name="attn_in_glse")),
                out_partial = SimpleNamespace(pt=self.io_pt.out_partial_torch, mk=self.mk.attach_input(torch_tensor=self.io_pt.out_partial_torch, name="attn_out_partial"))
            ),
            attn_out = SimpleNamespace(
                three_dim = SimpleNamespace(pt=self.io_pt.attn_out_torch["3d"], mk=self.mk.attach_input(torch_tensor=self.io_pt.attn_out_torch["3d"], name="attn_out_3dim")),
                two_dim = SimpleNamespace(pt=self.io_pt.attn_out_torch["2d"], mk=self.mk.attach_input(torch_tensor=self.io_pt.attn_out_torch["2d"], name="attn_out_2dim")),
            ),
            layer_out = SimpleNamespace(pt=self.io_pt.o_proj_res_out_torch, mk=self.mk.attach_input(torch_tensor=self.io_pt.o_proj_res_out_torch, name="attn_layer_out"))
        )
        
        #############################################################################        
        # mlp layer
        # #    rmsnorm (x) -> linear (gateup_proj) -> act (silu_mul) -> linear_res (down_proj)
        # torch tensor
        # mlp_layer_out_torch = torch.zeros((batch, hidden_size), dtype=torch.bfloat16, device="cuda")
        # mk tensor
        self.mlp_layer_io = SimpleNamespace(
            layer_in = self.attn_layer_io.layer_out,
            layernorm_out = SimpleNamespace(pt=self.io_pt.mlp_rms_out_torch, mk=self.mk.attach_input(torch_tensor=self.io_pt.mlp_rms_out_torch, name="mlp_rms_out")),
            mlp_mid = SimpleNamespace(pt=self.io_pt.mlp_mid_torch, mk=self.mk.attach_input(torch_tensor=self.io_pt.mlp_mid_torch, name="mlp_mid")),
            silu_mul_out = SimpleNamespace(pt=self.io_pt.silu_mul_out_torch, mk=self.mk.attach_input(torch_tensor=self.io_pt.silu_mul_out_torch, name="silu_mul_out")),
            layer_out = self.attn_layer_io.layer_in,
        )
        return self.params, self.public_pt, self.attn_layer_io, self.mlp_layer_io
    
    def qwen3_create_attn_layer(self, model, layer_id, is_long_kv = False, reuse_instance = False):
        # rmsnorm -> qkv_proj -> q/k rmsnorm -> rope -> gqa -> out_proj with_residual
        w_input_layernorm_torch, w_q_norm_torch, w_k_norm_torch, \
        w_q_torch, w_k_torch, w_v_torch, w_out_proj_torch, \
        k_cache_torch, v_cache_torch = Qwen3Info.get_weight_qwen3_attention(model, layer_id)
        
        self.w_qkv_proj_torch.append(torch.cat([w_q_torch, w_k_torch, w_v_torch], dim=0).contiguous())
        self.w_qk_norm_torch.append(torch.cat([w_q_norm_torch, w_k_norm_torch], dim=0).contiguous())
        
        layer_id_str = "_" + str(layer_id) 
        if (reuse_instance == True):    
            self.mk.attach_input(torch_tensor=w_input_layernorm_torch, name="w_layernorm"+layer_id_str)
            self.mk.attach_input(torch_tensor=self.w_qkv_proj_torch[layer_id], name="w_qkv_proj"+layer_id_str)
            self.mk.attach_input(torch_tensor=self.w_qk_norm_torch[layer_id], name="w_qk_norm"+layer_id_str)
            self.mk.attach_input(torch_tensor=w_out_proj_torch, name="w_o_proj"+layer_id_str)
            
            self.append_repl_weight_pair(layer_id, "w_layernorm_0", "w_layernorm"+layer_id_str)
            self.append_repl_weight_pair(layer_id, "w_qkv_proj_0", "w_qkv_proj"+layer_id_str)
            self.append_repl_weight_pair(layer_id, "w_qk_norm_0", "w_qk_norm"+layer_id_str)
            self.append_repl_weight_pair(layer_id, "w_o_proj_0", "w_o_proj"+layer_id_str)
            return 
        
        self.mk.rmsnorm_layer(
            input  = self.attn_layer_io.layer_in.mk,
            weight = self.mk.attach_input(torch_tensor=w_input_layernorm_torch, name="w_layernorm"+layer_id_str),
            output = self.attn_layer_io.layernorm_out.mk,
            sync_mode = (0, 0, 0),
            layout = self.Qwen3MegaConfig.rmsnorm_layout,
        )
        self.mk.linear_layer(
            input  = self.attn_layer_io.layernorm_out.mk,
            weight = self.mk.attach_input(torch_tensor=self.w_qkv_proj_torch[layer_id], name="w_qkv_proj"+layer_id_str),
            output = self.attn_layer_io.qkv_proj_out.mk,
            sync_mode = (0, 0, 0),
            layout = self.Qwen3MegaConfig.qkv_proj_layout,
        )
        self.mk.rmsnorm_layer(
            input  = self.attn_layer_io.qk_norm_states.mk,
            weight = self.mk.attach_input(torch_tensor=self.w_qk_norm_torch[layer_id], name="w_qk_norm"+layer_id_str),
            output = self.attn_layer_io.qk_norm_states.mk,
            sync_mode=(0, 0, 0),
            layout = self.Qwen3MegaConfig.merge_q_k_norm_layout,
        )
        # rope
        extra_layout = (2, 0, 0)
        fused_layout = tuple(a + b for a, b in zip(self.Qwen3MegaConfig.rope_layout[0], extra_layout)), self.Qwen3MegaConfig.rope_layout[1]
        self.mk.rope_layer(
            q=self.attn_layer_io.rope_io.q.mk,
            k=self.attn_layer_io.rope_io.k.mk,
            cos=self.mk.attach_input(torch_tensor=self.public_pt.cos, name="cos"+layer_id_str), # 所有层共享，不需要替换
            sin=self.mk.attach_input(torch_tensor=self.public_pt.sin, name="sin"+layer_id_str),
            q_embed=self.attn_layer_io.rope_io.q.mk,
            k_embed=self.attn_layer_io.rope_io.k.mk,
            sync_mode=(0, 0, 0),
            layout=fused_layout,
            fused_params=[99, 1, *extra_layout],
        )
        
        # attn    
        if is_long_kv == True:
            gqa_decode_layout = self.Qwen3MegaConfig.gqa_decode_layout_longkv
        else:
            gqa_decode_layout = self.Qwen3MegaConfig.gqa_decode_layout_shortkv
        self.mk.gqa_decode_layer(
            q=self.attn_layer_io.attn_in.q.mk,
            k_cache=self.attn_layer_io.attn_in.kcache.mk,
            v_cache=self.attn_layer_io.attn_in.vcache.mk,
            edge=self.attn_layer_io.attn_in.edge.mk,
            mask=self.attn_layer_io.attn_in.mask.mk,
            glse=self.attn_layer_io.attn_in.glse.mk,
            out_partial=self.attn_layer_io.attn_in.out_partial.mk,
            output=self.attn_layer_io.attn_out.three_dim.mk,
            sync_mode=(0, 0, 0),
            layout=gqa_decode_layout,
            fused_params=[99, 0],
        )
        self.mk.linear_with_residual_layer(
            input=self.attn_layer_io.attn_out.two_dim.mk,
            weight=self.mk.attach_input(torch_tensor=w_out_proj_torch, name="w_o_proj"+layer_id_str),
            residual=self.attn_layer_io.layer_in.mk,
            output=self.attn_layer_io.layer_out.mk,
            sync_mode=(0, 0, 0),
            layout=self.Qwen3MegaConfig.linear2_layout,
        )
    
    def qwen3_create_mlp_layer(self, model, layer_id, reuse_instance = False):
        # rmsnorm -> gateup_proj -> silu_mul -> down_proj with res
        w_rms_norm_torch, w_gate_proj, w_up_proj, w_down_proj_torch = Qwen3Info.get_weight_qwen3_mlp(model, layer_id)
        self.w_mlp_gateup_proj.append(torch.cat((w_gate_proj, w_up_proj), 0).contiguous())
        
        layer_id_str = "_" + str(layer_id)
        if (reuse_instance == True):
            self.mk.attach_input(torch_tensor=w_rms_norm_torch, name="w_norm"+layer_id_str)
            self.mk.attach_input(torch_tensor=self.w_mlp_gateup_proj[layer_id], name="w_gatedup"+layer_id_str)
            self.mk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj"+layer_id_str)
            
            self.append_repl_weight_pair(layer_id, "w_norm_0", "w_norm"+layer_id_str)
            self.append_repl_weight_pair(layer_id, "w_gatedup_0", "w_gatedup"+layer_id_str)
            self.append_repl_weight_pair(layer_id, "w_down_proj_0", "w_down_proj"+layer_id_str)
            return 
        
        self.mk.rmsnorm_layer(
            input = self.mlp_layer_io.layer_in.mk,
            weight = self.mk.attach_input(torch_tensor=w_rms_norm_torch, name="w_norm"+layer_id_str),
            output = self.mlp_layer_io.layernorm_out.mk,
            sync_mode=(0, 0, 0),
            layout=self.Qwen3MegaConfig.rmsnorm_layout,
        )
        self.mk.linear_layer(
            input  = self.mlp_layer_io.layernorm_out.mk,
            weight = self.mk.attach_input(torch_tensor=self.w_mlp_gateup_proj[layer_id], name="w_gatedup"+layer_id_str),
            output = self.mlp_layer_io.mlp_mid.mk,
            sync_mode=(0, 0, 0),
            layout = self.Qwen3MegaConfig.linear1_layout,
        )
        self.mk.silu_mul_layer(
            input  = self.mlp_layer_io.mlp_mid.mk,
            output = self.mlp_layer_io.silu_mul_out.mk,
            sync_mode=(2, 0, 0),
            layout = self.Qwen3MegaConfig.silu_mul_layout,
        )
        self.mk.linear_with_residual_layer(
            input    = self.mlp_layer_io.silu_mul_out.mk,
            weight   = self.mk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj"+layer_id_str),
            residual = self.mlp_layer_io.layer_in.mk,
            output   = self.mlp_layer_io.layer_out.mk,
            sync_mode=(0, 0, 0),
            layout   = self.Qwen3MegaConfig.linear2_layout,
        )
        
    def fill_meta(self):
        edge_torch = self.attn_layer_io.attn_in.edge.pt
        # edge_torch[0].fill_(step) # kv_seqlen
        edge_torch[1].fill_(self.params.num_kv_heads * self.params.head_dim) # kvcache onestep_size
        edge_torch[2].fill_(self.params.batch * self.params.max_kv_seqlen * self.params.num_kv_heads * self.params.head_dim) # kvcache onelayer_size
        
        k_torch_curstep = self.attn_layer_io.kv_curstep.k.pt
        v_torch_curstep = self.attn_layer_io.kv_curstep.v.pt
        meta = [edge_torch, self.public_pt.key_cache_5d, self.public_pt.value_cache_5d, k_torch_curstep, v_torch_curstep]
        return meta, self.attn_layer_io.layer_out.pt, self.mlp_layer_io.layer_out.pt
    
    def update_step(self, step, cos, sin):
        edge_torch = self.attn_layer_io.attn_in.edge.pt
        edge_torch[0].fill_(step)
        # print("step", step, self.public_pt.cos.size(), cos.size())
        self.public_pt.cos.copy_(cos)
        self.public_pt.sin.copy_(sin)
     
    def qwen3_create_decoder_layer(self, model, layer_num, params, io_pt, public_pt, is_long_kv, is_no_compile, output_dir):
        self.layer_num = layer_num
        self.qwen3_alloc_io_buffer(params, io_pt, public_pt)
        for layer_id in range(layer_num):
            # reuse_instance = False
            if (layer_id == 0):
                reuse_instance = False
            else:
                reuse_instance = True
            self.qwen3_create_attn_layer(model, layer_id, is_long_kv, reuse_instance)
            self.qwen3_create_mlp_layer(model, layer_id, reuse_instance)
        meta, mk_attn_out, mk_mlp_out = self.fill_meta()
        self.compile_load(meta_tensors=meta, enable_prefetch=False, is_no_compile=is_no_compile, output_dir=output_dir)  
        
    def __call__(self, cur_pos, position_embeddings, hidden_states):
        self.update_step(cur_pos - 1, cos=position_embeddings[0], sin=position_embeddings[1])
        self.attn_layer_io.layer_in.pt.copy_(hidden_states)
        for layer_id in range(self.layer_num):
            # print(layer_id)
            self.mk(self.params.batch, kernel_id=layer_id, layer_id=layer_id)  # attn的输入与mlp的输出是同一个tensor
        return self.mlp_layer_io.layer_out.pt
    
class MkLayersHybridLayout:
    def __init__(self, model_tag, instance_num, kernel_num, world_size, rank, max_batch_size, trace_name, profiling):
        self.instance_num = instance_num
        self.mk_layers = []
        for i in range(self.instance_num):
            layers = MkLayers(model_tag, instance_id=i, kernel_num=kernel_num, world_size=world_size, rank=rank, max_batch_size=max_batch_size, trace_name=trace_name, profiling=profiling)
            self.mk_layers.append(layers)
            
    def qwen3_create_decoder_layer(self, model_tag, model, layer_num, batch, is_no_compile, output_dir):
        params, self.io_pt, self.public_pt = MkLayers.qwen3_alloc_torch_buffer(model_tag, layer_num, batch, q_seqlen=1)   
        self.mk_layers[0].qwen3_create_decoder_layer(model, layer_num, params, self.io_pt, self.public_pt, False, is_no_compile, output_dir)
        self.mk_layers[1].qwen3_create_decoder_layer(model, layer_num, params, self.io_pt, self.public_pt, True, is_no_compile, output_dir+"longkv")
        
    def __call__(self, cur_pos, position_embeddings, hidden_states):
        # cur_pos == kv_seqlen; step = cur_pos - 1
        if cur_pos <= self.mk_layers[0].Qwen3MegaConfig.gqa_decode_layout_split_point:
            print("cur_pos1", cur_pos)
            return self.mk_layers[0](cur_pos, position_embeddings, hidden_states)
        else:
            print("cur_pos2", cur_pos)
            return self.mk_layers[1](cur_pos, position_embeddings, hidden_states)
        
        
# class MkModelInfer:
    