
from transformers import AutoTokenizer, AutoConfig
import torch
import torch.distributed as dist
import argparse
import os
import time
from einops import rearrange, einsum
# import megakernel as mi
import torch.nn.functional as F
# from tilelang.utils.profiler import do_bench

class TestUtil:
    @staticmethod
    def create_matrix_arange_row(shape, dtype=torch.bfloat16, device='cuda'):
        M, N = shape
        row_indices = torch.arange(M, dtype=dtype, device=device)
        matrix = row_indices.unsqueeze(1).expand(M, N).contiguous()  # contiguous is very important!
        return matrix
    
    @staticmethod
    def create_matrix_arange_col(shape, dtype=torch.bfloat16, device='cuda'):
        M, N = shape
        col_indices = torch.arange(N, dtype=dtype, device=device)
        matrix = col_indices.unsqueeze(0).expand(M, N).contiguous()
        return matrix

class TorchRef:
    @staticmethod
    def compile_capture(fn, is_compile):
        if is_compile:
            compiled_ref_fn = torch.compile(fn, backend="inductor")
        else:
            compiled_ref_fn = fn
            
        for _ in range(20):
            output = compiled_ref_fn()
        torch.cuda.synchronize()
        
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        # stream = torch.cuda.default_stream()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph):
                output = compiled_ref_fn()
        return graph, output
    
    # nn.Linear(in_features=K, out_features=N), 
    # F.linear 对应 转置B的gemm，即 A[m,k] * B[n,k] = C[m,n]
    @staticmethod
    def linear(x, w):
        return F.linear(x, w)
    
    @staticmethod
    def linear_o(x, w, out):
        with torch.no_grad():
            torch.matmul(x, w.t(), out=out)
        return out
    
    # hidden_states[seq_len, hidden], weight[hidden]
    # hidden_states.pow(2): 逐元素开方
    # mean(-1, keepdim=True): 在最后一维上取均值,即同一个seq下对所有hidden取均值，
    #                         并保持维度，维度变为 variance[seq_len, 1].
    # variance，逐元素开根号后广播，得到[seq_len, hidden], 同一seq下，所有hidden的值相同。
    # 一句话：一个样本特征 hidden，先开方，取均值，后开根号，乘以原来的值，再乘以权重。
    #        即每个样本都乘以自己的rms值，即均方根(一个标量值)：开方/求平均/开根号
    @staticmethod
    def rms_norm(hidden_states, weight, eps=1e-12):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return weight * hidden_states
    
    @staticmethod
    def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return torch.nn.functional.silu(x[..., :d]) * x[..., d:]
        
    @staticmethod
    def mlp(x, w_gatedup, w_down_proj):
        O2 = TorchRef.linear(x, w_gatedup)
        O3 = TorchRef.silu_and_mul(O2)
        D  = TorchRef.linear(O3, w_down_proj)
        return D
    
    @staticmethod
    def norm_mlp(x, w_rms_norm, w_gatedup, w_down_proj):
        O1 = TorchRef.rms_norm(x, w_rms_norm)
        O2 = TorchRef.linear(O1, w_gatedup)
        O3 = TorchRef.silu_and_mul(O2)
        D  = TorchRef.linear(O3, w_down_proj)
        return D
    
    @staticmethod
    def oproj_norm_mlp(x, x_residual, w_o_proj, w_rms_norm, w_gatedup, w_down_proj):
        O0 = TorchRef.linear(x, w_o_proj) + x_residual
        #
        O1 = TorchRef.rms_norm(O0, w_rms_norm)
        O2 = TorchRef.linear(O1, w_gatedup)
        O3 = TorchRef.silu_and_mul(O2)
        D  = TorchRef.linear(O3, w_down_proj) + O0
        return D
    
    @staticmethod
    def attention_sdpa(query, key, value, is_causal):
        if query.ndim == 3:
            q_for_sdpa = query.unsqueeze(2)         # [batch, heads, seqlen_q=1, dim]
        else:
            q_for_sdpa = query.permute(0, 2, 1, 3)
        k_for_sdpa = key.permute(0, 2, 1, 3)    # [batch, groups, seqlen_kv, dim]  groups即是num_kv_heads
        v_for_sdpa = value.permute(0, 2, 1, 3)  # [batch, groups, seqlen_kv, dim]

        attn_output_sdpa = F.scaled_dot_product_attention(
            q_for_sdpa, k_for_sdpa, v_for_sdpa, is_causal=is_causal, enable_gqa=True
        )
        attn_output = attn_output_sdpa.permute(0, 2, 1, 3)
        return attn_output

    @staticmethod
    def attention(query, key, value, mask, glse, Output_partial):
        #     """
        #     Inputs:
        #     - query (Tensor): [batch, heads, dim]
        #     - key (Tensor): [batch, seqlen_kv, groups, dim]
        #     - value (Tensor): [batch, seqlen_kv, groups, dim]
        #     - mask (Tensor): [batch, seqlen_kv, groups]
        #     Outputs:
        #     - output (Tensor): [batch, heads, dim]
        #     """
        dim = query.shape[-1]
        num_head_groups = query.shape[1] // key.shape[2]
        scale = dim**0.5
        key = rearrange(key, "b n h d -> b h n d")  # [batch_size, groups, seqlen_kv, dim]
        value = rearrange(value, "b n h d -> b h n d")  # [batch_size, groups, seqlen_kv, dim]

        query = rearrange(query, "b (h g) d -> b g h d", g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

        scores = einsum(query, key, "b g h d, b h s d -> b g h s")  # [batch_size, num_head_groups, groups, seqlen_kv]
        if mask is not None:
            mask = rearrange(mask, "b s h -> b h s")
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

        out = einsum(attention, value, "b g h s, b h s d -> b g h d")  # [batch_size, num_head_groups, groups, dim]
        out = rearrange(out, "b g h d -> b (h g) d")  # [batch_size, heads, dim]
        return out

    @staticmethod
    def attention_split(Q, K, V, mask, glse=None, Output_partial=None):
        dtype = torch.bfloat16
        
        def _flash_split_ref(Q, K, V, mask):
            num_split = 16
            batch = Q.size(0)
            nheads = Q.size(1)
            groups = K.size(2)
            dim = Q.size(-1)
            block_N = 32
            seqlen_kv = K.size(1)
            num_head_groups = nheads // groups
            
            scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
            acc_s = torch.empty((batch, num_head_groups, groups, block_N), device="cuda", dtype=torch.float)
            acc_s_cast = torch.empty((batch, num_head_groups, groups, block_N), device="cuda", dtype=dtype)
            acc_o = torch.empty((batch, num_head_groups, groups, dim), device="cuda", dtype=torch.float)
            scores_max = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
            scores_max_prev = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
            scores_scale = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
            scores_sum = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
            logsum = torch.empty((batch, num_head_groups, groups), device="cuda", dtype=torch.float)
            gacc_o = torch.empty((num_split, batch, nheads, dim), device="cuda", dtype=torch.float)
            glogsum = torch.empty((num_split, batch, nheads), device="cuda", dtype=torch.float)

            Q_ = Q * scale
            Q_ = rearrange(Q_, "b (h g) d -> b g h d", g=num_head_groups)

            for ks in range(num_split):
                acc_o.fill_(0)
                logsum.fill_(0)
                scores_max.fill_(float("-inf"))
                scores_max_prev.fill_(float("-inf"))
                for i in range(int((seqlen_kv // num_split) / block_N)):
                    acc_s.fill_(0)
                    acc_s = torch.einsum(
                        "bghd,bkhd->bghk",
                        Q_,
                        K[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :, :],
                    )  # [batch, nheads, block_N]
                    if mask is not None:
                        mask_local = mask[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :]
                        mask_local = rearrange(mask_local, "b s h -> b h s")
                        mask_local = mask_local.unsqueeze(1)
                        acc_s = acc_s.masked_fill(mask_local == 0, float("-inf"))
                    scores_max_prev = scores_max
                    scores_max = acc_s.max(dim=-1, keepdim=False).values  # [batch, nheads]
                    scores_scale = torch.exp2(scores_max_prev - scores_max)  # [batch, nheads]
                    acc_o *= scores_scale[:, :, :, None]
                    acc_s = torch.exp2(acc_s - scores_max[:, :, :, None])
                    acc_s_cast = acc_s.to(dtype)  # [batch, nheads, block_N]
                    acc_o += torch.einsum(
                        "bghk,bkhd->bghd",
                        acc_s_cast,
                        V[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :, :],
                    )
                    scores_sum = acc_s.sum(dim=-1, keepdim=False)
                    logsum = logsum * scores_scale + scores_sum
                acc_o_out = rearrange(acc_o, "b g h d->b (h g) d")
                logsum_out = rearrange(logsum, "b g h->b (h g)")
                acc_o_out /= logsum_out[:, :, None]
                logsum_out = torch.log2(logsum_out) + rearrange(scores_max, "b g h->b (h g)")
                gacc_o[ks, :, :, :] = acc_o_out
                glogsum[ks, :, :] = logsum_out

            return glogsum.to(dtype).permute(1, 2, 0), gacc_o.to(dtype).permute(1, 2, 0, 3)


        def _reduce_ref(Q, K, V, mask, glse, Output_partial):
            num_split = 16
            o = torch.empty_like(Output_partial[:, :, 0, :]).fill_(0)
            lse_logsum = torch.empty_like(glse[:, :, 0]).fill_(0)  # [batch, heads]
            lse_max = glse.max(dim=2, keepdim=False).values
            for ks in range(num_split):
                lse = glse[:, :, ks]
                lse_logsum += torch.exp2(lse - lse_max)
            lse_logsum = torch.log2(lse_logsum) + lse_max
            for ks in range(num_split):
                lse = glse[:, :, ks]
                scale = torch.exp2(lse - lse_logsum)  # [batch, heads]
                o += Output_partial[:, :, ks, :] * scale[:, :, None]
            return o.to(dtype)
        
        glse_, Output_partial_ = _flash_split_ref(Q, K, V, mask)
        return _reduce_ref(Q, K, V, mask, glse_, Output_partial_)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
        """Applies Rotary Position Embedding to the query and key tensors.
                           cos/sin表	                 Q使用的位置	      K使用的位置
        训练/prefill: 共享同一张表 [max_len, dim],	[0, 1, ..., L-1], 	 [0, 1, ..., L-1]
        推理：        共享同一张表 [max_len, dim],	  [L] (新token)	, 	  [L] (新token)	老的k不需要重复算rope
        """
        # Copied from transformers.models.llama.modeling_llama.rotate_half
        def rotate_half(x):
            # input:  [ a, b,  c, d,  e, f,  g, h]
            # output: [-b, a, -d, c, -f, e, -h, g]
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        
        # cos torch.Size([1, 1, 1, 128]) q torch.Size([1, 1, 16, 128]) k torch.Size([1, 1, 8, 128])
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def apply_rotary_pos_emb_triton(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
        from models.rope import apply_rotary_pos_emb_triton
        q_embed, k_embed = apply_rotary_pos_emb_triton(q, k, cos, sin, unsqueeze_dim=2)
        return q_embed, k_embed
  
class Qwen3Info:
    @staticmethod
    def get_basic_params(model_tag):
        if (model_tag == "qwen3-8b"):
            # https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
            hidden_size         = 4096        # K
            intermediate_size   = 12288 # torch.randn / ones / TestUtil.create_matrix_arange_col /
            num_attention_heads = 32
            num_key_value_heads = 8
            head_dim            = 128
            num_hidden_layers   = 36
        elif (model_tag == "qwen3_4b"):
            # https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json
            hidden_size         = 2560        
            intermediate_size   = 9728 
            num_attention_heads = 32
            num_key_value_heads = 8
            head_dim            = 128
            num_hidden_layers   = 36
        else:
            # "qwen3_06b"
            # https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json
            hidden_size         = 1024
            intermediate_size   = 3072
            num_attention_heads = 16
            num_key_value_heads = 8
            head_dim            = 128
            num_hidden_layers   = 28
        return [hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, num_hidden_layers]
        
    @staticmethod
    def load_model(rank, model_tag, page_size):
        from models.modeling_qwen3 import Qwen3ForCausalLM
        torch.cuda.set_device(rank)
        with torch.device("cuda"):
            if model_tag == "qwen3_06b":
                model_name = "/home/cjmcv/project/llm_models/Qwen/Qwen3-0.6B" # "/data/team/cjm/Qwen/Qwen3-0.6B"
            elif model_tag == "qwen3_4b":
                model_name = "/data/team/cjm/Qwen/Qwen3-4B"
            else:
                print("!! Do not support model: ", model_tag)
            model = Qwen3ForCausalLM.from_pretrained(model_name, world_size=1, max_num_pages=16, page_size=page_size).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(model_name) 
        return model, tokenizer
    
    @staticmethod
    def get_weight_qwen3_mlp(model, layer_id):
        layer = model.model.layers[layer_id]
        w_rms = layer.post_attention_layernorm.weight
        w_gate_proj = layer.mlp.gate_proj.weight
        w_up_proj = layer.mlp.up_proj.weight
        # w_gatedup = torch.cat((layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight), 0).contiguous()
        w_down_proj = layer.mlp.down_proj.weight
        return w_rms, w_gate_proj, w_up_proj, w_down_proj

    @staticmethod
    def get_weight_qwen3_attention(model, layer_id):
        # num_q_heads = model.config.num_attention_heads
        # num_kv_heads = model.config.num_key_value_heads
    
        layer = model.model.layers[layer_id]
        w_input_layernorm = layer.input_layernorm.weight
        w_q_norm = layer.self_attn.q_norm.weight
        w_k_norm = layer.self_attn.k_norm.weight
        w_q = layer.self_attn.q_proj.weight
        w_k = layer.self_attn.k_proj.weight
        w_v = layer.self_attn.v_proj.weight
        w_out_proj = layer.self_attn.o_proj.weight
        
        k_cache = model.model.kv_cache[0][layer_id]
        v_cache = model.model.kv_cache[1][layer_id]
        return w_input_layernorm, w_q_norm, w_k_norm, w_q, w_k, w_v, w_out_proj, k_cache, v_cache
class PerfReporter:

    def torch_profile(self, func):
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            func()
        print(prof.key_averages().table(sort_by="cuda_time_total"))
        prof.export_chrome_trace("trace.json") # chrome://tracing/        
    

    # Cosine Similarity 的变种
    # 余弦相似度公式: 点积 / (a平方和开根号 * b平方和开根号)
    # for (size_t i = 0; i < len; ++i) {
    #     dot_product += x[i] * y[i]; // 点积
    #     norm_sq_x += x[i] * x[i]; // x的L2范数平方, 0时直接返回0
    #     norm_sq_y += y[i] * y[i]; // y的L2范数平方, 0时直接返回0
    # }
    # double norm_x = std::sqrt(norm_sq_x);
    # double norm_y = std::sqrt(norm_sq_y);
    # double sim = dot_product / (norm_x * norm_y); // 需要避免norm_x/norm_y为0，然后需要裁剪到[-1,1]
    @staticmethod
    def assert_similar(x, y, eps=1e-2, name="tensor", assert_=False, print_=True):
        def print_red_warning(msg):
            print(f"\033[91m{msg}\033[0m")

        def calc_sim(x, y, name="tensor"):
            x, y = x.data.double(), y.data.double()
            denominator = (x * x + y * y).sum()
            if denominator == 0:
                print_red_warning(f"{name} all zero")
                return 1
            sim = 2 * (x * y).sum() / denominator
            return sim
        
        sim = calc_sim(x, y, name)
        diff = 1.0 - sim
        if not (0 <= diff <= eps):
            print_red_warning(f"{name} Error: {diff}")
            if assert_:
                raise AssertionError(f"{name} Error: {diff}")
            return False
        else:
            if print_:
                print(f"passed: {name} diff={diff}")
            return True
                
    def check_allclose_ret(self, target_run, torch_run, iter, print_mode):
        if (print_mode==2):
            torch.set_printoptions(threshold=float('inf'))
        torch.cuda.synchronize()
        
        # print("inner: ", torch_out, torch_out.data_ptr())
        for _ in range(iter):
            target_result = target_run()
            torch_result = torch_run()
            torch.cuda.synchronize()
            
            total_num = torch_result.numel()
            if (torch.allclose(target_result, torch_result, rtol=1e-2, atol=0)):
                print("allclose: True")
            else:
                if (print_mode >= 1):
                    print("target_out:", target_result.shape, "\n", target_result)
                    print("torch_out:", torch_result.shape, "\n", torch_result)
                    print("diff: ", target_result - torch_result)
                
                radio = abs((target_result - torch_result)/torch_result)
                
                threshold = [0.05, 0.10]
                count0 = (radio > threshold[0]).sum().item()
                count1 = (radio > threshold[1]).sum().item()
                print("radio > ", threshold[0], ": ", count0, "-", count0/total_num, " / ", threshold[1], ": ", count1, "-", count1/total_num)
            PerfReporter.assert_similar(target_result, torch_result, name="similar")    
             
    def time_cuda_event_record(self, name, func, test_iter):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
            
        starter.record()
        for _ in range(test_iter):
            func()
        ender.record()
        torch.cuda.synchronize()
        
        run_time = starter.elapsed_time(ender)
        print(name, "cuda_event time (ms): ", run_time / test_iter)
     
    def time_cpu_record(self, name, func, test_iter):
        torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        for _ in range(test_iter):
            func()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        run_time = (end_time - start_time) * 1000
        print(name, "run time (ms): ", run_time / test_iter)
        
    def generate_report(self, target_run, torch_run, warnup_iter, test_iter, allclose_iter, print_mode):  
        self.check_allclose_ret(target_run, torch_run, allclose_iter, print_mode)

        # self.time_cuda_event_record("torch_ref", torch_run, test_iter)   
        # self.time_cuda_event_record("mk", target_run, test_iter)
        # self.time_cpu_record("torch_ref", torch_run, test_iter)   
        # self.time_cpu_record("mk", target_run, test_iter)
        
        self.torch_profile(target_run)
        self.torch_profile(torch_run)
        
        latency = 0 # do_bench(lambda: target_run(), warmup=warnup_iter, rep=test_iter, backend="event") # "cupti": 不含kernel launch
        ref_latency = 0 # do_bench(lambda: torch_run(), warmup=warnup_iter, rep=test_iter, backend="event")
        print(f"Latency: {latency:.4f}ms vs {ref_latency:.4f}(torch) ms")
        
    @staticmethod
    def draw():
        # attention的计算量统计，围绕QK和PV两个gemm计算，softmax可直接忽略。
        # QK^t: [q_seqlen, head_dim] * [kv_seqlen, head_dim] = [q_seqlen, kv_seqlen], 计算量为 2 * q_seqlen * kv_seqlen * head_dim
        # PV:   [q_seqlen, kv_seqlen] * [kv_seqlen, head_dim] = [q_seqlen, head_dim], 计算量为 2 * q_seqlen * kv_seqlen * head_dim
        # softmax:  对每个Q头的score矩阵计算 全局 max + (Score-max) + exp(Score-max) + 全局 sum (exp 结果) + exp 结果 /sum
        #                                        0  +     1         +      1       +      0             +   1
        #          约等于 3 * q_seqlen * kv_seqlen
        # 总计算量为：heads * (QK^t + softmax + PV) 
        #          = heads * (4 * q_seqlen * kv_seqlen * head_dim + 3 * q_seqlen * kv_seqlen), GQA的kv_head少，但会重复被取来计算，计算量不变，只看q_heads.

        import matplotlib.pyplot as plt
        def cal_attn_tflops(q_seqlen, kv_seqlen, head_dim, heads):

            qk_flops = heads * 2 * q_seqlen * kv_seqlen * head_dim
            pv_flops = heads * 2 * q_seqlen * kv_seqlen * head_dim
            softmax_flops = heads * 3 * q_seqlen * kv_seqlen
    
            # attn_flops = heads * 4 * q_seqlen * kv_seqlen * head_dim
            attn_flops = qk_flops + softmax_flops + pv_flops
            attn_tflops = attn_flops / 1e12
            return attn_tflops
            
        q_seqlen = 1
        heads = 32
        head_dim = 128
        
        # L40: 0.6 / 4 / 8B 数据都接近
        kv_seqlen     = [  16,     32,       64,     128,      256,     512,     1024,    2048,    4096,    8192]
        torch_bf16_ms = [0.00645, 0.00674, 0.00854, 0.01239, 0.01276, 0.01484, 0.01897, 0.02788, 0.04652, 0.07821]
        xop_bf16_ms   = [0.00601, 0.00669, 0.00796, 0.00927, 0.01032, 0.01353, 0.01741, 0.02485, 0.04092, 0.07311]

        torch_bf16_tflops = []
        xop_bf16_tflops = []
        for i in range(len(kv_seqlen)):
            attn_tflops = cal_attn_tflops(q_seqlen, kv_seqlen[i], head_dim, heads)
            torch_bf16_tflops.append(attn_tflops / (torch_bf16_ms[i] / 1000))
            xop_bf16_tflops.append(attn_tflops / (xop_bf16_ms[i] / 1000))

        plot_x = range(len(kv_seqlen))
        plt.xticks(plot_x, kv_seqlen, rotation=45)
        plt.plot(plot_x, torch_bf16_tflops, label='fa2_bf16', marker='o', markersize=3)
        plt.plot(plot_x, xop_bf16_tflops, label='xop_bf16', marker='s', markersize=3)

        plt.title(f'perf-attn-q_seqlen{q_seqlen}-heads{heads}-head_dim{head_dim}')
        plt.xlabel('kv_seqlen')
        plt.ylabel('tflops')
        
        plt.legend()
        plt.grid(True)

        # plt.xticks(plot_x)
        plt.savefig(f'perf-attn-q_seqlen{q_seqlen}-heads{heads}-head_dim{head_dim}' + '.png', bbox_inches='tight')
        plt.show()
    
    # pushd build && make -j8 && popd
    
    # git clone --recursive https://www.github.com/megakernel-project/megakernel
    # pip install -e . -v
    # export DSL_HOME=$(pwd)
    # python demo_refac/single_silu_mul.py
    # --profiling https://ui.perfetto.dev/
    
    # nsys profile --trace=cuda,nvtx --output=my_nsys
    # ncu --set full --section "SpeedOfLight_RooflineChart" -k "persistent_kernel" -o my_profile python...
    # "kernel"
    # compute-sanitizer --tool memcheck python demo/single_mega.py --nc
    # compute-sanitizer --tool memcheck --shared-memory-check yes ./your_cuda_program
