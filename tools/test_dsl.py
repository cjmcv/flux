import os
import torch
import argparse
# 
os.environ["TL_DISABLE_WARP_SPECIALIZED"] = "1"

import tilelang
import tilelang.language as T

from xop.ops.dsl.pkt_util import TorchRef, PerfReporter, Qwen3Info
from xop.ops.dsl.micro_base import HparamSelectMode
from xop.ops.dsl.micro_linear import MicroLinearStrategy, MicroLinear
from xop.ops.dsl.micro_rmsnorm import MicroRmsNorm
from xop.ops.dsl.micro_silu_mul import MicroSiluMul
from xop.ops.dsl.micro_gqa_decode import MicroGqaDecode
from xop.ops.dsl.micro_rope import MicroRope
from xop.ops.dsl.micro_autogen import MicroAutoGen

import xop.util as xutil

def test_code_gen():
    @tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True})
    def kernel_load_B(M, N, BLOCK_M, BLOCK_N, threads, dtype="bfloat16"):
        @T.prim_func
        def rms_norm_load_B(B: T.Tensor((1, N), dtype)):
            with T.Kernel(1, threads=threads) as bx:
                B_shared = T.alloc_shared((1, N), dtype)
                T.copy(B[0:1, :], B_shared)
        return rms_norm_load_B

    @tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True})
    def rms_kernel_main(M, N, BLOCK_M, BLOCK_N, threads, eps=1e-12, dtype="bfloat16", accum_dtype="float32"):
        @T.prim_func
        def rms_norm(A: T.Tensor((M, N), dtype), B: T.Tensor((1, N), dtype), C: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(M, BLOCK_M), threads=threads) as bx:
                A_shared = T.alloc_shared((BLOCK_M, N), dtype)
                A_pow_local = T.alloc_fragment((BLOCK_M, N), accum_dtype)
                A_local = T.alloc_fragment((BLOCK_M, N), accum_dtype)
                A_powsum = T.alloc_fragment((BLOCK_M,), accum_dtype)
                B_shared = T.alloc_shared((1, N), dtype)
                B_local = T.alloc_fragment((1, N), accum_dtype)
                
                T.copy(A[bx * BLOCK_M : (bx + 1) * BLOCK_M, :], A_shared)
                T.copy(B[0:1, :], B_shared)
                
                # note: 在sm120, M=BLOCK_M=1, N=1024, threads=256时，下面的A拷贝需要加上“coalesced_width=1”转为标量处理才能正常生成kernel。
                # 因为向量化读取使用 float4 = 4 × 32位 = 8 x bfloat16，N = 1024 x bfloat16，所需线程数 = 1024 / 8 = 128 个线程。
                # coalesced_width=1时，转用uint32 = 2 x bfloat16，1024 / 2 = 512 可以满足。
                T.copy(A_shared, A_local) #, coalesced_width=1
                T.copy(B_shared, B_local)
                
                for i, j in T.Parallel(BLOCK_M, N):
                    A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
                T.reduce_sum(A_pow_local, A_powsum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    A_powsum[i] = T.rsqrt(A_powsum[i] / N + eps)
                for i, j in T.Parallel(BLOCK_M, N):
                    A_local[i, j] *= A_powsum[i] * B_local[0, j]
                T.copy(A_local, C[bx * BLOCK_M : (bx + 1) * BLOCK_M, :])

        return rms_norm
    
    @tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True})
    def linear_kernel_main(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, split_k, num_stages, thread_num, policy, enable_rasteration, dtype=T.bfloat16, accum_dtype=T.float32):
        
        @T.prim_func
        def linear(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
                B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
                C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
                C_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.annotate_layout({C_shared: tilelang.layout.make_swizzled_layout(C_shared)})
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                    T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
                    T.copy(B[bx * BLOCK_N, k * BLOCK_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True, policy=policy)
                    
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * BLOCK_M, bx * BLOCK_N])

        return linear
 
 
 
 
        #  def rms_norm(A: T.Tensor((M, N), dtype), B: T.Tensor((1, N), dtype), C: T.Tensor((M, N), dtype)):
        #     with T.Kernel(T.ceildiv(M, BLOCK_M), threads=threads) as bx:
        #         A_shared = T.alloc_shared((BLOCK_M, N), dtype)
        #         A_pow_local = T.alloc_fragment((BLOCK_M, N), accum_dtype)
        #         A_local = T.alloc_fragment((BLOCK_M, N), accum_dtype)
        #         A_powsum = T.alloc_fragment((BLOCK_M,), accum_dtype)
        #         B_shared = T.alloc_shared((1, N), dtype)
        #         B_local = T.alloc_fragment((1, N), accum_dtype)
                
        #         T.copy(A[0:1,:], A_local)
        #         T.copy(B[0:1,:], B_local)

        #         for i, j in T.Parallel(BLOCK_M, N):
        #             A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
        #         T.reduce_sum(A_pow_local, A_powsum, dim=1)
        #         for i in T.Parallel(BLOCK_M):
        #             A_powsum[i] = T.rsqrt(A_powsum[i] / N + eps)
        #         for i, j in T.Parallel(BLOCK_M, N):
        #             A_local[i, j] *= A_powsum[i] * B_local[0, j]
        #         T.copy(A_local, C[bx * BLOCK_M : (bx + 1) * BLOCK_M, :])
                
    @tilelang.jit(out_idx=[-1], pass_configs={"tl.disable_tma_lower": True})
    def fuesed_linear_kernel_main(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, split_k, num_stages, thread_num, policy, enable_rasteration, dtype=T.bfloat16, accum_dtype=T.float32):
        
        @T.prim_func
        def linear(
            A: T.Tensor((M, K), dtype),
            B1: T.Tensor((1, K), dtype),
            B2: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            RMS_BLOCK_M = 1
            with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
                B2_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
                C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)

                A1_shared = T.alloc_shared((RMS_BLOCK_M, K), dtype)
                A_pow_local = T.alloc_fragment((RMS_BLOCK_M, K), accum_dtype)
                A_local = T.alloc_fragment((RMS_BLOCK_M, K), accum_dtype)
                A_powsum = T.alloc_fragment((RMS_BLOCK_M,), accum_dtype)
                B1_local = T.alloc_fragment((1, K), accum_dtype)
                
                T.copy(A[0:1,:], A_local)
                T.copy(B1[0:1,:], B1_local)
                
                for i, j in T.Parallel(RMS_BLOCK_M, K):
                    A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
                T.reduce_sum(A_pow_local, A_powsum, dim=1)
                for i in T.Parallel(RMS_BLOCK_M):
                    A_powsum[i] = T.rsqrt(A_powsum[i] / K + 1e-12)
                for j in T.Parallel(K):
                    A_local[0, j] *= A_powsum[0] * B1_local[0, j]
                T.copy(A_local, A1_shared)
                
                for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                    T.copy(A1_shared[0, k * BLOCK_K], A_shared)
                    T.copy(B2[bx * BLOCK_N, k * BLOCK_K], B2_shared)
                    T.gemm(A_shared, B2_shared, C_local, transpose_B=True, policy=policy)
                    
                T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])

        return linear
       
    M = 1
    N = 6144
    K = 1024
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    rms_kernel = rms_kernel_main(M, K, 1, 1, 128)
    # print(rms_kernel.get_kernel_source())
    # c = rms_kernel(a, w1)

    linear_kernel = linear_kernel_main(M, N, K, 16, 64, 128, 1, 0, 128, 0, False)
    # print(linear_kernel.get_kernel_source())
    # c2 = linear_kernel(c, w2)
    # print(c2)
    
    fused_kernel = fuesed_linear_kernel_main(M, N, K, 16, 64, 128, 1, 0, 128, 0, False)
    # print(fused_kernel.get_kernel_source())
    # c3 = fused_kernel(a, w1, w2)
    # print(c3)
    
    def ref(iter):
        c = rms_kernel(a, w1)
        c2 = linear_kernel(c, w2)
        return c2
    def target(iter):
        return fused_kernel(a, w1, w2)
    
    xutil.profile(target, ref)
    
    # kernel2 = kernel_load_B(M, N, 1, 1, 128)
    # print(kernel2.get_kernel_source())
            
# def profile(target_func: callable, torch_ref_func: callable):
#     xutil.check_allclose_ret(target_func, torch_ref_func, iters=5, print_mode=0)
    
#     xutil.torch_profile(target_func)
#     xutil.torch_profile(torch_ref_func)
    
#     perf_result_xop = xutil.perf_gemm(warmup_iters=100, iters=500, name="target", fn=target_func)
#     perf_result_torch = xutil.perf_gemm(warmup_iters=100, iters=500, name="torch", fn=torch_ref_func)
#     print(f"Latency: {perf_result_xop.gemm_time_ms:0.5}ms vs {perf_result_torch.gemm_time_ms:0.5}(torch) ms")

def test_silu_mul():
    M, N = 32, 9728
    micro = MicroSiluMul(M,N, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info  = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

    PROBLEM_COUNT = 5
    test_data_list = []
    for _ in range(PROBLEM_COUNT):
        test_data = micro.gen_test_data(kernel.config)
        test_data_list.append(test_data)
    # test_data = micro.gen_test_data(kernel.config)
    def target_func(iter):
        test_data = test_data_list[iter % len(test_data_list)]
        return kernel(*test_data)
    def torch_ref(iter):
        test_data = test_data_list[iter % len(test_data_list)]
        return TorchRef.silu_and_mul(*test_data)
    xutil.profile(target_func, torch_ref)
    
def test_rms_norm():
    M = 1
    N = 1024
    micro = MicroRmsNorm(M,N, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, fn, info = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED
    print(kernel.get_kernel_source())
    # print(kernel.get_dispatch_source())
    
    test_data = micro.gen_test_data(kernel.config)
    def target_func(iter):
        return kernel(*test_data)
    def torch_ref(iter):
        return TorchRef.rms_norm(*test_data) 
    xutil.profile(target_func, torch_ref)
    
def test_merge_rms_norm():
    M = 16
    M2 = 8
    N = 2560
    micro = MicroRmsNorm(M,N, dtype=T.bfloat16, accum_dtype=T.float32, M2=M2)
    kernel, fn, info = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED
    
    test_data = micro.gen_test_data(kernel.config)
    def target_func(iter):
        return kernel(*test_data)
    
    a,b = test_data
    a1 = a[ : M, :]
    a2 = a[M : M+M2, :]
    b1 = b[0 : 1, :]
    b2 = b[1 : 2, :]
    def torch_ref(iter):
        c1 = TorchRef.rms_norm(a1, b1) 
        c2 = TorchRef.rms_norm(a2, b2) 
        return torch.cat([c1, c2], dim=0)
    xutil.profile(target_func, torch_ref)
    
def test_gemm():
    M = 1
    N = 6144
    K = 1024
    # N = 2560
    # K = 9728
    # config = [64,64,64,2,128,0,true]
    micro = MicroLinear(MicroLinearStrategy.GEMM, M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info = micro.get_kernel(HparamSelectMode.TUNING) # HEURISTIC, TUNING, TUNED

    test_data_list = []
    for _ in range(5):
        test_data = micro.gen_test_data(kernel.config)
        test_data_list.append(test_data)
    
    def target_func(iter):
        test_data = test_data_list[iter % len(test_data_list)]
        return kernel(*test_data)
    def torch_ref(iter):
        test_data = test_data_list[iter % len(test_data_list)]
        return TorchRef.linear(*test_data)
        
    # test_data = micro.gen_test_data(kernel.config)
    # def target_func(iter):
    #     return kernel(*test_data)
    # def torch_ref(iter):
    #     return TorchRef.linear(*test_data)
    
    xutil.profile(target_func, torch_ref)

# def test_silu_mul_gemm():
#     M = 32
#     # N = 19456
#     # K = 2560
#     N = 2560
#     K = 9728
#     # config = [64,64,64,2,128,0,true]
#     micro = MicroLinear(MicroLinearStrategy.SILU_MUL_GEMM, M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
#     kernel = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

#     a = torch.randn((M, K*2), dtype=torch.bfloat16, device="cuda")
#     b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

#     def target_func():
#         return kernel(a, b)
#     def torch_ref():
#         a2 = TorchRef.silu_and_mul(a)
#         return TorchRef.linear(a2, b)
#     xutil.profile(target_func, torch_ref)
    
def test_gemm_add():
    M = 1
    N = 1024
    K = 3072
    # N = 2560
    # K = 9728
    # config = [64,64,64,2,128,0,true]
    micro = MicroLinear(MicroLinearStrategy.GEMM_ADD, M,N,K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info  = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED

    test_data = micro.gen_test_data(kernel.config)
    a, b, r = test_data

    def target_func(iter):
        return kernel(a, b, r)
    def torch_ref(iter):
        return TorchRef.linear(a, b) + r
    
    xutil.profile(target_func, torch_ref)

def test_gqa_decode(num_heads, num_kv_heads, head_dim):
    batch = 1
    max_kv_seqlen = 8192
    target_kv_seqlen = 64
    valid_kv_seqlen = 64
    is_causal = False
    
    # config = [64,64,64,2,128,0,true]
    micro = MicroGqaDecode(batch, max_kv_seqlen, target_kv_seqlen, num_heads, num_kv_heads, head_dim, is_causal, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info  = micro.get_kernel(HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED
    
    test_data = micro.gen_test_data(kernel.config)
    q, k, v, edge, mask, glse, Output_partial = test_data
    
    edge[0].fill_(valid_kv_seqlen)
    def target_func(iter):
        return kernel(q, k, v, edge, mask, glse, Output_partial)
    
    k_slice = k[:, :valid_kv_seqlen, :, :]
    v_slice = v[:, :valid_kv_seqlen, :, :]
        
    # if q.ndim == 3:
    #     q_for_sdpa = q.unsqueeze(2)         # [batch, num_heads, seqlen_q=1, head_dim]
    # else:
    #     q_for_sdpa = q.permute(0, 2, 1, 3)
    # k_for_sdpa = k_slice.permute(0, 2, 1, 3)    # [batch, num_kv_heads, seqlen_kv, head_dim]  groups即是num_kv_heads
    # v_for_sdpa = v_slice.permute(0, 2, 1, 3)  # [batch, num_kv_heads, seqlen_kv, head_dim]
    
    # def torch_ref():
    #     return torch.nn.functional.scaled_dot_product_attention(
    #         q_for_sdpa, k_for_sdpa, v_for_sdpa, is_causal=is_causal, enable_gqa=True
    #     )
        
    def torch_ref(iter):
        return TorchRef.attention_sdpa(q, k_slice, v_slice, is_causal)
        # return TorchRef.attention(q, k, v, mask, glse, Output_partial)
        # return TorchRef.attention_split(q, k, v, mask, glse, Output_partial)
    xutil.profile(target_func, torch_ref)

def test_rope(num_heads, num_kv_heads, head_dim):
    batch = 1
    seqlen = 1
    
    micro = MicroRope(batch, seqlen, num_heads, num_kv_heads, head_dim, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info  = micro.get_kernel(HparamSelectMode.TUNING) # HEURISTIC, TUNING, TUNED
    # kernel.export_sources(kernel_path="demo/gen/single_micro.cu")
    
    test_data = micro.gen_test_data(kernel.config)
    q, k, cos, sin = test_data
    
    def triton_ref(iter):
        q_emb, k_emb = TorchRef.apply_rotary_pos_emb_triton(q, k, cos, sin, unsqueeze_dim=2)
        return torch.cat((q_emb, k_emb), dim=-2)

    def torch_ref(iter):
        q_emb, k_emb = TorchRef.apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2)
        return torch.cat((q_emb, k_emb), dim=-2)

    def target_func(iter):
        q_emb, k_emb = kernel(q, k, cos, sin)
        return torch.cat((q_emb, k_emb), dim=-2)
      
    xutil.profile(target_func, triton_ref)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true", help="autogen")

    args = parser.parse_args()
    
    model_tag = "qwen3_06b" # "qwen3_4b"
    hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, num_hidden_layers \
        = Qwen3Info.get_basic_params(model_tag)  
    
    # test_silu_mul()
    # test_rms_norm()
    # test_merge_rms_norm()
    # test_gemm()
    ## test_silu_mul_gemm() # 逻辑有误，silu_mul被重复计算
    # test_gemm_add()
     
    # test_gqa_decode(num_heads, num_kv_heads, head_dim)
    # test_rope(num_heads, num_kv_heads, head_dim)

    # test_code_gen()
    
    if (args.gen is True): 
        gen = MicroAutoGen(model_tag, batch_size=1, hidden_size=hidden_size, intermediate_size=intermediate_size, 
                        max_kv_seqlen=8192, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
        gen.gen_qwen3_ops(layer_id=99, mode=HparamSelectMode.HEURISTIC) # HEURISTIC, TUNING, TUNED
    # print(">> Finish gen_qwen3_ops.")
    # print("Test single_micro completed.")
    
    # PerfReporter.draw()