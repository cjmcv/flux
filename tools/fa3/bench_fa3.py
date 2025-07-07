import torch
from benchmark import benchmark_forward
from xop.ops.flash_attn import flash_attn_func as flash_attn_func_v3
from torch.nn.functional import scaled_dot_product_attention
import math
import argparse

parser = argparse.ArgumentParser(description='Benchmark FlashAttention3')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim

print(f"FlashAttention3 Benchmark")
print(f"batch: {batch}, head: {head}, headdim: {headdim}")

print("fp16:")
for is_causal in [False, True]:
    print(f"is_causal: {is_causal}")
    for seq_len in {1024, 2048, 4096, 8192}: # , 16384, 32768
        # torch.float16
        flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
        q = torch.randn(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda")
        scale = 1 / headdim**0.5
        
        for i in range(5): flash_attn_func_v3(q, k, v, softmax_scale=scale, causal=is_causal)
        torch.cuda.synchronize()
        _, time = benchmark_forward(flash_attn_func_v3, q, k, v, softmax_scale=scale, causal=is_causal, repeats=100, verbose=False, desc='Triton')
        print(f'{seq_len} TFLOPS:{flops/time.mean*1e-12}')

        o1,lse = flash_attn_func_v3(q, k, v, softmax_scale=scale, causal=is_causal)
        # [batch_size, seqlen, nheads, headdim] => [batch_size, nheads, seqlen, headdim]
        q2 = q.permute(0, 2, 1, 3)
        k2 = k.permute(0, 2, 1, 3)
        v2 = v.permute(0, 2, 1, 3)
        o2 = scaled_dot_product_attention(q2, k2, v2, is_causal=is_causal, scale=scale)
        o2 = o2.permute(0, 2, 1, 3)
        # print("o1:", o1)
        # print("o2:", o2)
        # print("o3:", o1-o2)
        print(torch.allclose(o1, o2, atol=1e-2)) # fp16acc:atol=1e-2  normal: atol=1e-3

# print("fp8:")
# for is_causal in [False, True]:
#     print(f"is_causal: {is_causal}")
#     for seq_len in {1024, 2048, 4096, 8192}: # , 16384, 32768
#         flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
#         q = torch.randn(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
#         k = torch.randn(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
#         v = torch.randn(batch, seq_len, head, headdim, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
#         scale = 1 / headdim**0.5

#         q_descale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
#         k_descale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
#         v_descale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
#         for i in range(5): flash_attn_func_v3(q, k, v, scale, causal=is_causal, q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
#         torch.cuda.synchronize()
#         _, time = benchmark_forward(flash_attn_func_v3, q, k, v, scale, causal=is_causal, q_descale=q_descale, k_descale=k_descale, v_descale=v_descale, repeats=100, verbose=False, desc='Triton')
#         print(f'{seq_len} flops:{flops/time.mean*1e-12}')