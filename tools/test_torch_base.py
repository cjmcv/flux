import torch
from torch.profiler import profile, ProfilerActivity

a = torch.randn(1024, 4096, device='cuda')
b = torch.randn(4096, 4096, device='cuda')

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    c = torch.matmul(a, b)

print(prof.key_averages().table(sort_by="cuda_time_total"))