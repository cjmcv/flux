import torch
a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')

from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    c = torch.matmul(a, b)

print(prof.key_averages().table(sort_by="cuda_time_total"))