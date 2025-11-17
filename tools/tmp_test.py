#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import time

def test_individual_allocation():
    weights = []
    for i in range(100):
        weights.append(torch.randn(512, 512, device='cuda'))
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for w in weights:
        x = torch.randn(512, 512, device='cuda')
        _ = torch.matmul(x, w)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

def test_contiguous_allocation():
    total_size = 100 * 512 * 512
    storage = torch.empty(total_size, device='cuda')
    weights = []
    
    for i in range(100):
        start_idx = i * 512 * 512
        end_idx = (i + 1) * 512 * 512
        weights.append(storage[start_idx:end_idx].view(512, 512))
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for w in weights:
        x = torch.randn(512, 512, device='cuda')
        _ = torch.matmul(x, w)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

for i in range(100):
    test_individual_allocation()
    
time_contiguous = test_contiguous_allocation()    
time_individual = test_individual_allocation()


print(f"time_individual: {time_individual:.2f}ms")
print(f"time_contiguous: {time_contiguous:.2f}ms")
print(f"speedup: {time_individual/time_contiguous:.2f}x")