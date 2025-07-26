
import sys
import torch
import time
import math
from typing import Tuple

def pad_row_to_alignment(x, align):
    row = x.size(0)
    pad_rows = (align - (row % align)) % align
    if pad_rows == 0:
        return x
  
    padding = torch.zeros(pad_rows, x.size(1), dtype=x.dtype, device="cuda")
    return torch.cat([x, padding], dim=0)

# scale_a[m, k//128]
# x and x_scale are internally read in multiples of 4, and use predicate tensors handle boundary cases.
# But x_scale is transposed before input, if the m-dimension is not padded to a multiple of 4, the transposed data will be mismatched.
# org: [[1,1]] =>transpose [[1],[1]], in memory, they are the same, like [1,1]
# org: [[1,1]] =>pad [[1,1], [0,0], [0,0], [0,0]] =>transpose [[1,0,0,0], [1,0,0,0]] => it looks like [1,0,0,0,1,0,0,0]
def gemm_v2_blockscale_fp8_scale_preprocess(scale_a):
    torch.set_printoptions(precision=8)
    chunk_size = 16384
    if (scale_a.size(0) <= chunk_size):
        return pad_row_to_alignment(scale_a, 4).t().contiguous()
        
    chunks = torch.split(scale_a, chunk_size, dim=0)
    if len(chunks) >= 2 and chunks[-1].size(0) < chunk_size:
        last_two = torch.cat([chunks[-2], chunks[-1]], dim=0)
        last_two = pad_row_to_alignment(last_two, 4).contiguous()
        chunks = chunks[:-2] + (last_two,)

    # print("origin: ", scale_a.shape)
    # for i, chunk in enumerate(chunks):
    #     print(f"chunk {i+1} shape: {chunk.shape}")
    # chunks_t = [chunk.t() for chunk in chunks]
    # for i, chunk in enumerate(chunks_t):
    #     print(f"chunk_t {i+1} shape: {chunk.shape}")

    chunks_t_flat = [chunk.t().reshape(1, -1) for chunk in chunks]
    # for i, chunk in enumerate(chunks_t_flat):
    #     print(f"chunks_t_flat {i+1} shape: {chunk.shape}")
    
    flattened_chunks = []
    for i, chunk in enumerate(chunks_t_flat):
        flattened_chunks.append(chunk)

    res = torch.cat(flattened_chunks, dim=1).contiguous()
    # print("res: ", res.shape)
    return res