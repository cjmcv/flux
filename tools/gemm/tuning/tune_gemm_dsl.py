import os
import torch
import argparse
import itertools
# 
os.environ["TL_DISABLE_WARP_SPECIALIZED"] = "1"

import tilelang
import tilelang.language as T

from xop.ops.dsl.pkt_util import TorchRef, Qwen3Info
from xop.ops.dsl.micro_base import HparamSelectMode
from xop.ops.dsl.micro_linear import MicroLinearStrategy, MicroLinear

import xop.util as xutil

# 全局 kernel 缓存: key -> (kernel, name, info, micro)
_kernel_cache = {}

def get_tuned_gemm(strategy, M, N, K, dtype=T.bfloat16, accum_dtype=T.float32):
    key = (strategy, M, N, K, dtype, accum_dtype)
    if key not in _kernel_cache:
        op = MicroLinear(strategy, M, N, K, dtype, accum_dtype)
        _kernel_cache[key], _, _ = op.get_kernel(HparamSelectMode.TUNED)
    return _kernel_cache[key]

def tune_gemm(M, N, K):
    op = MicroLinear(MicroLinearStrategy.GEMM, M, N, K, dtype=T.bfloat16, accum_dtype=T.float32)
    kernel, name, info = op.get_kernel(HparamSelectMode.TUNING)

    test_data_list = []
    for _ in range(5):
        test_data = op.gen_test_data(kernel.config)
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

# python tools/gemm/tuning/tune_gemm_dsl.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true", help="autogen")

    args = parser.parse_args()
    
    model_tag = "qwen3_06b" # "qwen3_4b"
    hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim, num_hidden_layers \
        = Qwen3Info.get_basic_params(model_tag)  

    # test_gemm(M=1, N=6144, K=1024, HparamSelectMode.TUNING) # HparamSelectMode.TUNING)
    
    # space_M = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]
    # space_NK = [(6144,1024)]

    # for M, NK in itertools.product(
    #     space_M, space_NK
    # ):
    #     test_gemm(M=M, N=NK[0], K=NK[1], mode=HparamSelectMode.TUNING) # HEURISTIC, TUNING, TUNED
        
    space_M = [1,2,4,8]
    space_NK = [(6144,1024)]

    for M, NK in itertools.product(
        space_M, space_NK
    ):
        tune_gemm(M=M, N=NK[0], K=NK[1])
    
    # for M, NK in itertools.product(
    #     space_M, space_NK
    # ):
    #     get_tuned_gemm(MicroLinearStrategy.GEMM, M, N=NK[0], K=NK[1], dtype=T.bfloat16, accum_dtype=T.float32)