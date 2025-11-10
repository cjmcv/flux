from enum import IntEnum, auto
from typing import Tuple, List
import itertools
import dataclasses
import numpy as np
import time

import torch
import math

import xop.util as xutil
from xop.common import Meta

def ceil_div(a, b):
    return math.ceil(a / b)

def init_test_env(seed: int = 3):
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_printoptions(precision=8)
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    np.random.seed(3)
@dataclasses.dataclass
class TuningConfig:
    G: int
    M: int
    N: int
    K: int
    transpose_weight: bool
    dtypeA: str
    dtypeB: str
    dtypeC: str
    has_bias: bool
    
def gen_tuning_space(space_dtype, space_G, space_M, space_NK, space_has_bias):
    space: List[TuningConfig] = []
    space_transpose_weight = [False] # , True
    for G, NK, M, transpose_weight, dtype, has_bias in itertools.product(
        space_G, space_NK, space_M, space_transpose_weight, space_dtype, space_has_bias
    ):
        config = TuningConfig(
            G=G, M=M, N=NK[0], K=NK[1], transpose_weight=transpose_weight, 
            dtypeA=dtype[0], dtypeB=dtype[1], dtypeC=dtype[2], has_bias=has_bias
        )
        space.append(config)
    return space

def get_tuned_base_info(tuned_data):
    return tuned_data[1], tuned_data[2], tuned_data[3] # 0: Measured time consumption of tuning

# mode: -1: ignore (Used during multi-card tuning, e.g., only rank 0 needs to save the results in the communication kernel.) 
#        1: normal tuning.
#        2: runing hparam.
def set_tuning_target(tuning, mode, id, schema, arch):
    tuning[0], tuning[1], tuning[2], tuning[3] = mode, id, schema, arch
    
def write_tuning_result(fp, add_func_name, fn, shape, tuning, tuned_data, pref_iters, mode=1):
    m = shape[0]
    n = shape[1]
    k = shape[2]
    g = shape[3]
    
    tuned_data.sort()
    id, schema, arch = get_tuned_base_info(tuned_data[0]) # Get the fastest one.
    set_tuning_target(tuning, mode, id, schema, arch)
    fn(tuning)  # Run it once to retrieve the metadata.

    # tuning: 0:meta_end_idx, 1:id, 2:schema, 3:~meta
    #         [20-28): cublasLt-algo
    meta_start_idx = 3
    meta_end_idx = tuning[0] # len
    cublasLt_start_idx = 20 # 8*uint64_t = 32*int16_t
    cublasLt_end_idx = 51
   
    meta_str = ''              #
    for i in range(meta_start_idx, meta_end_idx):
        meta_str += "(int16_t)ME::" + str(Meta(tuning[i].item())) + ','
    meta_str += "(int16_t)ME::" + str(Meta(tuning[meta_end_idx].item()))

    for sid in range(min(3, len(tuned_data))):
        prefix = ''
        if sid != 0:
            prefix = '// '
        if tuned_data[sid][2] == Meta.GemmLt: # cublasLt schema
            cublas_algo_str = ''
            for i in range(cublasLt_start_idx, cublasLt_end_idx):
                cublas_algo_str += str(tuning[i].item()) + ','
            cublas_algo_str += str(tuning[cublasLt_end_idx].item())
            selected_res = "{0}, {1}, {2}".format(str(tuned_data[sid][1]), "(int16_t)ME::"+str(tuned_data[sid][2]), cublas_algo_str)
        else: # cutlass
            selected_res = "{0}, {1}".format(str(tuned_data[sid][1]), "(int16_t)ME::"+str(tuned_data[sid][2]))
            
        gemm_time_ms = round(tuned_data[sid][0] * 1000 / pref_iters, 3)
        tflops = round(xutil.calculate_tflops(m,n,k, gemm_time_ms), 3)
        message = "  {0}tins.{1}({{{2},{3},{4},{5},{6}}}, /*config*/{{{7}}}); // {8} ms vs {9} tflops\n".format(prefix, add_func_name, m, n, k, g, meta_str, selected_res, str(gemm_time_ms), str(tflops))
        fp.write(message)
        fp.flush()
        print(message)
    
    # double check
    fastest_id = tuned_data[0][1]
    fastest_schema = tuned_data[0][2]
    if (fastest_id != tuning[1].item() or fastest_schema != tuning[2].item()):
        print("fastest_config is not matched: {0},{1} vs {2},{3}".format(str(fastest_id), str(fastest_schema), str(tuning[1].item()), str(tuning[2].item())))
        raise RuntimeError
 
 
def profiling_core_cudagraph(fn: callable, tuning, ret, add_func_name, shape, schema, warmup_iters, pref_iters, fp):
    tuned_data = []
    for sub_schema in schema.sub_schema:
        for id in range(500):
            # warmup and check if exist.
            set_tuning_target(tuning, 1, id, sub_schema, schema.arch)
            fn()
            code = 0
            if (isinstance(ret, torch.Tensor) and ret is None):
                code = -1
            else:
                code = ret
            print(1, id, sub_schema, "-> code: ", code)
            if (code == -1):
                break

            for i in range(warmup_iters + pref_iters):
                if (i == warmup_iters):
                    torch.cuda.synchronize()
                    start = time.time()
                set_tuning_target(tuning, 1, id, sub_schema, schema.arch)
                fn()
            torch.cuda.synchronize()
            elapsed_time = time.time() - start
            tuned_data.append((elapsed_time, id, sub_schema, schema.arch))

    write_tuning_result(fp, add_func_name, fn, shape, tuning, tuned_data, pref_iters)
    
def profiling_core(fn: callable, add_func_name, shape, schema, warmup_iters, pref_iters, fp):
    tuning = torch.zeros(100, dtype=torch.int16, device='cpu')

    tuned_data = []
    for sub_schema in schema.sub_schema:
        for id in range(500):
            # warmup and check if exist.
            set_tuning_target(tuning, 1, id, sub_schema, schema.arch)
            ret = fn(tuning)
            code = 0
            if (isinstance(ret, torch.Tensor) and ret is None):
                code = -1
            else:
                code = ret
            print(1, id, sub_schema, "-> code: ", code)
            if (code == -1):
                break

            for i in range(warmup_iters + pref_iters):
                if (i == warmup_iters):
                    torch.cuda.synchronize()
                    start = time.time()
                set_tuning_target(tuning, 1, id, sub_schema, schema.arch)
                fn(tuning)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start
            tuned_data.append((elapsed_time, id, sub_schema, schema.arch))

    write_tuning_result(fp, add_func_name, fn, shape, tuning, tuned_data, pref_iters)
       
def gen_tuning_file_head(fp, tag):
    fp.write('#include "xop/xop.h"\n')
    fp.write('#include "xop/ops_impl/global_resource.h"\n\n')
    fp.write('namespace xop {\n')
    fp.write('using namespace cutlass;\n')
    fp.write('using ME = UnifiedMetaEnum;\n\n')
    fp.write('static int tuned_config_{0} = []() {{\n'.format(tag.lower()))
    
    fp.write('  TunedConfigRegister& tins = TunedConfigRegister::instance();\n')
 
def gen_tuning_file_tail(fp):
    fp.write('  return 0;\n}();\n}')
    fp.write('// clang-format on')    