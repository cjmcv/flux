from enum import IntEnum, auto
from typing import Tuple, List
import itertools
import dataclasses
import numpy as np
import time

import torch
import math

# Sync from enum class UnifiedMetaEnum
class Meta(IntEnum):
    GemmNormal = 0           # meta type
    GemmNormalSimt = auto()
    GemmBlockScaleFp8 = auto()
    GemmGroupedBlockScaleFp8 = auto()
    GemmLt = auto()
    GemmAllreduce = 20
    Void = 50                # data type
    FP16 = auto()   
    BF16 = auto()
    FP32 = auto()
    E4M3 = auto()
    E5M2 = auto()
    S8 = auto()
    S32 = auto()
    Sm80 = 60                # arch
    Sm89 = auto()
    Sm90 = auto()
    RRR = 70                 # layout
    RCR = auto()
    RCC = auto()
    def __str__(self):
        return {
            Meta.GemmNormal: "GemmNormal",
            Meta.GemmNormalSimt: "GemmNormalSimt",
            Meta.GemmBlockScaleFp8: "GemmBlockScaleFp8",
            Meta.GemmGroupedBlockScaleFp8: "GemmGroupedBlockScaleFp8",
            Meta.GemmLt: "GemmLt",
            Meta.GemmAllreduce: "GemmAllreduce",
            Meta.Void: "Void",
            Meta.FP16: "FP16",
            Meta.BF16: "BF16",
            Meta.FP32: "FP32",
            Meta.E4M3: "E4M3",
            Meta.E5M2: "E5M2",
            Meta.S8: "S8",
            Meta.S32: "S32",
            Meta.Sm80: "Sm80",
            Meta.Sm89: "Sm89",
            Meta.Sm90: "Sm90",
            Meta.RRR: "RRR",
            Meta.RCR: "RCR",
            Meta.RCC: "RCC",
        }.get(self, "Unknown")

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

def write_tuning_result(fp, add_func_name, fn, shape, tuning, tuning_data, pref_iters):
    m = shape[0]
    n = shape[1]
    k = shape[2]
    g = shape[3]
    
    tuning_data.sort()
    tuning[0], tuning[1], tuning[2] = 1, tuning_data[0][1], tuning_data[0][2]
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

    for sid in range(min(3, len(tuning_data))):
        prefix = ''
        if sid != 0:
            prefix = '// '
        if tuning_data[sid][2] == Meta.GemmLt: # cublasLt schema
            cublas_algo_str = ''
            for i in range(cublasLt_start_idx, cublasLt_end_idx):
                cublas_algo_str += str(tuning[i].item()) + ','
            cublas_algo_str += str(tuning[cublasLt_end_idx].item())
            selected_res = "{0}, {1}, {2}".format(str(tuning_data[sid][1]), "(int16_t)ME::"+str(tuning_data[sid][2]), cublas_algo_str)
        else: # cutlass
            selected_res = "{0}, {1}".format(str(tuning_data[sid][1]), "(int16_t)ME::"+str(tuning_data[sid][2]))
            
        message = "  {0}tins.{1}({{{2},{3},{4},{5},{6}}}, /*config*/{{{7}}}); // {8}ms\n".format(prefix, add_func_name, m, n, k, g, meta_str, selected_res, str(round(tuning_data[sid][0] * 1000 / pref_iters, 3)))
        fp.write(message)
        fp.flush()
        print(message)
    
    # double check
    fastest_id = tuning_data[0][1]
    fastest_schema = tuning_data[0][2]
    if (fastest_id != tuning[1].item() or fastest_schema != tuning[2].item()):
        print("fastest_config is not matched: {0},{1} vs {2},{3}".format(str(fastest_id), str(fastest_schema), str(tuning[1].item()), str(tuning[2].item())))
        raise RuntimeError
 
 
def profiling_core_cudagraph(fn: callable, tuning, ret, add_func_name, shape, schema, warmup_iters, pref_iters, fp):
    tuning_data = []
    for sub_schema in schema.sub_schema:
        for id in range(500):
            # warmup and check if exist.
            tuning[0], tuning[1], tuning[2] = 1, id, sub_schema
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
                tuning[0], tuning[1], tuning[2] = 1, id, sub_schema
                fn()
            torch.cuda.synchronize()
            elapsed_time = time.time() - start
            tuning_data.append((elapsed_time, id, sub_schema))

    write_tuning_result(fp, add_func_name, fn, shape, tuning, tuning_data, pref_iters)
    
def profiling_core(fn: callable, add_func_name, shape, schema, warmup_iters, pref_iters, fp):
    tuning = torch.zeros(100, dtype=torch.int16, device='cpu')

    tuning_data = []
    for sub_schema in schema.sub_schema:
        for id in range(500):
            # warmup and check if exist.
            tuning[0], tuning[1], tuning[2] = 1, id, sub_schema
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
                tuning[0], tuning[1], tuning[2] = 1, id, sub_schema
                fn(tuning)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start
            tuning_data.append((elapsed_time, id, sub_schema))

    write_tuning_result(fp, add_func_name, fn, shape, tuning, tuning_data, pref_iters)
       
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