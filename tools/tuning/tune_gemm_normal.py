
import argparse
import dataclasses
import itertools
import os
from functools import partial
from typing import List
from enum import IntEnum, auto
import numpy as np
import time
import torch

import ctlop
from tune_common import Meta

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=8)
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
np.random.seed(3)
print = partial(print, flush=True)

@dataclasses.dataclass
class TuningConfig:
    M: int
    N: int
    K: int
    transpose_weight: bool
    dtype: str
    has_bias: bool

def gen_tuning_space():
    space: List[TuningConfig] = []
    space_M = list(range(1, 31)) #  [1024, 2048, 4096, 8192] # , 16384
    space_NK = [(512,256)] # (3584,5120), (5120,2560), (5120,13824), (27648,5120), 49152
    space_transpose_weight = [False] # , True
    space_dtype = [torch.float16] # , torch.bfloat16
    space_has_bias = [False]
    for NK, M, transpose_weight, dtype, has_bias in itertools.product(
        space_NK, space_M, space_transpose_weight, space_dtype, space_has_bias
    ):
        config = TuningConfig(
            M=M, N=NK[0], K=NK[1], transpose_weight=transpose_weight, dtype=dtype, has_bias=has_bias
        )
        space.append(config)
    return space


def get_torch_output(input: torch.Tensor, weight: torch.Tensor):
    output = torch.matmul(input, weight.t())
    return output.cpu()

def run_ctlop_profiling(input: torch.Tensor, weight: torch.Tensor, config: TuningConfig, fp):
    m = input.size(0)
    k = input.size(1)
    if config.transpose_weight:
        weight = weight.t().contiguous()
        n = weight.size(1)
    else:
        n = weight.size(0)

    bias = None
    if config.has_bias:
        bias = torch.zeros([m, n], dtype=input.dtype, device=input.device, requires_grad=False)

    tuning = torch.zeros(100, dtype=torch.int8, device='cpu')
    output = torch.empty([m, n], dtype=input.dtype, device=input.device, requires_grad=False)
    op = ctlop.GemmNormal(input_dtype=input.dtype, output_dtype=input.dtype, transpose_weight=config.transpose_weight)

    fastest_time = 99999
    fastest_config = [0,Meta.GemmNormal]
    for schema in [Meta.GemmNormal, Meta.GemmNormalSimt]:
        for id in range(100):
            # warmup and check if exist.
            tuning[0], tuning[1], tuning[2] = 1, id, schema
            output = op.forward(input, weight, bias=None, output_buf=None, 
                                input_scale=None, weight_scale=None, output_scale=None, 
                                tuning = tuning, fast_accum=False)
            if (output is None):
                break

            warmup_iters = 100
            iters = 200
            for i in range(warmup_iters + iters):
                if (i == warmup_iters):
                    torch.cuda.synchronize()
                    start = time.time()
                tuning[0], tuning[1], tuning[2] = 1, id, schema
                output = op.forward(input, weight, bias=None, output_buf=None, 
                                    input_scale=None, weight_scale=None, output_scale=None, 
                                    tuning = tuning, fast_accum=False)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start
            if (fastest_time > elapsed_time):
                fastest_time = elapsed_time
                fastest_config = [id, schema]

    tuning[0], tuning[1], tuning[2] = 1, fastest_config[0], fastest_config[1]
    # fp.write("fastest: {0}ms, {1}".format(str(fastest_time * 1000 / iters), str(fastest_id)))
    output = op.forward(input, weight, bias=None, output_buf=None, 
                        input_scale=None, weight_scale=None, output_scale=None, 
                        tuning = tuning, fast_accum=False)

    # tuning: 0:meta_len, 1:id, 2:schema, 3:~meta
    meta_len = tuning[0]       # len
    meta_str = ''              #              
    for i in range(3, meta_len):
        meta_str += "(int8_t)ME::" + str(Meta(tuning[i].item())) + ','
    meta_str += "(int8_t)ME::" + str(Meta(tuning[meta_len].item()))
    message = "  tins.add({{{0}, {1}, {2}, {3}}}, /*config*/{{{4}, {5}}}); // {6}ms\n".format(m, n, k, meta_str, str(fastest_config[0]), "(int8_t)ME::"+str(fastest_config[1]), str(round(fastest_time * 1000 / iters, 3)))
    # double check
    if (fastest_config[0] != tuning[1].item() or fastest_config[1] != tuning[2].item()):
        print("fastest_config is not matched: {0},{1} vs {2},{3}".format(str(fastest_config[0]), str(fastest_config[1]), str(tuning[1].item()), str(tuning[2].item())))
        raise RuntimeError
    
    fp.write(message)
    fp.flush()
    print(message)
    return output.cpu()

def tune_one_config(config: TuningConfig, fp):
    input = torch.rand((config.M, config.K), dtype=config.dtype).cuda()
    weight = torch.rand((config.N, config.K), dtype=config.dtype).cuda()
    # start_time = time.time()
    torch_output = get_torch_output(input, weight)
    # print(f"torch compute time: {(time.time() - start_time) * 1000} ms")
    ctlop_output = run_ctlop_profiling(input, weight, config, fp)

    if config.dtype == torch.bfloat16:
        atol, rtol = 0.02, 0.02
    else:
        atol, rtol = 0.01, 0.01
    ctlop.torch_allclose(ctlop_output, torch_output, atol=atol, rtol=rtol)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="./tools/", type=str, help="Directory to store generated files"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output_dir and not os.path.isdir(args.output_dir):
        raise Exception(f"{args.output_dir} not exist")

    tag = "GemmNormal"
    fp = {}
    fp[tag] = open("tuned_config_{0}.cu".format(tag.lower()), "w")

    fp[tag].write('#include "ctlop/ops_impl/gemm_normal/gemm_v2_impl.h"\n')
    fp[tag].write('#include "ctlop/ops_impl/gemm_normal/gemm_v2_simt_impl.h"\n\n')
    fp[tag].write('namespace ctlop {\n')
    fp[tag].write('using namespace cutlass;\n')
    fp[tag].write('using ME = UnifiedMetaEnum;\n\n')
    fp[tag].write('static int tuned_config_{0} = []() {{\n'.format(tag.lower()))
    
    fp[tag].write('  TunedConfigRegister& tins = TunedConfigRegister::instance();\n')
    
    arch: int = ctlop.get_arch()
    name: str = f"config_single_gemm_sm{arch}"
    config_space = gen_tuning_space()
    for i, config in enumerate(config_space):
        print(f"==== #{i + 1}/{len(config_space)} Tuning for {config}")
        tune_one_config(config=config, fp=fp[tag])

    fp[tag].write('  return 0;\n}();\n}')
    fp[tag].write('// clang-format on')