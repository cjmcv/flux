
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

import flux

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
    space_M = list(range(1, 101)) #  [1024, 2048, 4096, 8192] # , 16384
    space_NK = [(27648,5120)] # (3584,5120), (5120,2560), (5120,13824), (27648,5120), 49152
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

# Sync from UnifiedMetaEnum
class Meta(IntEnum):
    Normal = 0      # meta type
    Void = 10       # data type
    FP16 = auto()   
    BF16 = auto()
    FP32 = auto()
    E4M3 = auto()
    E5M2 = auto()
    S8 = auto()
    S32 = auto()
    Sm80 = 20        # arch
    Sm89 = auto()
    Sm90 = auto()
    RRR = 30        # layout
    RCR = auto()
    RCC = auto()
    def __str__(self):
        return {
            Meta.Normal: "Normal",
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

def run_flux_profiling(input: torch.Tensor, weight: torch.Tensor, config: TuningConfig, fp):
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

    tuning = torch.zeros(20, dtype=torch.int8, device='cpu')
    output = torch.empty([m, n], dtype=input.dtype, device=input.device, requires_grad=False)
    op = flux.GemmNormal(input_dtype=input.dtype, output_dtype=input.dtype, transpose_weight=config.transpose_weight)

    fastest_time = 99999
    fastest_id = 0
    for id in range(100):
        # warmup and check, tuning_id=id
        tuning[0], tuning[1] = 1, id
        output = op.forward(input, weight, bias=None, output_buf=None, 
                            input_scale=None, weight_scale=None, output_scale=None, 
                            tuning = tuning, fast_accum=False)
        if (output is None):
            break

        warmup_iters = 10
        iters = 20
        for i in range(warmup_iters + iters):
            if (i == warmup_iters):
                torch.cuda.synchronize()
                start = time.time()
            tuning[0], tuning[1] = 1, id
            output = op.forward(input, weight, bias=None, output_buf=None, 
                                input_scale=None, weight_scale=None, output_scale=None, 
                                tuning = tuning, fast_accum=False)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start
        if (fastest_time > elapsed_time):
            fastest_time = elapsed_time
            fastest_id = id

    tuning[0], tuning[1] = 1, fastest_id
    # fp.write("fastest: {0}ms, {1}".format(str(fastest_time * 1000 / iters), str(fastest_id)))
    output = op.forward(input, weight, bias=None, output_buf=None, 
                        input_scale=None, weight_scale=None, output_scale=None, 
                        tuning = tuning, fast_accum=False)

    # tins.add({1,27648,5120,(int8_t)ME::Normal,(int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP16, (int8_t)ME::FP32, (int8_t)ME::RCR, (int8_t)ME::Sm80}, 5);
    # tuning: 0:meta_len, 1:id, 2~meta
    meta_len = tuning[0]       # len
    meta_str = ''              #              
    for i in range(2, meta_len):
        meta_str += "(int8_t)ME::" + str(Meta(tuning[i].item())) + ','
    meta_str += "(int8_t)ME::" + str(Meta(tuning[meta_len].item()))

    message = "  tins.add({{{0}, {1}, {2}, {3}}}, /*id*/{4}); // {5}ms\n".format(m, n, k, meta_str, str(fastest_id), str(round(fastest_time * 1000 / iters, 3)))
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
    flux_output = run_flux_profiling(input, weight, config, fp)

    if config.dtype == torch.bfloat16:
        atol, rtol = 0.02, 0.02
    else:
        atol, rtol = 0.01, 0.01
    flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)

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

    tag = "Normal"
    fp = {}
    fp[tag] = open("tuned_{0}_sm89.cu".format(tag.lower()), "w")

    fp[tag].write('#include "flux/ops_impl/normal/gemm_v2_impl.h"\n')
    fp[tag].write('#include "flux/ops_impl/normal/gemm_v2_simt_impl.h"\n\n')
    fp[tag].write('namespace xop {\n')
    fp[tag].write('using namespace cutlass;\n')
    fp[tag].write('using ME = UnifiedMetaEnum;\n\n')
    fp[tag].write('static int tuned_{0}_sm89 = []() {{\n'.format(tag.lower()))
    
    fp[tag].write('  TunedConfigRegister& tins = TunedConfigRegister::instance();\n')
    
    arch: int = flux.get_arch()
    name: str = f"config_single_gemm_sm{arch}"
    config_space = gen_tuning_space()
    for i, config in enumerate(config_space):
        print(f"==== #{i + 1}/{len(config_space)} Tuning for {config}")
        tune_one_config(config=config, fp=fp[tag])

    fp[tag].write('  return 0;\n}();\n}')
    fp[tag].write('// clang-format on')