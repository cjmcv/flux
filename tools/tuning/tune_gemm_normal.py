
import argparse
import dataclasses
import itertools
import os
from functools import partial
from typing import List
import numpy as np

import time
import torch

import ctlop
from tune_common import Meta, per_token_cast_to_fp8, per_block_cast_to_fp8

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

warmup_iters = 10
pref_iters = 20

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


class GemmNormalSchema:
    name = "GemmNormal"
    sub_schema = [Meta.GemmNormal, Meta.GemmNormalSimt]
    space_dtype = [(torch.bfloat16,torch.bfloat16,torch.bfloat16)]
    def gen_scale(self, input: torch.Tensor, weight: torch.Tensor):
        return input, None, weight, None
    def get_ref_output(self, input: torch.Tensor, weight: torch.Tensor, 
                       input_scale: torch.Tensor, weight_scale: torch.Tensor):
        output = torch.matmul(input, weight.t())
        return output.cpu()
class GemmBlockScaleFp8Schema:
    impl = "GemmBlockScaleFp8"
    sub_schema = [Meta.GemmBlockScaleFp8]
    space_dtype = [(torch.float8_e4m3fn,torch.float8_e4m3fn,torch.bfloat16)]
    def gen_scale(self, input: torch.Tensor, weight: torch.Tensor):
        x, x_scale = per_token_cast_to_fp8(input)
        y, y_scale = per_block_cast_to_fp8(weight)
        return x, x_scale.t().contiguous(), y, y_scale.t().contiguous()
    def get_ref_output(self, input: torch.Tensor, weight: torch.Tensor, 
                       input_scale: torch.Tensor, weight_scale: torch.Tensor):
        # output = torch.matmul(input, weight.t())
        # return output.cpu()
        return None

class GemmGroupedBlockScaleFp8Schema:
    impl = "GemmGroupedBlockScaleFp8Impl"
    sub_schema = [Meta.GemmGroupedBlockScaleFp8]
    space_dtype = [(torch.float8_e4m3fn,torch.float8_e4m3fn,torch.bfloat16)]
    def gen_scale(self, input: torch.Tensor, weight: torch.Tensor, config: TuningConfig):
        x_list = []
        x_scale_list = []
        y_list = []
        y_scale_list = []

        for i in range(config.G):
            x, x_scale = per_token_cast_to_fp8(input)
            y, y_scale = per_block_cast_to_fp8(weight)

            x_list.append(x)
            x_scale_list.append(x_scale.t().contiguous())
            y_list.append(y)
            y_scale_list.append(y_scale.t().contiguous())

        return x_list, x_scale_list, y_list, y_scale_list
        
    def get_ref_output(self, input: torch.Tensor, weight: torch.Tensor, 
                       input_scale: torch.Tensor, weight_scale: torch.Tensor):
        # output = torch.matmul(input, weight.t())
        # return output.cpu()
        return None
      
def str2schema(schema_name):
    string_to_schema = {
        "GemmNormal": GemmNormalSchema(),
        "GemmBlockScaleFp8": GemmBlockScaleFp8Schema(),
        "GemmGroupedBlockScaleFp8": GemmGroupedBlockScaleFp8Schema(),
    }
    return string_to_schema.get(schema_name, None)

def gen_tuning_space(schema):
    space: List[TuningConfig] = []
    # space_G = [1]
    # space_M = list(range(1, 31)) # [8,16,32,64,128,512,1024] #, 2048, 4096   # , 16384
    # space_NK = [(27648, 5120)] #(576, 7168) (3584,5120), (5120,2560), (5120,13824), (27648,5120), 49152
    
    space_G = [4, 8]
    space_M = [2048, 4096] # [8,16,32,64,128,512,1024] #, 2048, 4096   # , 16384
    space_NK = [(576, 7168)] #(576, 7168) (3584,5120), (5120,2560), (5120,13824), (27648,5120), 49152
    
    space_transpose_weight = [False] # , True
    space_dtype = schema.space_dtype
    space_has_bias = [False]
    for G, NK, M, transpose_weight, dtype, has_bias in itertools.product(
        space_G, space_NK, space_M, space_transpose_weight, space_dtype, space_has_bias
    ):
        config = TuningConfig(
            G=G, M=M, N=NK[0], K=NK[1], transpose_weight=transpose_weight, 
            dtypeA=dtype[0], dtypeB=dtype[1], dtypeC=dtype[2], has_bias=has_bias
        )
        space.append(config)
    return space

def profiling_core(tuning, shape, fn: callable, fp):
    m = shape[0]
    n = shape[1]
    k = shape[2]
    g = shape[3]

    tuning_data = []
    for sub_schema in schema.sub_schema:
        for id in range(500):
            # warmup and check if exist.
            tuning[0], tuning[1], tuning[2] = 1, id, sub_schema
            ret = fn()
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

    tuning_data.sort()
    tuning[0], tuning[1], tuning[2] = 1, tuning_data[0][1], tuning_data[0][2]
    fn()

    # tuning: 0:meta_len, 1:id, 2:schema, 3:~meta
    meta_len = tuning[0]       # len
    meta_str = ''              #
    for i in range(3, meta_len):
        meta_str += "(int16_t)ME::" + str(Meta(tuning[i].item())) + ','
    meta_str += "(int16_t)ME::" + str(Meta(tuning[meta_len].item()))

    for sid in range(min(3, len(tuning_data))):
        prefix = ''
        if sid != 0:
            prefix = '// '
        message = "  {0}tins.add({{{1},{2},{3},{4},{5}}}, /*config*/{{{6}, {7}}}); // {8}ms\n".format(prefix, m, n, k, g, meta_str, str(tuning_data[sid][1]), "(int16_t)ME::"+str(tuning_data[sid][2]), str(round(tuning_data[sid][0] * 1000 / pref_iters, 3)))
        fp.write(message)
        fp.flush()
        print(message)

    # double check
    fastest_id = tuning_data[0][1]
    fastest_schema = tuning_data[0][2]
    if (fastest_id != tuning[1].item() or fastest_schema != tuning[2].item()):
        print("fastest_config is not matched: {0},{1} vs {2},{3}".format(str(fastest_id), str(fastest_schema), str(tuning[1].item()), str(tuning[2].item())))
        raise RuntimeError
    
def run_ctlop_profiling(schema, input: torch.Tensor, weight: torch.Tensor, 
                        input_scale: torch.Tensor, weight_scale: torch.Tensor,
                        config: TuningConfig, fp):
    m = input.size(0)
    k = input.size(1)
    if config.transpose_weight:
        weight = weight.t().contiguous()
        n = weight.size(1)
    else:
        n = weight.size(0)
    g = 1
    # bias = None
    # if config.has_bias:
    #     bias = torch.zeros([m, n], dtype=input.dtype, device=input.device, requires_grad=False)

    tuning = torch.zeros(100, dtype=torch.int16, device='cpu')
    output = torch.empty([m, n], dtype=config.dtypeC, device=input.device, requires_grad=False)
    op = ctlop.GemmNormal(input_dtype=config.dtypeA, output_dtype=config.dtypeC, transpose_weight=config.transpose_weight)

    def fn():
        return op.forward(input, weight, output=output, bias=None, 
                          input_scale=input_scale, weight_scale=weight_scale, output_scale=None, 
                          tuning=tuning, fast_accum=False)
    
    profiling_core(tuning, [m,n,k,g], fn, fp)
    return output.cpu()

def run_ctlop_grouped_profiling(schema, inputs: List[torch.Tensor], weights: List[torch.Tensor], 
                                inputs_scale: List[torch.Tensor], weights_scale: List[torch.Tensor],
                                config: TuningConfig, fp):
    m = inputs[0].size(0)
    k = inputs[0].size(1)
    n = weights[0].size(0)
    g = len(inputs)
    # print("groups: ", g)
    outputs = []
    for i in range(0, g):
        outputs.append(torch.empty([m, n], dtype=config.dtypeC, device=inputs[0].device, requires_grad=False))
    
    tuning = torch.zeros(100, dtype=torch.int16, device='cpu')
    op = ctlop.GemmNormal(input_dtype=config.dtypeA, output_dtype=config.dtypeC, transpose_weight=config.transpose_weight)

    def fn():
        return op.grouped_forward(inputs, weights, outputs=outputs, 
                                  inputs_scale=inputs_scale, weights_scale=weights_scale, 
                                  tuning=tuning)
    profiling_core(tuning, [m,n,k,g], fn, fp)

    return torch.cat(outputs, dim=0).cpu()

def tune_one_config(schema, config: TuningConfig, fp):
    input = torch.rand((config.M, config.K), dtype=torch.bfloat16).cuda()
    weight = torch.rand((config.N, config.K), dtype=torch.bfloat16).cuda()
    # start_time = time.time()

    # print(f"torch compute time: {(time.time() - start_time) * 1000} ms")
    if (isinstance(schema, GemmGroupedBlockScaleFp8Schema)):
        x, x_scale, y, y_scale = schema.gen_scale(input.clone(), weight.clone(), config)
        ref_output = schema.get_ref_output(x, y, x_scale, y_scale)
        ctlop_output = run_ctlop_grouped_profiling(schema, x, y, x_scale, y_scale, config, fp)
    else:
        x, x_scale, y, y_scale = schema.gen_scale(input.clone(), weight.clone())
        ref_output = schema.get_ref_output(x, y, x_scale, y_scale)
        ctlop_output = run_ctlop_profiling(schema, x, y, x_scale, y_scale, config, fp)

    if ref_output is not None:
        if config.dtypeC == torch.bfloat16:
            atol, rtol = 0.02, 0.02
        else:
            atol, rtol = 0.01, 0.01
        ctlop.torch_allclose(ctlop_output, ref_output, atol=atol, rtol=rtol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=str, default="None")
    parser.add_argument("--output_dir", default="./tools/", type=str, help="Directory to store generated files")
    args = parser.parse_args()

    if (args.schema == "None"):
        print("usage: python tools/tuning/tune_gemm_normal.py --schema=GemmGroupedBlockScaleFp8 (GemmNormal(GemmNormalSimt) / GemmBlockScaleFp8 / GemmGroupedBlockScaleFp8)")
        exit()

    if args.output_dir and not os.path.isdir(args.output_dir):
        raise Exception(f"{args.output_dir} not exist")

    tag = args.schema
    fp = {}
    fp[tag] = open("tuned_config_{0}.cu".format(tag.lower()), "w")

    fp[tag].write('#include "ctlop/ctlop.h"\n')
    fp[tag].write('#include "ctlop/ops_impl/global_resource.h"\n\n')
    fp[tag].write('namespace ctlop {\n')
    fp[tag].write('using namespace cutlass;\n')
    fp[tag].write('using ME = UnifiedMetaEnum;\n\n')
    fp[tag].write('static int tuned_config_{0} = []() {{\n'.format(tag.lower()))
    
    fp[tag].write('  TunedConfigRegister& tins = TunedConfigRegister::instance();\n')
    
    schema = str2schema(args.schema)
    config_space = gen_tuning_space(schema)
    for i, config in enumerate(config_space):
        print(f"==== #{i + 1}/{len(config_space)} Tuning for {config}")
        tune_one_config(schema, config=config, fp=fp[tag])

    fp[tag].write('  return 0;\n}();\n}')
    fp[tag].write('// clang-format on')