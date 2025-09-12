
import argparse
import os
from functools import partial
from typing import List
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch

import xop
import xop.util as xutil
from tune_common import Meta, TuningConfig
import tune_common as common 

common.init_test_env(3)
print = partial(print, flush=True)

warmup_iters = 20
pref_iters = 20
is_use_fp16_acc = False # True

class GemmNormalSchema:
    name = "GemmNormal"
    sub_schema = [Meta.GemmNormal] # GemmNormalSimt, Meta.GemmLt
    # test_input_dtype = torch.float16
    # space_dtype = [(torch.float16,torch.float16,torch.float16)] # (torch.bfloat16,torch.bfloat16,torch.bfloat16)
    if is_use_fp16_acc:
        test_input_dtype = torch.float16
        space_dtype = [(torch.float16,torch.float16,torch.float16)]
    else:
        test_input_dtype = torch.bfloat16
        space_dtype = [(torch.bfloat16,torch.bfloat16,torch.bfloat16)]
    def gen_scale(self, input: torch.Tensor, weight: torch.Tensor):
        return input, None, weight, None
    def get_ref_output(self, input: torch.Tensor, weight: torch.Tensor, 
                       input_scale: torch.Tensor, weight_scale: torch.Tensor,
                       bias: torch.Tensor):
        output = torch.matmul(input, weight.t())
        if (bias != None):
            output += bias
        return output.cpu()

class GemmV2BlockScaleFp8Schema:
    impl = "GemmV2BlockScaleFp8"
    sub_schema = [Meta.GemmBlockScaleFp8]
    
    if is_use_fp16_acc:
        test_input_dtype = torch.float16
        space_dtype = [(torch.float8_e4m3fn,torch.float8_e4m3fn,torch.float16)]
    else:
        test_input_dtype = torch.bfloat16
        space_dtype = [(torch.float8_e4m3fn,torch.float8_e4m3fn,torch.bfloat16)]
    def gen_scale(self, input: torch.Tensor, weight: torch.Tensor):
        x, x_scale = xutil.per_token_cast_to_fp8(input, is_use_fp16_acc)
        y, y_scale = xutil.per_block_cast_to_fp8(weight, is_use_fp16_acc)
        x_scale = xop.gemm_v2_blockscale_fp8_scale_a_preprocess(x_scale)
        return x, x_scale, y, y_scale.contiguous()
    def get_ref_output(self, input: torch.Tensor, weight: torch.Tensor, 
                       input_scale: torch.Tensor, weight_scale: torch.Tensor,
                       bias: torch.Tensor):
        # output = torch.matmul(input, weight.t())
        # return output.cpu()
        return None
class GemmBlockScaleFp8Schema:
    impl = "GemmBlockScaleFp8"
    sub_schema = [Meta.GemmBlockScaleFp8]
    test_input_dtype = torch.bfloat16
    space_dtype = [(torch.float8_e4m3fn,torch.float8_e4m3fn,torch.bfloat16)]
    def gen_scale(self, input: torch.Tensor, weight: torch.Tensor):
        x, x_scale = xutil.per_token_cast_to_fp8(input)
        y, y_scale = xutil.per_block_cast_to_fp8(weight)
        return x, x_scale.t().contiguous(), y, y_scale.t().contiguous()
    def get_ref_output(self, input: torch.Tensor, weight: torch.Tensor, 
                       input_scale: torch.Tensor, weight_scale: torch.Tensor,
                       bias: torch.Tensor):
        # output = torch.matmul(input, weight.t())
        # return output.cpu()
        return None

class GemmGroupedBlockScaleFp8Schema:
    impl = "GemmGroupedBlockScaleFp8Impl"
    sub_schema = [Meta.GemmGroupedBlockScaleFp8]
    test_input_dtype = torch.bfloat16
    space_dtype = [(torch.float8_e4m3fn,torch.float8_e4m3fn,torch.bfloat16)]
    def gen_scale(self, input: torch.Tensor, weight: torch.Tensor, config: TuningConfig):
        x_list = []
        x_scale_list = []
        y_list = []
        y_scale_list = []

        for i in range(config.G):
            x, x_scale = xutil.per_token_cast_to_fp8(input)
            y, y_scale = xutil.per_block_cast_to_fp8(weight)

            x_list.append(x)
            x_scale_list.append(x_scale.t().contiguous())
            y_list.append(y)
            y_scale_list.append(y_scale.t().contiguous())

        return x_list, x_scale_list, y_list, y_scale_list
        
    def get_ref_output(self, input: torch.Tensor, weight: torch.Tensor, 
                       input_scale: torch.Tensor, weight_scale: torch.Tensor,
                       bias: torch.Tensor):
        # output = torch.matmul(input, weight.t())
        # return output.cpu()
        return None
      
def str2schema(schema_name):
    string_to_schema = {
        "GemmNormal": GemmNormalSchema(),
        "GemmV2BlockScaleFp8": GemmV2BlockScaleFp8Schema(),
        "GemmBlockScaleFp8": GemmBlockScaleFp8Schema(),
        "GemmGroupedBlockScaleFp8": GemmGroupedBlockScaleFp8Schema(),
    }
    return string_to_schema.get(schema_name, None)

# schema 1: [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]
def get_tuning_space(schema):
    space_G = [1]
    space_M = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096] #,8192,16384,32768,65536 [8192] # list(range(1, 31)) # [8,16,32,64,128,512,1024] #, 2048, 4096   # , 16384
    space_NK = [(4096, 4096)] #(576, 7168) (3584,5120), (5120,2560), (5120,13824), (27648,5120), 49152
    space_has_bias = [False]    
    # space_G = [4, 8]
    # space_M = [2048, 4096] # [8,16,32,64,128,512,1024] #, 2048, 4096   # , 16384
    # space_NK = [(576, 7168)] #(576, 7168) (3584,5120), (5120,2560), (5120,13824), (27648,5120), 49152
    return common.gen_tuning_space(schema.space_dtype, space_G, space_M, space_NK, space_has_bias)
    
def run_xop_profiling(schema, input: torch.Tensor, weight: torch.Tensor, 
                        input_scale: torch.Tensor, weight_scale: torch.Tensor,
                        bias: torch.Tensor,
                        config: TuningConfig, fp):
    m = input.size(0)
    k = input.size(1)
    if config.transpose_weight:
        weight = weight.t().contiguous()
        n = weight.size(1)
    else:
        n = weight.size(0)
    g = 1

    output = torch.empty([m, n], dtype=config.dtypeC, device=input.device, requires_grad=False)
    op = xop.GemmNormal(input_dtype=config.dtypeA, output_dtype=config.dtypeC, transpose_weight=config.transpose_weight)

    def fn(tuning):
        return op.forward(input, weight, output=output, bias=bias, 
                          input_scale=input_scale, weight_scale=weight_scale, output_scale=None, 
                          tuning=tuning, fast_accum=is_use_fp16_acc)

    common.profiling_core(fn, "add", [m,n,k,g], schema, warmup_iters, pref_iters, fp)
    return output.cpu()

def run_xop_grouped_profiling(schema, inputs: List[torch.Tensor], weights: List[torch.Tensor], 
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
    
    op = xop.GemmNormal(input_dtype=config.dtypeA, output_dtype=config.dtypeC, transpose_weight=config.transpose_weight)

    def fn(tuning):
        return op.grouped_forward(inputs, weights, outputs=outputs, 
                                  inputs_scale=inputs_scale, weights_scale=weights_scale, 
                                  tuning=tuning)
    common.profiling_core((fn, "add", [m,n,k,g], schema, warmup_iters, pref_iters, fp))

    return torch.cat(outputs, dim=0).cpu()

def tune_one_config(schema, config: TuningConfig, fp):
    input = torch.rand((config.M, config.K), dtype=schema.test_input_dtype).cuda() # torch.bfloat16
    weight = torch.rand((config.N, config.K), dtype=schema.test_input_dtype).cuda()
    # start_time = time.time()

    # print(f"torch compute time: {(time.time() - start_time) * 1000} ms")
    if (isinstance(schema, GemmGroupedBlockScaleFp8Schema)):
        x, x_scale, y, y_scale = schema.gen_scale(input.clone(), weight.clone(), config)
        ref_output = schema.get_ref_output(x, y, x_scale, y_scale)
        xop_output = run_xop_grouped_profiling(schema, x, y, x_scale, y_scale, config, fp)
    else:
        x, x_scale, y, y_scale = schema.gen_scale(input.clone(), weight.clone())
        bias = None
        if config.has_bias:
            bias = torch.zeros([y.size(0)], dtype=x.dtype, device=x.device, requires_grad=False)
        ref_output = schema.get_ref_output(x, y, x_scale, y_scale, bias)
        xop_output = run_xop_profiling(schema, x, y, x_scale, y_scale, bias, config, fp)

    if ref_output is not None:
        if config.dtypeC == torch.bfloat16:
            atol, rtol = 0.02, 0.02
        else:
            atol, rtol = 0.01, 0.01

        if is_use_fp16_acc:
            atol, rtol = 0.1, 0.1
        xutil.torch_allclose(xop_output, ref_output, atol=atol, rtol=rtol)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=str, default="None")
    parser.add_argument("--output_path", default="./tools/", type=str, help="Directory to store generated files")
    args = parser.parse_args()

    if (args.schema == "None"):
        print("usage: python3 tools/gemm/tuning/tune_gemm_normal.py --schema=GemmNormal (GemmNormal(GemmNormalSimt) / GemmV2BlockScaleFp8 / GemmBlockScaleFp8 / GemmGroupedBlockScaleFp8)")
        exit()

    if args.output_path and not os.path.isdir(args.output_path):
        raise Exception(f"{args.output_path} not exist")

    tag = args.schema
    fp = {}
    fp[tag] = open(args.output_path+"/tuned_config_{0}.cu".format(tag.lower()), "w")
    common.gen_tuning_file_head(fp[tag], tag)
    
    schema = str2schema(args.schema)
    config_space = get_tuning_space(schema)
    for i, config in enumerate(config_space):
        print(f"==== #{i + 1}/{len(config_space)} Tuning for {config}")
        tune_one_config(schema, config=config, fp=fp[tag])

    common.gen_tuning_file_tail(fp[tag])