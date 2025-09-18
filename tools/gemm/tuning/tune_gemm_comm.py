
import argparse
import os
from functools import partial
from typing import List
import multiprocessing as mp
from typing import Any

import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import xop
import xop.util as xutil
from tune_common import Meta, TuningConfig
import tune_common as common 

common.init_test_env(3)
print = partial(print, flush=True)

GEMM_COMM_ENABLE_CUDA_GRAPH = 1
warmup_iters = 20
pref_iters = 100
is_use_fp16_acc = False # True

class GemmAllreduceV2Schema:
    name = "GemmAllreduceV2"
    sub_schema = [Meta.GemmAllreduce]
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
    def get_ref_output(self, rank: int, group: ProcessGroup, 
                       input: torch.Tensor, weight: torch.Tensor, 
                       input_scale: torch.Tensor, weight_scale: torch.Tensor,
                       bias: torch.Tensor):
        output = torch.nn.functional.linear(input, weight, bias) #, out=output
        dist.all_reduce(output, group=group)
        return output.cpu()
    
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
    impl = "GemmGroupedBlockScaleFp8Sm90Impl"
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
        "GemmAllreduceV2": GemmAllreduceV2Schema(),
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
    
def run_xop_profiling_graph(rank: int, group: ProcessGroup, 
                      schema, input: torch.Tensor, weight: torch.Tensor, 
                      input_scale: torch.Tensor, weight_scale: torch.Tensor,
                      bias: torch.Tensor, config: TuningConfig, fp):
    m = input.size(0)
    k = input.size(1)
    if config.transpose_weight:
        weight = weight.t().contiguous()
        n = weight.size(1)
    else:
        n = weight.size(0)
    g = 1

    tuning = torch.zeros(100, dtype=torch.int16, device='cpu')
    output = torch.empty([m, n], dtype=config.dtypeC, device=input.device, requires_grad=False)
    op = xop.GemmCommRs(
        input_dtype=config.dtypeA,
        output_dtype=config.dtypeC,
        transpose_weight=config.transpose_weight,
        group=group,
        rank=rank,
    )
    
    def fn(tuning):
        return op.forward(input, weight, output=output, bias=bias,
            input_scale=input_scale, weight_scale=weight_scale, output_scale=None,
            tuning = tuning, fast_accum=is_use_fp16_acc)
        
    schema_cnt = 10
    func_graph = []
    tuning_data = []
    sub_schema = schema.sub_schema[0]
    for id in range(schema_cnt):
        # preallocate
        tuning[0], tuning[1], tuning[2] = 1, id, sub_schema
        fn(tuning)
        tuning[0], tuning[1], tuning[2] = 1, id, sub_schema
        
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), op.ar.capture():
            with torch.cuda.graph(graph):        
                fn(tuning)

        # func_graph.append(lambda: graph.replay())
    
        for i in range(warmup_iters + pref_iters):
            if (i == warmup_iters):
                torch.cuda.synchronize()
                start = time.time()
            graph.replay()
        torch.cuda.synchronize()
        elapsed_time = time.time() - start
        tuning_data.append((elapsed_time, id, sub_schema))
    
    if (rank == 0):
        common.write_tuning_result(fp, "Add2Comm", fn, [m,n,k,g], tuning, tuning_data, pref_iters, mode=2)
    return None # output.cpu()

def run_xop_profiling(rank: int, group: ProcessGroup, 
                      schema, input: torch.Tensor, weight: torch.Tensor, 
                      input_scale: torch.Tensor, weight_scale: torch.Tensor,
                      bias: torch.Tensor, config: TuningConfig, fp):
    m = input.size(0)
    k = input.size(1)
    if config.transpose_weight:
        weight = weight.t().contiguous()
        n = weight.size(1)
    else:
        n = weight.size(0)
    g = 1

    output = torch.empty([m, n], dtype=config.dtypeC, device=input.device, requires_grad=False)
    op = xop.GemmCommRs(
        input_dtype=config.dtypeA,
        output_dtype=config.dtypeC,
        transpose_weight=config.transpose_weight,
        group=group,
        rank=rank,
    )
    
    def fn(tuning):
        return op.forward(
            input,
            weight,
            output=output,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output_scale=None,
            tuning = tuning,
            fast_accum=is_use_fp16_acc,
        )
    common.profiling_core(fn, "Add2Comm", [m,n,k,g], schema, warmup_iters, pref_iters, fp)
        
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
    
    tuning = torch.zeros(100, dtype=torch.int16, device='cpu')    
    op = xop.GemmNormal(input_dtype=config.dtypeA, output_dtype=config.dtypeC, transpose_weight=config.transpose_weight)
    
    def fn(tuning):
        return op.grouped_forward(inputs, weights, outputs=outputs, 
                                  inputs_scale=inputs_scale, weights_scale=weights_scale, 
                                  tuning=tuning)
    common.profiling_core((fn, tuning, "Add2Comm", [m,n,k,g], schema, warmup_iters, pref_iters, fp))

    return torch.cat(outputs, dim=0).cpu()

def tune_one_config(rank: int, xop_group: ProcessGroup, nccl_group: ProcessGroup, schema, config: TuningConfig, fp):
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
        ref_output = schema.get_ref_output(rank, nccl_group, x, y, x_scale, y_scale, bias)
        if GEMM_COMM_ENABLE_CUDA_GRAPH == 1:
            xop_output = run_xop_profiling_graph(rank, xop_group, schema, x, y, x_scale, y_scale, bias, config, fp)
        else:
            xop_output = run_xop_profiling(rank, xop_group, schema, x, y, x_scale, y_scale, bias, config, fp)

    if ref_output is not None and xop_output is not None:
        if config.dtypeC == torch.bfloat16:
            atol, rtol = 0.02, 0.02
        else:
            atol, rtol = 0.01, 0.01

        if is_use_fp16_acc:
            atol, rtol = 0.1, 0.1
        xutil.torch_allclose(xop_output, ref_output, atol=atol, rtol=rtol, print_prefix="rank<" + str(rank) + ">")
    
def run_worker(world_size, rank, port, args, a):
    device = torch.device(f"cuda:{rank + xop.ALLREDUCE_GPUID_OFFSET}")
    torch.cuda.set_device(device)
    
    distributed_init_method = f"tcp://localhost:{port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    nccl_group = dist.group.WORLD
    xop_group = torch.distributed.new_group(list(range(world_size)), backend="gloo")
    
    #################################
    tag = args.schema
    fp = {}
    fp[tag] = open(args.output_path+"/tuned_config_{0}_rank{1}.cu".format(tag.lower(), rank), "w")
    common.gen_tuning_file_head(fp[tag], tag)
    
    schema = str2schema(args.schema)
    config_space = get_tuning_space(schema)
    for i, config in enumerate(config_space):
        print(f"==== #{i + 1}/{len(config_space)} Tuning for {config}")
        tune_one_config(rank, xop_group, nccl_group, schema, config=config, fp=fp[tag])

    common.gen_tuning_file_tail(fp[tag])
    #################################
    # exponent = args.M  # 65536: 17
    # run(world_size, rank, 1, args, xop_group, nccl_group, xop_perf, torch_perf)
    # for m in range(1, exponent):
    #     m = 2**m
    #     run(world_size, rank, m, args, xop_group, nccl_group, xop_perf, torch_perf)
        
    dist.barrier(group=nccl_group)
    dist.destroy_process_group(group=nccl_group)    

    
    
def multi_process_parallel(
    world_size: int, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    port = 12355
    for i in range(world_size):
        proc_args = (world_size, i, port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.daemon = True
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert (
            procs[i].exitcode == 0
        ), f"Process {i} failed with exit code {procs[i].exitcode}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=str, default="None")
    parser.add_argument("--output_path", default="./tools/", type=str, help="Directory to store generated files")
    args = parser.parse_args()

    if (args.schema == "None"):
        print("usage: python3 tools/gemm/tuning/tune_gemm_comm.py --schema=GemmAllreduceV2 (GemmAllreduceV2 / GemmV2BlockScaleFp8 / GemmBlockScaleFp8 / GemmGroupedBlockScaleFp8)")
        exit()

    if args.output_path and not os.path.isdir(args.output_path):
        raise Exception(f"{args.output_path} not exist")

    world_sizes = [2] # [2,4,8]
    for world_size in world_sizes:
        available_gpus = torch.cuda.device_count()
        if world_size > available_gpus:
            print(
                f"Skipping world_size={world_size}, requires {world_size} GPUs, found {available_gpus}"
            )
            continue

        print(f"Running test for world_size={world_size}")
        multi_process_parallel(
            world_size, run_worker, target_args=(args, 1)
        )
        print(f"custom allreduce tp = {world_size}: OK")
        