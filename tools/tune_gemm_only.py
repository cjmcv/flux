################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import argparse
import dataclasses
import itertools
import os
from functools import partial
from typing import List

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
    space_dtype = [torch.bfloat16] # , torch.float16
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
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    output = torch.matmul(input, weight.t())
    end_event.record()
    torch.cuda.synchronize()

    print(f"torch.matmul: {start_event.elapsed_time(end_event)} ms")
    return output.cpu()


def run_flux_profiling(
    prof_ctx: flux.ProfilingContext, input: torch.Tensor, weight: torch.Tensor, config: TuningConfig
):
    m = input.size(0)
    if config.transpose_weight:
        weight = weight.t().contiguous()
        n = weight.size(1)
    else:
        n = weight.size(0)

    bias = None
    if config.has_bias:
        bias = torch.zeros([m, n], dtype=input.dtype, device=input.device, requires_grad=False)

    output = torch.empty([m, n], dtype=input.dtype, device=input.device, requires_grad=False)
    op = flux.GemmOnly(weight.dtype, transpose_weight=config.transpose_weight)

    output = op.profiling(input, weight, bias=bias, output_buf=output, prof_ctx=prof_ctx)
    return output.cpu()


def tune_one_config(prof_ctx: flux.ProfilingContext, config: TuningConfig):
    input = torch.rand((config.M, config.K), dtype=config.dtype).cuda()
    weight = torch.rand((config.N, config.K), dtype=config.dtype).cuda()
    # start_time = time.time()
    torch_output = get_torch_output(input, weight)
    # print(f"torch compute time: {(time.time() - start_time) * 1000} ms")
    flux_output = run_flux_profiling(prof_ctx, input, weight, config)

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

    arch: int = flux.get_arch()
    name: str = f"config_gemm_only_sm{arch}"
    prof_ctx = flux.ProfilingContext(name)
    config_space = gen_tuning_space()
    for i, config in enumerate(config_space):
        print(f"==== #{i + 1}/{len(config_space)} Tuning for {config}")
        tune_one_config(prof_ctx=prof_ctx, config=config)
        print(prof_ctx.get_latest_prof_result())

    if os.path.isdir(args.output_dir):
        code_path = os.path.join(args.output_dir, f"{name}.cu")
        result_path = os.path.join(args.output_dir, f"{name}.prof.log")

        with open(code_path, "w") as fout:
            print(prof_ctx.get_code(), file=fout)

        with open(result_path, "w") as fout:
            for record in prof_ctx.get_all_prof_results():
                print(record, file=fout)
    else:
        print("Generated Code:")
        print(prof_ctx.get_code())
        print()
        print("Profiling Results:")
        for record in prof_ctx.get_all_prof_results():
            print(record)
