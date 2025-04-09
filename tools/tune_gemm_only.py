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
import torch

import flux

# <NT> 表示使用 16 个工作空间，每个工作空间的大小为 8MB, 这种配置适用于大多数常见的深度学习模型和任务，能够在性能和内存占用之间取得较好的平衡.
# 工作空间（workspace）是一个用于存储中间计算结果、临时数据或者辅助信息的内存区域。
# 通过工作空间，可以避免频繁地在设备内存和主机内存之间进行数据传输，从而提高计算效率。
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# <NT> 让 PyTorch 采用确定性算法，保证每次运行的结果相同（某些情形下为了提升计算性能，可能会使用非确定性算法）
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=8)
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
np.random.seed(3)
# <NT> functools.partial作用是固定一个函数的某些参数，然后生成一个新的可调用对象。
# print里，当 flush=True 时，会强制将缓冲区中的数据立即输出，而不是等到缓冲区满或者程序结束才输出。
# flush默认为False，这里则将 flush 参数固定为 True，然后把新生成的函数对象重新赋值给 print。
print = partial(print, flush=True)


@dataclasses.dataclass
class TuningConfig:
    M: int
    N: int
    K: int
    transpose_weight: bool
    dtype: str
    has_bias: bool

# <NT> itertools.product 会从多个集合中各取一个元素组成的所有可能的有序对或元组的集合
def gen_tuning_space():
    space: List[TuningConfig] = []
    space_M = [1024, 2048, 4096, 8192, 16384]
    space_N = [6144, 12288, 49152]
    space_K = [3072, 6144, 12288]
    space_transpose_weight = [False, True]
    space_dtype = [torch.bfloat16, torch.float16]
    space_has_bias = [False, True]
    for M, N, K, transpose_weight, dtype, has_bias in itertools.product(
        space_M, space_N, space_K, space_transpose_weight, space_dtype, space_has_bias
    ):
        config = TuningConfig(
            M=M, N=N, K=K, transpose_weight=transpose_weight, dtype=dtype, has_bias=has_bias
        )
        space.append(config)
    return space


def get_torch_output(input: torch.Tensor, weight: torch.Tensor):
    return torch.matmul(input, weight.t()).cpu()


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
    torch_output = get_torch_output(input, weight)
    flux_output = run_flux_profiling(prof_ctx, input, weight, config)

    if config.dtype == torch.bfloat16:
        atol, rtol = 0.02, 0.02
    else:
        atol, rtol = 0.01, 0.01
    flux.torch_allclose(flux_output, torch_output, atol=atol, rtol=rtol)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="", type=str, help="Directory to store generated files"
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
