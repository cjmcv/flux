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

import sys
from contextlib import contextmanager

import torch


def get_arch():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    major = properties.major
    minor = properties.minor
    return major * 10 + minor


def torch_allclose(x, y, rtol, atol, verbose=True):
    if not torch.allclose(x, y, rtol=rtol, atol=atol):
        print(f"shape of x: {x.shape}")
        print(f"shape of y: {y.shape}")

        print("x:", file=sys.stderr)
        print(x, file=sys.stderr)
        print("y:", file=sys.stderr)
        print(y, file=sys.stderr)
        print("x-y", x - y, file=sys.stderr)
        diff_loc = torch.isclose(x, y, rtol=rtol, atol=atol) == False
        print("x diff:", file=sys.stderr)
        print(x[diff_loc], file=sys.stderr)
        print("y diff:", file=sys.stderr)
        print(y[diff_loc], file=sys.stderr)
        num_diff = torch.sum(diff_loc)

        if len(y.shape) == 1:
            diff_rate = num_diff / y.shape[0]
        else:
            diff_rate = num_diff / (y.shape[0] * y.shape[1])
        print(f"diff count: {num_diff} ({diff_rate*100:.3f}%), {list(y.shape)}", file=sys.stderr)
        max_diff = torch.max(torch.abs(x - y))
        rtol_abs = rtol * torch.min(torch.abs(y))
        print(f"diff max: {max_diff}, atol: {atol}, rtol_abs: {rtol_abs}", file=sys.stderr)
        diff_indices = (diff_loc == True).nonzero(as_tuple=False)
        print(f"diff locations:\n{diff_indices}", file=sys.stderr)
        print("--------------------------------------------------------------\n", file=sys.stderr)
        raise RuntimeError

    if verbose:
        print("all close!")


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype.itemsize == 1 and dtype.is_floating_point


@contextmanager
def with_torch_deterministic(mode: bool, warn_only: bool = True):
    old_mode = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(old_mode, warn_only=warn_only)

__all__ = [
    "is_fp8_dtype",
    "get_arch",
    "torch_allclose",
    "with_torch_deterministic",
]
