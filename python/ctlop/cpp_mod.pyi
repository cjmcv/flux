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

from typing import List, Optional
import torch

# class TuningRecord:
#     pass

# class ProfilingContext:
#     def __init__(self, name: str): ...
#     def get_code(self) -> str: ...
#     def get_latest_prof_result(self) -> str: ...
#     def get_all_prof_results(self) -> List[str]: ...
#     def get_latest_record(self) -> TuningRecord: ...
#     def get_all_records(self) -> List[TuningRecord]: ...

# class SingleGemm:
#     """
#     support 4 mode: FP16/BF16 mode, FP8 mode, INT8 Dequant mode, INT8 (GEMM)Only mode

#     About shapes:
#     * input: [M, K] for all types
#     * weight: [K, N] if transpose_weight, [N, K] if not transpose_weight.
#     * bias: [1, N] for FP8 or INT8 Dequant, [M, N] for FP16/BF16/INT8(Only).
#     * input_scale: always None for FP16/BF16/Int8 Only. [M, 1] for INT8 Dequant. [1] for FP8
#     * weight_scale: always None for FP16/BF16/Int8 Only. [1, N] for INT8 Dequant. [1] for FP8
#     * output_scale: always None for FP16/BF16/INT8 Dequant/INT8 Only. for FP8 ???

#     for transpose_weight=True:
#         for FP16:  output_FP16 = [input_FP16 * weight_FP16]_FP32.to(FP16) + bias_FP16
#         for BF16:  output_BF16 = [input_BF16 * weight_BF16]_FP32.to(BF16) + bias_BF16
#         for FP8:   output_BF16 = [[input_FP8 * weight_FP8]_FP32 * input_scale_FP32 * weight_scale_FP32]_FP32.to(BF16) + bias_FP16
#         for INT8 Dequant:  output_BF16 = [[input_INT8 * weight_INT8]_INT32.to(FP32) * input_scale_FP32 * weight_scale_FP32]_FP32.to(BF16) + bias_BF16
#         for INT8 Only: output_INT32 = [input_INT8 * weight_INT8]_INT32 + bias_INT32
#     for transpose_weight=False, replace weight with weight.T
#     """

#     def __init__(
#         self,
#         input_dtype: torch.dtype,
#         output_dtype: Optional[torch.dtype] = None,
#         transpose_weight: bool = False,
#     ): ...
#     def forward(
#         self,
#         input: torch.Tensor,
#         weight: torch.Tensor,
#         bias: Optional[torch.Tensor] = None,
#         output_buf: Optional[torch.Tensor] = None,
#         input_scale: Optional[torch.Tensor] = None,
#         weight_scale: Optional[torch.Tensor] = None,
#         output_scale: Optional[torch.Tensor] = None,
#         fast_accum: bool = False,
#     ) -> torch.Tensor: ...
#     def profiling(
#         self,
#         input: torch.Tensor,
#         weight: torch.Tensor,
#         bias: Optional[torch.Tensor] = None,
#         output_buf: Optional[torch.Tensor] = None,
#         input_scale: Optional[torch.Tensor] = None,
#         weight_scale: Optional[torch.Tensor] = None,
#         output_scale: Optional[torch.Tensor] = None,
#         fast_accum: bool = False,
#         prof_ctx: Optional[ProfilingContext] = None,
#     ) -> torch.Tensor: ...


class GemmNormal:
    def __init__(
        self,
        input_dtype: torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
        transpose_weight: bool = False,
    ): ...
    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output_buf: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        fast_accum: bool = False,
        tuning_id: int = -1,
    ) -> torch.Tensor: ...