//===- gemm_only.h ----------------------------------------------- C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <torch/all.h>
#include "flux/tuning.h"

#define CHECK_TYPE(x, st) FLUX_CHECK_EQ(x.scalar_type(), st) << "Inconsistency type of Tensor " #x
#define CHECK_CUDA(x) FLUX_CHECK(x.is_cuda()) << #x << " must be a CUDA tensor"
#define CHECK_CONTIGUOUS(x) FLUX_CHECK(x.is_contiguous()) << #x << " must be contiguous"
#define CHECK_INPUT(x, st) \
  CHECK_CUDA(x);           \
  CHECK_CONTIGUOUS(x);     \
  CHECK_TYPE(x, st)
  
namespace bytedance::flux::ths_op {
class GemmOnly {
 public:
  GemmOnly(
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight);
  ~GemmOnly();

  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum);
  torch::Tensor profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      c10::intrusive_ptr<ProfilingContext> opt_ctx);

 private:
  class GemmOnlyImpl;
  GemmOnlyImpl *impl_ = nullptr;
};

}  // namespace bytedance::flux::ths_op
