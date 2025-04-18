//===- gemm_only.cc ----------------------------------------------- C++ ---===//
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

#include "comm_none/ths_op/gemm_only.h"
#include "flux/args/comm_none.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/ths_op/ths_op.h"
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime_api.h>
#include <utility>

namespace bytedance {
namespace flux {
namespace ths_op {
using torch::Tensor;

class GemmOnly::GemmOnlyImpl {
 private:
  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;
  const bool transpose_weight;
  torch::Tensor gemm_buffer;

 private:
  auto
  get_gemm_meta(bool has_bias, bool fast_accum) {
    auto arch = get_arch();
    auto input_dtype = from_torch_dtype(this->input_dtype);
    auto output_dtype = from_torch_dtype(this->output_dtype);
    DataTypeEnum accum_type = _FP32{}();

    auto dt_conf = make_gemm_dtype_config(
        input_dtype, input_dtype, has_bias ? output_dtype : _Void{}(), output_dtype, accum_type);

    // FP8 GEMM RRR layout is not supported, details can be viewed in issue #43
    auto gemm_layout = transpose_weight ? _RRR{}() : _RCR{}();
    auto impl = ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}();
    UnifiedImplMeta impl_spec = None{};

    bool use_fast_accum = fast_accum and dt_conf.is_input_fp8();
      
    impl_spec = make_gemm_v2_meta(use_fast_accum);
    auto meta = make_gemm_meta(dt_conf, arch, gemm_layout, impl, impl_spec);
    return meta;
  };

  RuntimeConfig
  get_rt_conf(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale) {
    CHECK_INPUT(input, this->input_dtype);
    CHECK_INPUT(weight, this->input_dtype);
    TORCH_CHECK(input.dim() == 2, "input shape is not 2");
    TORCH_CHECK(weight.dim() == 2, "weight dim is not 2");
    int32_t m = input.size(0);
    int32_t k = input.size(1);
    int32_t n = transpose_weight ? weight.size(1) : weight.size(0);

    if (bias.has_value()) {
      CHECK_INPUT(bias.value(), this->output_dtype);
      FLUX_CHECK_EQ(bias->dim(), 2);
      FLUX_CHECK_EQ(m, bias->size(0));
      FLUX_CHECK_EQ(n, bias->size(1));
    }
    if (output_buf.has_value()) {
      CHECK_INPUT(output_buf.value(), this->output_dtype);
      FLUX_CHECK_EQ(output_buf->dim(), 2);
      FLUX_CHECK_EQ(m, output_buf->size(0));
      FLUX_CHECK_EQ(n, output_buf->size(1));
    }
    int32_t wk = transpose_weight ? weight.size(0) : weight.size(1);
    FLUX_CHECK_EQ(wk, k) << "weight k-dim mismatch";
    return make_runtime_config(m, n, k);
  }

  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    auto rt_conf = get_rt_conf(input, weight, bias, output_buf, input_scale, weight_scale);
    int m = rt_conf.m();
    int n = rt_conf.n();
    int k = rt_conf.k();

    torch::Tensor output;
    if (output_buf.has_value()) {
      output = output_buf.value();
    } else {
      output = torch::empty({m, n}, weight.options().dtype(output_dtype));
    }

    auto meta = get_gemm_meta(/*has_bias=*/bias.has_value(), /*fast_accum=*/fast_accum);
    OpRegistry::OpPtr gemm_op;

    if (hparams.has_value()) {
      gemm_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      gemm_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    auto data_ptr_or = [](auto &&optional_tensor, void *other) -> void * {
      if (optional_tensor.has_value()) {
        return optional_tensor->data_ptr();
      } else {
        return other;
      }
    };
    
    // initialize mnk for streamk get_workspace_size
    const GemmOnlyArguments args{
      .m = m,
      .n = n,
      .k = k,
      .alpha = 1.0,
      .beta = bias.has_value() ? 1.0f : 0.0f,
      .input = input.data_ptr(),
      .weight = weight.data_ptr(),
      .bias = bias.has_value() ? bias->data_ptr() : nullptr,
      .output = output.data_ptr()};

    int64_t workspace_size = gemm_op->get_workspace_size(args);
    this->lazy_init_gemm_buffer(input, workspace_size);
    void *workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;
    gemm_op->run(args, workspace, stream);
    

    return output;
  }

  void
  lazy_init_gemm_buffer(torch::Tensor input, int64_t buffer_size) {
    if (buffer_size <= 0)
      return;
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!this->gemm_buffer.defined() || buffer_size > this->gemm_buffer.numel()) {
      this->gemm_buffer = torch::empty({buffer_size}, input.options().dtype(at::ScalarType::Byte));
    }
  }

 public:
  GemmOnlyImpl(
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight)
      : input_dtype(input_dtype),
        output_dtype(output_dtype),
        transpose_weight(transpose_weight) {}

  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum) {
    return forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(output_buf),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        c10::nullopt);
  }

  torch::Tensor
  profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    auto meta =
        unify_type(this->get_gemm_meta(/*has_bias=*/bias.has_value(), /*fast_accum=*/fast_accum));
    auto rt_conf = this->get_rt_conf(input, weight, bias, output_buf, input_scale, weight_scale);

    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    //

    int m = rt_conf.m();
    int n = rt_conf.n();
    int k = rt_conf.k();

    int cache_size = 100 * 1024 * 1024; // 100MB 
    int total_bytes = (m*k + k*n) * 4; //torch.finfo(dtype).bits // 8 # + M*N
    int problem_count = 1 + int((3 * cache_size) / total_bytes);

    std::vector<torch::Tensor> inputs;
    std::vector<torch::Tensor> weights;
    for (int i=0; i<problem_count; i++) {
      inputs.push_back(input.clone());
      weights.push_back(weight.clone());      
    }
    // for (int i=0; i<problem_count; i++) {
    //   printf("in: %p, ", inputs[i].data_ptr());
    // }
    // for (int i=0; i<problem_count; i++) {
    //   printf("w: %p, ", weights[i].data_ptr());
    // }
    printf("problem_count: %d, %d, %d.\n", problem_count, cache_size, total_bytes);

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          constexpr int warm_iters = 20;
          constexpr int iters = 100;
          float total_elapsed = 0;

          auto stream = c10::cuda::getCurrentCUDAStream();
          for (int iter = 0; iter < warm_iters + iters; ++iter) {
            int problem_idx = iter % problem_count;
            GpuTimer timer;
            timer.start(stream);
            auto output [[maybe_unused]] = this->forward_impl(
                inputs[problem_idx],
                weights[problem_idx],
                bias,
                output_buf,
                input_scale,
                weight_scale,
                output_scale,
                fast_accum,
                hparams);
            timer.stop();
            if (iter >= warm_iters) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          // printf("avg_elapsed: %f.\n", avg_elapsed);
          ctx->add(meta, rt_conf, hparams, avg_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_conf);

    return this->forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(output_buf),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        std::move(best_hparams));
  }
};

GemmOnly::GemmOnly(
    c10::ScalarType input_dtype,
    c10::ScalarType output_dtype,
    bool transpose_weight)
    : impl_(
          new GemmOnly::GemmOnlyImpl(input_dtype, output_dtype, transpose_weight)) {}

GemmOnly::~GemmOnly() { delete impl_; }

torch::Tensor
GemmOnly::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output_buf,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum) {
  FLUX_CHECK(impl_ != nullptr) << "GemmOnly is not initialized";
  return impl_->forward(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output_buf),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum);
}
torch::Tensor
GemmOnly::profiling(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output_buf,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(impl_ != nullptr) << "GemmOnly is not initialized";
  return impl_->profiling(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output_buf),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(output_scale),
      fast_accum,
      std::move(opt_ctx));
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
