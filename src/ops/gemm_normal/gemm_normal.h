
#pragma once

#include <torch/all.h>
#include <cuda_runtime_api.h>

namespace xop {
class GemmNormal {
public:
  GemmNormal(
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight);
  ~GemmNormal();

  torch::Tensor forward(
      torch::Tensor lhs,
      torch::Tensor rhs,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> tuning,
      bool fast_accum);

  int grouped_forward(
      std::vector<torch::Tensor> inputs,
      std::vector<torch::Tensor> weights,
      std::vector<torch::Tensor> outputs,
      c10::optional<std::vector<torch::Tensor>> inputs_scale,
      c10::optional<std::vector<torch::Tensor>> weights_scale,
      c10::optional<torch::Tensor> tuning);

private:
  class GemmNormalImpl;
  GemmNormalImpl *impl_ = nullptr;
};

// Preprocess Funcs
void gemm_w4a16_sm90_reorder_weight(torch::Tensor weight);
void print_used_size_of_device_buffer();

// 
void gemm_dsl(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> output_buf, cudaStream_t stream);
}  // namespace xop
