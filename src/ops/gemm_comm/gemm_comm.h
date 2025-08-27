
#pragma once

#include <torch/all.h>
  
namespace xop {
class GemmComm {
public:
  GemmComm(
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight);
  ~GemmComm();

  int forward(
      torch::Tensor lhs,
      torch::Tensor rhs,
      torch::Tensor output,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> tuning,
      bool fast_accum,
      bool registered,
      int64_t fa, 
      int64_t reg_buffer, 
      int64_t reg_buffer_sz_bytes);

  int grouped_forward(
      std::vector<torch::Tensor> inputs,
      std::vector<torch::Tensor> weights,
      std::vector<torch::Tensor> outputs,
      c10::optional<std::vector<torch::Tensor>> inputs_scale,
      c10::optional<std::vector<torch::Tensor>> weights_scale,
      c10::optional<torch::Tensor> tuning);

private:
  class GemmCommImpl;
  GemmCommImpl *impl_ = nullptr;
};

}  // namespace xop
