
#pragma once

#include <torch/all.h>
  
namespace ctlop {
class GemmNormal {
public:
  GemmNormal(
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight);
  ~GemmNormal();

  int forward(
      torch::Tensor lhs,
      torch::Tensor rhs,
      torch::Tensor output,
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

}  // namespace ctlop
