
#pragma once

#include <torch/all.h>
  
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
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> tuning,
      bool fast_accum);

private:
  class GemmNormalImpl;
  GemmNormalImpl *impl_ = nullptr;
};

}  // namespace xop
