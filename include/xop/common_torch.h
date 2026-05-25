
#pragma once
#include "c10/util/Optional.h"
#include "xop/xop.h"
#include <torch/torch.h>
#include <ATen/core/ivalue.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace xop {

inline int8_t from_torch_dtype(at::ScalarType torch_dtype) {
  switch (torch_dtype) {
    case at::ScalarType::Float:            return (int)UnifiedMetaEnum::FP32;
    case at::ScalarType::Int:              return (int)UnifiedMetaEnum::S32;
    case at::ScalarType::Char:             return (int)UnifiedMetaEnum::S8;
    case at::ScalarType::Half:             return (int)UnifiedMetaEnum::FP16;
    case at::ScalarType::BFloat16:         return (int)UnifiedMetaEnum::BF16;
    case at::ScalarType::Float8_e4m3fn:    return (int)UnifiedMetaEnum::E4M3;
    case at::ScalarType::Float8_e5m2:      return (int)UnifiedMetaEnum::E5M2;
  }
  throw std::runtime_error(std::string("unsupported torch_dtype:") + at::toString(torch_dtype));
}

inline at::ScalarType to_torch_dtype(UnifiedMetaEnum dtype) {
  switch (dtype) {
    case UnifiedMetaEnum::FP32:            return at::ScalarType::Float;
    case UnifiedMetaEnum::S32:             return at::ScalarType::Int;
    case UnifiedMetaEnum::S8:              return at::ScalarType::Char;
    case UnifiedMetaEnum::FP16:            return at::ScalarType::Half;
    case UnifiedMetaEnum::BF16:            return at::ScalarType::BFloat16;
    case UnifiedMetaEnum::E4M3:            return at::ScalarType::Float8_e4m3fn;
    case UnifiedMetaEnum::E5M2:            return at::ScalarType::Float8_e5m2;
  }
  throw std::runtime_error(std::string("unsupported dtype: ") + std::to_string((int32_t)dtype));
}

struct TorchDefaultConfig {

  static std::vector<int16_t> 
  MakeDefaultMeta(UnifiedMetaEnum arch, c10::ScalarType input_dtype, c10::ScalarType weight_dtype, c10::ScalarType output_dtype, bool fast_accum, bool is_transpose_weight=false, bool is_group=false) {
  
    std::vector<int16_t> meta;
    meta.resize(8);
    meta[kMetaId] = -1;                                  // id
    // (GemmNormal / GemmNormalSimt / GemmGrouped / GemmBlockScaleFp8 / GemmGroupedBlockScaleFp8 / GemmLt)
    // meta[kMetaSchema] = (int16_t)UnifiedMetaEnum::GemmNormal; // schema type 

    meta[kMetaTypeA] = from_torch_dtype(input_dtype);  // type A
    meta[kMetaTypeB] = from_torch_dtype(weight_dtype);  // type B
    meta[kMetaTypeCD] = from_torch_dtype(output_dtype); // type C/D

    if (fast_accum)
      meta[kMetaTypeAcc] = (int16_t)UnifiedMetaEnum::FP16;        // type acc
    else
      meta[kMetaTypeAcc] = (int16_t)UnifiedMetaEnum::FP32;

    if (is_transpose_weight)                           // layout
      meta[kMetaLayout] = (int16_t)UnifiedMetaEnum::RRR; 
    else
      meta[kMetaLayout] = (int16_t)UnifiedMetaEnum::RCR;

    meta[kMetaArch] = (int16_t)arch;               // arch

    // adjust
    if (from_torch_dtype(input_dtype) == (int)UnifiedMetaEnum::E4M3) {
      if (is_group) {
        meta[kMetaSchema] = (int16_t)UnifiedMetaEnum::GemmGroupedBlockScaleFp8;
      }
      else {
        meta[kMetaSchema] = (int16_t)UnifiedMetaEnum::GemmBlockScaleFp8;        
      }
      if (arch != UnifiedMetaEnum::Sm90 && arch != UnifiedMetaEnum::Sm89) {
        printf("fp8 kernel is only supported on GPUs with the sm_89 or sm_90 architecture.");
      }
    }
    else {
      meta[kMetaSchema] = (int16_t)UnifiedMetaEnum::GemmNormal;
      if (arch == UnifiedMetaEnum::Sm89) {
        meta[kMetaArch] = (int16_t)UnifiedMetaEnum::Sm80;       
      }
    }
    return meta;
  }

  static std::unique_ptr<RtArguments> GetBaseRtConf(
      torch::Tensor input,
      torch::Tensor weight,
      torch::Tensor output,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool is_transpose_weight,
      UnifiedMetaEnum *default_schema) {
    XOP_CHECK_INPUT(input, input_dtype);
    // XOP_CHECK_INPUT(weight, input_dtype);
    TORCH_CHECK(input.dim() == 2, "input shape is not 2");
    TORCH_CHECK(weight.dim() == 2, "weight dim is not 2");
    int32_t m = input.size(0);
    int32_t k = input.size(1);
    int32_t n = output.size(1); // is_transpose_weight ? weight.size(1) : weight.size(0); // true是RRR，正常使用是false，对应linear层的RCR

    std::unique_ptr<RtArguments> rt_args;
    if (weight_scale.has_value()) {
      rt_args = std::make_unique<RtBlockScaleArguments>();
      ((RtBlockScaleArguments *)rt_args.get())->ptr_blockscale_B = weight_scale.value().data_ptr();
      if (input_scale.has_value()) {
        ((RtBlockScaleArguments *)rt_args.get())->ptr_blockscale_A = input_scale.value().data_ptr();
        *default_schema = UnifiedMetaEnum::GemmBlockScaleFp8;
      }
      else {
        ((RtBlockScaleArguments *)rt_args.get())->ptr_blockscale_A = nullptr;
        // w4a16, 2*w4=1*int8, weight_int8[N,K/2], weight_scale[group_num, N] (weight_scale has been transposed in w4a16)
        // group_size = K / group_num.
        rt_args->g = weight.size(1) * 2 / weight_scale.value().size(0); 
        *default_schema = UnifiedMetaEnum::GemmW4A16;
      }
    }
    else {
      rt_args = std::make_unique<RtArgumentsV2>();
      *default_schema = UnifiedMetaEnum::GemmNormal;
    }

    rt_args->C_s = -1;
    if (bias.has_value()) {
      XOP_CHECK_INPUT(bias.value(), output_dtype);
      if (bias->dim() == 2) {
        XOP_CHECK_EQ(n, bias->size(1));
        XOP_CHECK((bias->size(0) == m) || (bias->size(0) == 1));
        if (bias->size(0) == 1) {
          rt_args->C_s = 0;
        }
      }
      else {
        XOP_CHECK_EQ(n, bias->size(0));
        rt_args->C_s = 0;
      }
    }
    ///////////////////////////
    // if (m < 32) {
    //   padded_input_ = torch::zeros({32, k}, input.options());
    //   padded_input_.slice(0, 0, m).slice(1, 0, k).copy_(input);
    //   m = 32;
    //   // std::cout << "Tensor padded_input_:\n" << padded_input_ << std::endl;
    // }
    ///////////////////////////
    rt_args->m = m;
    rt_args->n = n;
    rt_args->k = k;
    rt_args->l = 1;
    if (rt_args->g == 0)
      rt_args->g = 1;
    // rt_args->ptr_A = padded_input_.data_ptr();
    rt_args->ptr_A = input.data_ptr();
    rt_args->ptr_B = weight.data_ptr();
    rt_args->ptr_C = bias.has_value() ? bias.value().data_ptr() : nullptr;
    rt_args->ptr_D = output.data_ptr();
    rt_args->alpha = 1.0f;
    rt_args->beta = 0.0f;


    // int32_t k_remainder = k % 16;
    // if (k_remainder != 0) {
    //   // padded_input_ = torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions({0, 16-k_remainder, 0, 0}).mode(torch::kConstant).value(0.0)); // [pad_left, pad_right, pad_top, pad_bottom]
    //   int new_k = k+16-k_remainder;
    //   padded_input_ = torch::zeros({m, new_k}, input.options());
    //   padded_input_.slice(0, 0, m).slice(1, 0, k).copy_(input);
    // }
    return rt_args;
  }

  static RunModeEnum GetRunMode(c10::optional<torch::Tensor> tuning) {
    if (tuning.has_value()) {
      int16_t *tdata = (int16_t *)tuning.value().data_ptr();
      if (tdata[0] == 1)
        return kRunWithTuning;
      else if (tdata[0] == 2) {
        return kRunWithHparam; // max_m = tdata[1];
      }
    }
    return kRunWithNormal;
  }
};

}  // namespace xop
