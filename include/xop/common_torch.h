
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

}  // namespace xop
