//===- ths_op.h --------------------------------------------------- C++ ---===//
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
#include "c10/util/Optional.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include <torch/torch.h>
#include <ATen/core/ivalue.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace bytedance {
namespace flux {
namespace ths_op {

inline DataTypeEnum
from_torch_dtype(at::ScalarType torch_dtype) {
  switch (torch_dtype) {
    case at::ScalarType::Float: {
      return _FP32{};
    }; break;
    case at::ScalarType::Int: {
      return _S32{};
    }; break;
    case at::ScalarType::Char: {
      return _S8{};
    }; break;
    case at::ScalarType::Half: {
      return _FP16{};
    }; break;
    case at::ScalarType::BFloat16: {
      return _BF16{};
    }; break;
    case at::ScalarType::Float8_e4m3fn: {
      return _E4M3{};
    }; break;
    case at::ScalarType::Float8_e5m2: {
      return _E5M2{};
    }; break;
    default:
      throw std::runtime_error(
          std::string("unsupported torch_dtype:") + at::toString(torch_dtype));
  }
  return DataTypeEnum{};
}

inline at::ScalarType
to_torch_dtype(DataTypeEnum dtype) {
  switch (dtype) {
    case _FP32{}: {
      return at::ScalarType::Float;
    }; break;
    case _S32{}: {
      return at::ScalarType::Int;
    }; break;
    case _S8{}: {
      return at::ScalarType::Char;
    }; break;
    case _FP16{}: {
      return at::ScalarType::Half;
    }; break;
    case _BF16{}: {
      return at::ScalarType::BFloat16;
    }; break;
    case _E4M3{}: {
      return at::ScalarType::Float8_e4m3fn;
    }; break;
    case _E5M2{}: {
      return at::ScalarType::Float8_e5m2;
    }; break;
    default:
      throw std::runtime_error(
          std::string("unsupported dtype: ") + std::string(enum_to_string(dtype)));
  }
  return at::ScalarType::Undefined;
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
