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
#include <torch/csrc/utils/pybind.h>

#define FLUX_TORCH_EXTENSION_NAME flux_ths_pybind

namespace bytedance {
namespace flux {
namespace ths_op {

// DataTypeEnum from_torch_dtype(at::ScalarType torch_dtype);
// at::ScalarType to_torch_dtype(DataTypeEnum dtype);

// Registry of functions that register
// functions into module
class ThsOpsInitRegistry {
 public:
  using OpInitFunc = std::function<void(py::module &)>;
  static ThsOpsInitRegistry &instance();
  void register_one(std::string name, OpInitFunc &&func);
  void initialize_all(py::module &m) const;

 private:
  std::map<std::string, OpInitFunc> registry_;
  mutable std::mutex register_mutex_;

  ThsOpsInitRegistry() {}
  ThsOpsInitRegistry(const ThsOpsInitRegistry &) = delete;
  ThsOpsInitRegistry &operator=(const ThsOpsInitRegistry &) = delete;
};


}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
