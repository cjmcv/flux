//===- runtime_config.h ------------------------------------------- C++ ---===//
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
#include "flux/flux.h"

namespace bytedance::flux {

using UnifiedCommRuntimeConfig =
    std::variant<None>; // , ReduceScatterRuntimeConfig, AllGatherRuntimeConfig

// Runtime config used for Dispacher of ops.
template <class... Ts>
struct RuntimeConfigTpl : FluxNamedTupleBase<RuntimeConfigTpl, Ts...> {
  using Base = FluxNamedTupleBase<RuntimeConfigTpl, Ts...>;
  static constexpr char const *Name = "RuntimeConfig";
  static constexpr char const *LowerName = "runtime_config";
  static constexpr std::array<char const *, 4> Fields = {"m", "n", "k", "comm_spec"};
  FLUX_NAMED_TUPLE_DEFINE_FIELD(m, 0)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(n, 1)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(k, 2)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(comm_spec, 3)

  constexpr RuntimeConfigTpl(cute::tuple<Ts...> const &tup) : Base(tup) {}
};

using RuntimeConfig = RuntimeConfigTpl<int, int, int, UnifiedCommRuntimeConfig>;

inline RuntimeConfig
make_runtime_config(
    int m = 0, int n = 0, int k = 0, UnifiedCommRuntimeConfig const &comm_rt_conf = None{}) {
  return {cute::make_tuple(m, n, k, comm_rt_conf)};
}

}  // namespace bytedance::flux
