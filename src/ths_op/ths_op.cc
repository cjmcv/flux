//===- ths_op.cc -------------------------------------------------- C++ ---===//
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

#include "flux/ths_op/ths_op.h"
#include "flux/cuda/cuda_common.h"
// #include "flux/cuda/moe_utils.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include <ATen/Context.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <utility>

namespace bytedance::flux::ths_op {

DataTypeEnum
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

at::ScalarType
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

PyTuningRecord::PyTuningRecord(
    UnifiedGemmMeta meta, RuntimeConfig rt_conf, UnifiedGemmHParams best_hparams)
    : meta(std::move(meta)), rt_conf(std::move(rt_conf)), best_hparams(std::move(best_hparams)) {}


ProfilingContext::ProfilingContext(std::string name) : codegen(std::move(name)), counter(0) {}

TuningConfigGenerator const &
ProfilingContext::get_codegen() const {
  return this->codegen;
}

std::string
ProfilingContext::get_code() const {
  return this->get_codegen().str();
}

std::string
ProfilingContext::to_string_topk(TopHParams const &top_hparams, int topk) const {
  std::ostringstream ss;
  int top_idx = 0;
  for (auto iter = top_hparams.begin(); top_idx < topk && iter != top_hparams.end(); ++iter) {
    ss << " * TopK=" << (++top_idx);
    ss << std::setprecision(3) << " (" << iter->first.first << " ms): " << iter->second;
    if (top_idx < topk) {
      ss << "\n";
    }
  }
  return std::move(ss).str();
}

std::vector<std::string>
ProfilingContext::get_all_prof_results() const {
  std::vector<std::string> ret;
  for (auto const &par : prof_results) {
    std::ostringstream ss;
    auto [meta, rt_conf] = par.first;
    const auto &top_hparams = par.second;
    ss << meta << "\n" << rt_conf << "\n";
    ss << to_string_topk(top_hparams, kReturnTopK);
    ret.emplace_back(std::move(ss).str());
  }
  return ret;
}

std::vector<PyTuningRecord>
ProfilingContext::get_all_records() const {
  std::vector<PyTuningRecord> rets;
  for (auto const &par : prof_results) {
    std::ostringstream ss;
    auto [meta, rt_conf] = par.first;
    const auto &top_hparams = par.second;
    rets.emplace_back(meta, rt_conf, top_hparams.begin()->second);
  }
  return rets;
}

std::string
ProfilingContext::get_latest_prof_result() const {
  FLUX_CHECK(latest_key_ptr != nullptr) << "no latest prof results found";
  auto key = *latest_key_ptr;
  auto iter = prof_results.find(key);
  FLUX_CHECK(iter != prof_results.end())
      << "key not found: (" << key.first << ", " << key.second << ")";
  std::ostringstream ss;
  ss << key.first << "\n" << key.second << "\n";
  ss << to_string_topk(iter->second, kReturnTopK);
  return std::move(ss).str();
}

PyTuningRecord
ProfilingContext::get_latest_record() const {
  FLUX_CHECK(latest_key_ptr != nullptr) << "no latest prof results found";
  auto key = *latest_key_ptr;
  auto iter = prof_results.find(key);
  FLUX_CHECK(iter != prof_results.end())
      << "key not found: (" << key.first << ", " << key.second << ")";
  auto [meta, rt_conf] = key;
  auto const &top_hparams = iter->second;
  return PyTuningRecord(meta, rt_conf, top_hparams.begin()->second);
}

void
ProfilingContext::add(
    UnifiedGemmMeta const &meta,
    RuntimeConfig const &rt_conf,
    UnifiedGemmHParams hparams,
    float elapsed_ms) {
  auto key = std::make_pair(meta, rt_conf);
  if (prof_results.count(key) == 0) {
    prof_results[key] = TopHParams();
  }
  prof_results[key].emplace(std::make_pair(elapsed_ms, counter++), std::move(hparams));
}

UnifiedGemmHParams
ProfilingContext::record_best(UnifiedGemmMeta const &meta, RuntimeConfig const &rt_conf) {
  auto key = std::make_pair(meta, rt_conf);
  latest_key_ptr = std::make_unique<decltype(key)>(key);

  auto iter = prof_results.find(key);
  FLUX_CHECK(iter != prof_results.end()) << "no prof results found for" << meta << ", " << rt_conf;
  auto const &top_hparams = iter->second;
  auto best_hparams = top_hparams.begin()->second;
  codegen.add(meta, rt_conf, best_hparams);
  return best_hparams;
}

}  // namespace bytedance::flux::ths_op
