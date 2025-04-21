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

// Wraps c++ types in class holder, in order to communicate with python
struct TuningRecord : public torch::CustomClassHolder {
  UnifiedGemmMeta meta;
  RuntimeConfig rt_conf;
  UnifiedGemmHParams best_hparams;

  TuningRecord(
    UnifiedGemmMeta meta, RuntimeConfig rt_conf, UnifiedGemmHParams best_hparams)
    : meta(std::move(meta)), rt_conf(std::move(rt_conf)), best_hparams(std::move(best_hparams)) {}
};


class ProfilingContext : public torch::CustomClassHolder {
 private:
  TuningConfigGenerator codegen;

  using TopHParams = std::map<std::pair<float, int>, UnifiedGemmHParams>;
  std::map<std::pair<UnifiedGemmMeta, RuntimeConfig>, TopHParams> prof_results;

  int counter;
  std::unique_ptr<std::pair<UnifiedGemmMeta, RuntimeConfig>> latest_key_ptr;

  std::string to_string_topk(TopHParams const &top_hparams, int topk) const {
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
 public:
  static constexpr int kReturnTopK = 5;

  ProfilingContext(std::string name) : codegen(std::move(name)), counter(0) {}

  TuningConfigGenerator const &get_codegen() const {
    return this->codegen;
  }
  // get generated code
  std::string get_code() const {
    return this->get_codegen().str();
  }

  // get all prof results as a vector, each element is the prof result of
  // a (GemmMeta,RuntimeConf) pair.
  std::vector<std::string> get_all_prof_results() const {
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

  std::vector<TuningRecord> get_all_records() const {
    std::vector<TuningRecord> rets;
    for (auto const &par : prof_results) {
      std::ostringstream ss;
      auto [meta, rt_conf] = par.first;
      const auto &top_hparams = par.second;
      rets.emplace_back(meta, rt_conf, top_hparams.begin()->second);
    }
    return rets;
  }

  // the prof result of the latest (GemmMeta, RuntimeConf) pair that has
  // finished profiling (i.e. record_best() has been called)
  std::string get_latest_prof_result() const {
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

  TuningRecord get_latest_record() const {
    FLUX_CHECK(latest_key_ptr != nullptr) << "no latest prof results found";
    auto key = *latest_key_ptr;
    auto iter = prof_results.find(key);
    FLUX_CHECK(iter != prof_results.end())
        << "key not found: (" << key.first << ", " << key.second << ")";
    auto [meta, rt_conf] = key;
    auto const &top_hparams = iter->second;
    return TuningRecord(meta, rt_conf, top_hparams.begin()->second);
  }

  // add a single record
  void add(
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

  // called after all records of (meta,rt_conf) have been added
  // this function will: 1. append the best config record of (meta,rt_conf) to codegen;
  // 2. update the latest_key_ptr to be (meta,rt_conf)
  UnifiedGemmHParams record_best(UnifiedGemmMeta const &meta, RuntimeConfig const &rt_conf) {
    auto key = std::make_pair(meta, rt_conf);
    latest_key_ptr = std::make_unique<decltype(key)>(key);
  
    auto iter = prof_results.find(key);
    FLUX_CHECK(iter != prof_results.end()) << "no prof results found for" << meta << ", " << rt_conf;
    auto const &top_hparams = iter->second;
    auto best_hparams = top_hparams.begin()->second;
    codegen.add(meta, rt_conf, best_hparams);
    return best_hparams;
  }
};

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
