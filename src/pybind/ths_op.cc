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

#include "ths_op.h"
#include "../comm_none/ths_op2.h"
// #include "ths_pybind.h"
#include <c10/cuda/CUDAStream.h>

namespace bytedance::flux::ths_op {

ThsOpsInitRegistry &
ThsOpsInitRegistry::instance() {
  static ThsOpsInitRegistry inst;
  return inst;
}

void
ThsOpsInitRegistry::register_one(std::string name, OpInitFunc &&func) {
  std::lock_guard<std::mutex> guard(register_mutex_);
  registry_.emplace(std::move(name), std::move(func));
}

void
ThsOpsInitRegistry::initialize_all(py::module &m) const {
  std::lock_guard<std::mutex> guard(register_mutex_);
  for (auto const &par : registry_) {
    auto [name, func] = par;
    func(m);
  }
}

void
init_profiling_context(py::module &m) {
  py::class_<ProfilingContext, c10::intrusive_ptr<ProfilingContext>>(m, "ProfilingContext")
      .def(py::init<std::string>())
      .def("get_code", &ProfilingContext::get_code)
      .def("get_all_prof_results", &ProfilingContext::get_all_prof_results)
      .def("get_latest_prof_result", &ProfilingContext::get_latest_prof_result)
      .def("get_latest_record", &ProfilingContext::get_latest_record)
      .def("get_all_records", &ProfilingContext::get_all_records);
}

void
init_tuning_record(py::module &m) {
  py::class_<PyTuningRecord, c10::intrusive_ptr<PyTuningRecord>>(m, "TuningRecord");
}

PYBIND11_MODULE(FLUX_TORCH_EXTENSION_NAME, m) {
  init_tuning_record(m);
  init_profiling_context(m);

  // Initialize ops in registry
  ThsOpsInitRegistry::instance().initialize_all(m);
}

}  // namespace bytedance::flux::ths_op
