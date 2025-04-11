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
// #include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_pybind.h"
// #include "coll/ths_op/all_gather_types.h"
// #include "coll/ths_op/reduce_scatter_op.h"
#include <c10/cuda/CUDAStream.h>
// #include "flux/cuda/moe_utils.h"

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

// torch::Tensor
// calc_scatter_index_impl(
//     const torch::Tensor choosed_experts,  // topk * ntokens
//     const torch::Tensor splits            // of expert_num
// ) {
//   CHECK_INPUT(choosed_experts, at::ScalarType::Int);
//   CHECK_INPUT(splits, at::ScalarType::Int);
//   torch::Tensor scatter_index = at::empty_like(choosed_experts);
//   auto stream = at::cuda::getCurrentCUDAStream();
//   calc_scatter_index(
//       choosed_experts.data_ptr<int>(),
//       splits.data_ptr<int>(),
//       scatter_index.data_ptr<int>(),
//       choosed_experts.numel(),
//       splits.numel(),
//       stream);
//   return scatter_index;
// }

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

void
init_dist_env_tp(py::module &m) {
  py::class_<DistEnvTP>(m, "DistEnvTP")
      .def(
          py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group, int32_t nnodes) {
            return new DistEnvTP(tp_group, nnodes);
          }),
          py::arg("tp_group"),
          py::arg("nnodes"))
      .def("__repr__", &DistEnvTP::toString);
}

void
init_dist_env_tp_with_ep(py::module &m) {
  py::class_<DistEnvTPWithEP>(m, "DistEnvTPWithEP")
      .def(
          py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                      int32_t nnodes,
                      c10::intrusive_ptr<c10d::ProcessGroup> ep_group) {
            return new DistEnvTPWithEP(tp_group, nnodes, ep_group);
          }),
          py::arg("tp_group"),
          py::arg("nnodes"),
          py::arg("ep_group") = py::none())
      .def("__repr__", &DistEnvTPWithEP::toString);
}

void
init_moe_arguments(py::module &m) {
  py::class_<MoeArguments, c10::intrusive_ptr<MoeArguments>>(m, "MoeArguments")
      .def(
          py::init([](int32_t max_ntokens,
                      int32_t hidden,
                      int32_t ffn_hidden,
                      int32_t nexperts,
                      int32_t topk,
                      torch::ScalarType input_dtype,
                      py::object py_output_dtype) {
            auto output_dtype = py_output_dtype.is(py::none())
                                    ? input_dtype
                                    : torch::python::detail::py_object_to_dtype(py_output_dtype);
            return new MoeArguments(
                max_ntokens, hidden, ffn_hidden, nexperts, topk, input_dtype, output_dtype);
          }),
          py::arg("max_ntokens"),
          py::arg("hidden"),
          py::arg("ffn_hidden"),
          py::arg("nexperts"),
          py::arg("topk"),
          py::arg("input_dtype"),
          py::arg("output_dtype") = py::none());
}


PYBIND11_MODULE(FLUX_TORCH_EXTENSION_NAME, m) {
  m.def("load_tuning_record", &load_tuning_record);

  init_tuning_record(m);
  init_profiling_context(m);

  // Initialize ops in registry
  ThsOpsInitRegistry::instance().initialize_all(m);
}

}  // namespace bytedance::flux::ths_op
