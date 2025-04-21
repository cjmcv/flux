//===- gemm_only.cc ----------------------------------------------- C++ ---===//
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

#include "comm_none/ths_op/gemm_only.h"
#include "ths_op.h"

namespace bytedance::flux::ths_op {

namespace py = pybind11;
using GemmOnlyCls = TorchClassWrapper<GemmOnly>;

static int _register_gemm_only_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_only", [](py::module &m) {
    py::class_<GemmOnlyCls>(m, "GemmOnly")
        .def(
            py::init([](torch::ScalarType input_dtype,
                        py::object py_output_dtype,
                        bool transpose_weight) {
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);
              return new GemmOnlyCls(input_dtype, output_dtype, transpose_weight);
            }),
            py::arg("input_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("transpose_weight") = false)
        .def(
            "forward",
            &GemmOnlyCls::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output_buf") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false)
        .def(
            "profiling",
            &GemmOnlyCls::profiling,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output_buf") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("prof_ctx") = nullptr);
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op
