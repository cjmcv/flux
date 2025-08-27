
#include "ops/gemm_comm/gemm_comm.h"
#include "ths_op.h"


#ifdef ENABLE_GEMM_COMM
namespace xop {

namespace py = pybind11;
using GemmCommCls = TorchClassWrapper<GemmComm>;

static int _register_single_gemm_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_comm", [](py::module &m) {
    py::class_<GemmCommCls>(m, "GemmComm")
        .def(
            py::init([](torch::ScalarType input_dtype,
                        py::object py_output_dtype,
                        bool transpose_weight) {
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);
              return new GemmCommCls(input_dtype, output_dtype, transpose_weight);
            }),
            py::arg("input_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("transpose_weight") = false)
        .def(
            "forward",
            &GemmCommCls::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("output"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("tuning") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("registered") = false,
            py::arg("fa"),
            py::arg("reg_buffer"),
            py::arg("reg_buffer_sz_bytes"))
        .def(
            "grouped_forward",
            &GemmCommCls::grouped_forward,
            py::arg("inputs"),
            py::arg("weights"),
            py::arg("outputs"),
            py::arg("inputs_scale") = py::none(),
            py::arg("weights_scale") = py::none(),
            py::arg("tuning") = py::none());

  });
  return 0;
}();
}  // namespace xop
#endif // ENABLE_GEMM_COMM