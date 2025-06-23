
#include "ops/gemm_normal/gemm_normal.h"
#include "ths_op.h"


#ifdef ENABLE_GEMM_NORMAL
namespace ctlop {

namespace py = pybind11;
using GemmNormalCls = TorchClassWrapper<GemmNormal>;

static int _register_single_gemm_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_normal", [](py::module &m) {
    py::class_<GemmNormalCls>(m, "GemmNormal")
        .def(
            py::init([](torch::ScalarType input_dtype,
                        py::object py_output_dtype,
                        bool transpose_weight) {
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);
              return new GemmNormalCls(input_dtype, output_dtype, transpose_weight);
            }),
            py::arg("input_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("transpose_weight") = false)
        .def(
            "forward",
            &GemmNormalCls::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("output"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("tuning") = py::none(),
            py::arg("fast_accum") = false)
        .def(
            "grouped_forward",
            &GemmNormalCls::grouped_forward,
            py::arg("inputs"),
            py::arg("weights"),
            py::arg("outputs"),
            py::arg("inputs_scale") = py::none(),
            py::arg("weights_scale") = py::none(),
            py::arg("tuning") = py::none());

  });
  return 0;
}();
}  // namespace ctlop
#endif // ENABLE_GEMM_NORMAL