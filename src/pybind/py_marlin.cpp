
#include "ops/marlin/marlin_cuda.h"
#include "ths_op.h"

#ifdef ENABLE_MARLIN_KERNEL
namespace xop {

namespace py = pybind11;

static int _register_allreduce_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("marlin", [](py::module &m) {
    m.def("helloABCM",
          &helloABCM,
          py::arg("a"));

    m.def("marlin_fp16xint4_matmul",
          &marlin_fp16xint4_matmul,
          py::arg("A"),
          py::arg("B"),
          py::arg("C"),
          py::arg("s"),
          py::arg("workspace"),
          py::arg("thread_k") = -1,
          py::arg("thread_n") = -1,
          py::arg("sms") = -1,
          py::arg("max_par")= 8);
  });
  return 0;
}();
}  // namespace xop
#endif // ENABLE_MARLIN_KERNEL