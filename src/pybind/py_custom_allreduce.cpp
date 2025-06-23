
#include "ops/allreduce_normal/custom_all_reduce.h"
#include "ths_op.h"

#ifdef ENABLE_ALLREDUCE_CUSTOM
namespace ctlop {

namespace py = pybind11;

static int _register_allreduce_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("custom_allreduce", [](py::module &m) {
    m.def("helloABC",
          &helloABC,
          py::arg("a"));

    m.def("init_custom_ar",
          &init_custom_ar,
          py::arg("fake_ipc_ptrs"),
          py::arg("rank_data"),
          py::arg("rank"),
          py::arg("full_nvlink"));
    m.def("all_reduce",
          &all_reduce,
          py::arg("_fa"),
          py::arg("inp"),
          py::arg("out"),
          py::arg("_reg_buffer"),
          py::arg("reg_buffer_sz_bytes"));

    m.def("dispose", &dispose, py::arg("_fa"));
    m.def("meta_size", &meta_size);
    m.def("register_buffer",
          &register_buffer,
          py::arg("_fa"),
          py::arg("fake_ipc_ptrs"));
    m.def("get_graph_buffer_ipc_meta",
          &get_graph_buffer_ipc_meta,
          py::arg("_fa"));
    m.def("register_graph_buffers",
          &register_graph_buffers,
          py::arg("_fa"),
          py::arg("handles"),
          py::arg("offsets"));
  });
  return 0;
}();
}  // namespace ctlop
#endif // ENABLE_ALLREDUCE_CUSTOM