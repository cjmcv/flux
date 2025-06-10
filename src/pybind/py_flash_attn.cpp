
#include "ops/allreduce_normal/flash_api.h"
#include "ths_op.h"

using namespace ctlop;

namespace ctlop {

namespace py = pybind11;

static int _register_flash_attn_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("flash_attn", [](py::module &m) {
    m.def("mha_fwd",
          &mha_fwd,
          py::arg("q") = py::none(),
          py::arg("k") = py::none(),
          py::arg("v") = py::none(),
          py::arg("k_new_") = py::none(),
          py::arg("v_new_") = py::none(),
          py::arg("q_v_") = py::none(),
          py::arg("out_") = py::none(),
          py::arg("cu_seqlens_q_") = py::none(),
          py::arg("cu_seqlens_k_") = py::none(),
          py::arg("cu_seqlens_k_new_") = py::none(),
          py::arg("seqused_q_") = py::none(),
          py::arg("seqused_k_") = py::none(),
          py::arg("max_seqlen_q_") = py::none(),
          py::arg("max_seqlen_k_") = py::none(),
          py::arg("page_table_") = py::none(),
          py::arg("kv_batch_idx_") = py::none(),
          py::arg("leftpad_k_") = py::none(),
          py::arg("rotary_cos_") = py::none(),
          py::arg("rotary_sin_") = py::none(),
          py::arg("seqlens_rotary_") = py::none(),
          py::arg("q_descale_") = py::none(),
          py::arg("k_descale_") = py::none(),
          py::arg("v_descale_") = py::none(),
          py::arg("softmax_scale"),
          py::arg("is_causal"),
          py::arg("window_size_left"),
          py::arg("window_size_right"),
          py::arg("softcap"),
          py::arg("is_rotary_interleaved"),
          py::arg("scheduler_metadata_") = py::none(),
          py::arg("num_splits"),
          py::arg("pack_gqa_") = py::none(),
          py::arg("sm_margin")
      );
  });
  return 0;
}();
}  // namespace ctlop