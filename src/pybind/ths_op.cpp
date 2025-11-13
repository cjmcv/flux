
#include "ths_op.h"

#define ENABLE_GEMM_NORMAL 1
#define ENABLE_MARLIN_KERNEL 1
// #define ENABLE_GEMM_COMM 1
// #define ENABLE_ALLREDUCE_CUSTOM 1
// #define ENABLE_FLASH_ATTEN 1

using namespace xop;
namespace py = pybind11;

PYBIND11_MODULE(XOP_TORCH_EXTENSION_NAME, m) {
  // Initialize ops in registry
  ThsOpsInitRegistry::instance().initialize_all(m);
}

#ifdef ENABLE_GEMM_NORMAL
#include "ops/gemm_normal/gemm_normal.h"

using GemmNormalCls = TorchClassWrapper<GemmNormal>;

static int _register_gemm_normal_ops [[maybe_unused]] = []() {
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

    m.def("gemm_w4a16_sm90_reorder_weight",
        &gemm_w4a16_sm90_reorder_weight,
        py::arg("weight"));
        
    m.def("print_used_size_of_device_buffer",
        &print_used_size_of_device_buffer);
  });



  return 0;
}();
#endif // ENABLE_GEMM_NORMAL

#ifdef ENABLE_GEMM_COMM
#include "ops/gemm_comm/gemm_comm.h"

using GemmCommCls = TorchClassWrapper<GemmComm>;

static int _register_gemm_comm_ops [[maybe_unused]] = []() {
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
#endif // ENABLE_GEMM_COMM

#ifdef ENABLE_ALLREDUCE_CUSTOM
#include "ops/allreduce_normal/custom_all_reduce.h"

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
#endif // ENABLE_ALLREDUCE_CUSTOM

#ifdef ENABLE_MARLIN_KERNEL
#include "ops/marlin/marlin_cuda.h"

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
#endif // ENABLE_MARLIN_KERNEL

#ifdef ENABLE_FLASH_ATTEN
#include "ops/flash_attn/flash_api.h"

static int _register_flash_attn_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("flash_attn", [](py::module &m) {
    m.def("mha_fwd",
      &mha_fwd,
      py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("k_new") = py::none(),
      py::arg("v_new") = py::none(),
      py::arg("q_v") = py::none(),
      py::arg("out") = py::none(),
      py::arg("cu_seqlens_q") = py::none(),
      py::arg("cu_seqlens_k") = py::none(),
      py::arg("cu_seqlens_k_new") = py::none(),
      py::arg("seqused_q") = py::none(),
      py::arg("seqused_k") = py::none(),
      py::arg("max_seqlen_q") = py::none(),
      py::arg("max_seqlen_k") = py::none(),
      py::arg("page_table") = py::none(),
      py::arg("kv_batch_idx") = py::none(),
      py::arg("leftpad_k") = py::none(),
      py::arg("rotary_cos") = py::none(),
      py::arg("rotary_sin") = py::none(),
      py::arg("seqlens_rotary") = py::none(),
      py::arg("q_descale") = py::none(),
      py::arg("k_descale") = py::none(),
      py::arg("v_descale") = py::none(),
      py::arg("softmax_scale"),
      py::arg("is_causal"),
      py::arg("window_size_left"),
      py::arg("window_size_right"),
      py::arg("softcap"),
      py::arg("is_rotary_interleaved"),
      py::arg("scheduler_metadata") = py::none(),
      py::arg("num_splits"),
      py::arg("pack_gqa") = py::none(),
      py::arg("sm_margin"));
      
    // m.def("mha_fwd",
    //       &mha_fwd,
    //       py::arg("q"),
    //       py::arg("k"),
    //       py::arg("v"),
    //       py::arg("k_new_") = py::none(),
    //       py::arg("v_new_") = py::none(),
    //       py::arg("q_v_") = py::none(),
    //       py::arg("out_") = py::none(),
    //       py::arg("cu_seqlens_q_") = py::none(),
    //       py::arg("cu_seqlens_k_") = py::none(),
    //       py::arg("cu_seqlens_k_new_") = py::none(),
    //       py::arg("seqused_q_") = py::none(),
    //       py::arg("seqused_k_") = py::none(),
    //       py::arg("max_seqlen_q_") = py::none(),
    //       py::arg("max_seqlen_k_") = py::none(),
    //       py::arg("page_table_") = py::none(),
    //       py::arg("kv_batch_idx_") = py::none(),
    //       py::arg("leftpad_k_") = py::none(),
    //       py::arg("rotary_cos_") = py::none(),
    //       py::arg("rotary_sin_") = py::none(),
    //       py::arg("seqlens_rotary_") = py::none(),
    //       py::arg("q_descale_") = py::none(),
    //       py::arg("k_descale_") = py::none(),
    //       py::arg("v_descale_") = py::none(),
    //       py::arg("softmax_scale"),
    //       py::arg("is_causal"),
    //       py::arg("window_size_left"),
    //       py::arg("window_size_right"),
    //       py::arg("softcap"),
    //       py::arg("is_rotary_interleaved"),
    //       py::arg("scheduler_metadata_") = py::none(),
    //       py::arg("num_splits"),
    //       py::arg("pack_gqa_") = py::none(),
    //       py::arg("sm_margin")
    //   );
  });
  return 0;
}();
#endif // ENABLE_FLASH_ATTEN

// } // namespace xop