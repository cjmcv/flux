#include "ctlop/ops_impl/gemm_normal/gemm_v2_impl.h"
#include "ctlop/ops_impl/gemm_normal/gemm_v2_simt_impl.h"

namespace ctlop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_normal_sm89 = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({1, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/39); // 1.578ms
  tins.add({2, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/39); // 1.583ms
