#include "ctlop/ops_impl/gemm_normal/gemm_v2_impl.h"
#include "ctlop/ops_impl/gemm_normal/gemm_v2_simt_impl.h"

namespace ctlop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_normal_sm89 = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({1, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.572ms
  tins.add({2, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.577ms
  tins.add({3, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.579ms
  tins.add({4, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.588ms
  tins.add({5, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.59ms
  tins.add({6, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/39); // 1.596ms
  tins.add({7, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.588ms
  tins.add({8, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.593ms
  tins.add({9, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/44); // 1.601ms
  tins.add({10, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.594ms
  tins.add({11, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.598ms
  tins.add({12, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.603ms
  tins.add({13, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.605ms
  tins.add({14, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.605ms
  tins.add({15, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.612ms
  tins.add({16, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.607ms
  tins.add({17, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.608ms
  tins.add({18, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.611ms
