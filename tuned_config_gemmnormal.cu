#include "ctlop/ops_impl/gemm_normal/gemm_v2_impl.h"
#include "ctlop/ops_impl/gemm_normal/gemm_v2_simt_impl.h"

namespace ctlop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_config_gemmnormal = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({1, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{44, (int8_t)ME::GemmNormal}); // 0.048ms
  tins.add({2, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{21, (int8_t)ME::GemmNormalSimt}); // 0.045ms
  tins.add({3, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{19, (int8_t)ME::GemmNormal}); // 0.046ms
  tins.add({4, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{2, (int8_t)ME::GemmNormalSimt}); // 0.042ms
  tins.add({5, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{9, (int8_t)ME::GemmNormalSimt}); // 0.035ms
