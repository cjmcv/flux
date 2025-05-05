#include "ctlop/ops_impl/gemm_normal/gemm_v2_impl.h"
#include "ctlop/ops_impl/gemm_normal/gemm_v2_simt_impl.h"

namespace ctlop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_config_gemmnormal = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({1, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{23, (int8_t)ME::GemmNormalSimt}); // 0.059ms
  tins.add({2, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{24, (int8_t)ME::GemmNormal}); // 0.051ms
  tins.add({3, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{10, (int8_t)ME::GemmNormalSimt}); // 0.056ms
  tins.add({4, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{38, (int8_t)ME::GemmNormal}); // 0.065ms
  tins.add({5, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{25, (int8_t)ME::GemmNormal}); // 0.057ms
  tins.add({6, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{9, (int8_t)ME::GemmNormalSimt}); // 0.054ms
  tins.add({7, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{41, (int8_t)ME::GemmNormal}); // 0.052ms
  tins.add({8, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{20, (int8_t)ME::GemmNormal}); // 0.037ms
  tins.add({9, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{1, (int8_t)ME::GemmNormal}); // 0.046ms
  tins.add({10, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{10, (int8_t)ME::GemmNormal}); // 0.047ms
  tins.add({11, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{16, (int8_t)ME::GemmNormal}); // 0.049ms
  tins.add({12, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{4, (int8_t)ME::GemmNormal}); // 0.035ms
  tins.add({13, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{9, (int8_t)ME::GemmNormalSimt}); // 0.036ms
  tins.add({14, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{9, (int8_t)ME::GemmNormal}); // 0.053ms
  tins.add({15, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{9, (int8_t)ME::GemmNormalSimt}); // 0.038ms
  tins.add({16, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{1, (int8_t)ME::GemmNormalSimt}); // 0.039ms
  tins.add({17, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{6, (int8_t)ME::GemmNormalSimt}); // 0.037ms
  tins.add({18, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{9, (int8_t)ME::GemmNormal}); // 0.041ms
  tins.add({19, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{2, (int8_t)ME::GemmNormal}); // 0.035ms
  tins.add({20, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{6, (int8_t)ME::GemmNormalSimt}); // 0.04ms
  tins.add({21, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{1, (int8_t)ME::GemmNormal}); // 0.048ms
  tins.add({22, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{8, (int8_t)ME::GemmNormalSimt}); // 0.03ms
  tins.add({23, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{11, (int8_t)ME::GemmNormalSimt}); // 0.038ms
  tins.add({24, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{8, (int8_t)ME::GemmNormalSimt}); // 0.058ms
  tins.add({25, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{24, (int8_t)ME::GemmNormal}); // 0.058ms
  tins.add({26, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{21, (int8_t)ME::GemmNormalSimt}); // 0.034ms
  tins.add({27, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{1, (int8_t)ME::GemmNormal}); // 0.055ms
  tins.add({28, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{36, (int8_t)ME::GemmNormal}); // 0.065ms
  tins.add({29, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{4, (int8_t)ME::GemmNormal}); // 0.047ms
  tins.add({30, 512, 256, (int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::BF16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*config*/{1, (int8_t)ME::GemmNormal}); // 0.04ms
  return 0;
}();
}// clang-format on