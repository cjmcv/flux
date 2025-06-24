#include "xop/xop.h"
#include "xop/ops_impl/global_resource.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_config_gemmgroupedblockscalefp8 = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({8192,7168,4096,4,(int16_t)ME::E4M3,(int16_t)ME::E4M3,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm90}, /*config*/{15, (int16_t)ME::GemmGroupedBlockScaleFp8}); // 0.022ms
  // tins.add({1,27648,5120,1,(int16_t)ME::E4M3,(int16_t)ME::E4M3,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm90}, /*config*/{7, (int16_t)ME::GemmGroupedBlockScaleFp8}); // 0.022ms
  // tins.add({1,27648,5120,1,(int16_t)ME::E4M3,(int16_t)ME::E4M3,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm90}, /*config*/{16, (int16_t)ME::GemmGroupedBlockScaleFp8}); // 0.024ms
  tins.add({2,27648,5120,1,(int16_t)ME::E4M3,(int16_t)ME::E4M3,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm90}, /*config*/{10, (int16_t)ME::GemmGroupedBlockScaleFp8}); // 0.048ms
  // tins.add({2,27648,5120,1,(int16_t)ME::E4M3,(int16_t)ME::E4M3,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm90}, /*config*/{27, (int16_t)ME::GemmGroupedBlockScaleFp8}); // 0.049ms
  // tins.add({2,27648,5120,1,(int16_t)ME::E4M3,(int16_t)ME::E4M3,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm90}, /*config*/{35, (int16_t)ME::GemmGroupedBlockScaleFp8}); // 0.051ms
  tins.add({3,27648,5120,1,(int16_t)ME::E4M3,(int16_t)ME::E4M3,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm90}, /*config*/{32, (int16_t)ME::GemmGroupedBlockScaleFp8}); // 0.017ms
  // tins.add({3,27648,5120,1,(int16_t)ME::E4M3,(int16_t)ME::E4M3,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm90}, /*config*/{31, (int16_t)ME::GemmGroupedBlockScaleFp8}); // 0.018ms
  // tins.add({3,27648,5120,1,(int16_t)ME::E4M3,(int16_t)ME::E4M3,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm90}, /*config*/{7, (int16_t)ME::GemmGroupedBlockScaleFp8}); // 0.051ms
  return 0;
}();
}// clang-format on