#include "ctlop/ctlop.h"
#include "ctlop/ops_impl/global_resource.h"

namespace ctlop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_config_gemmnormal = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({1, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{110, (int16_t)ME::GemmNormal}); // 1.537ms
  // tins.add({1, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{98, (int16_t)ME::GemmNormal}); // 1.539ms
  // tins.add({1, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.539ms
  tins.add({2, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{110, (int16_t)ME::GemmNormal}); // 1.539ms
  // tins.add({2, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{122, (int16_t)ME::GemmNormal}); // 1.539ms
  // tins.add({2, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{117, (int16_t)ME::GemmNormal}); // 1.546ms
  tins.add({3, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{122, (int16_t)ME::GemmNormal}); // 1.542ms
  // tins.add({3, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.542ms
  // tins.add({3, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{128, (int16_t)ME::GemmNormal}); // 1.544ms
  tins.add({4, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{129, (int16_t)ME::GemmNormal}); // 1.544ms
  // tins.add({4, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{110, (int16_t)ME::GemmNormal}); // 1.547ms
  // tins.add({4, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{74, (int16_t)ME::GemmNormal}); // 1.547ms
  tins.add({5, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{129, (int16_t)ME::GemmNormal}); // 1.544ms
  // tins.add({5, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{125, (int16_t)ME::GemmNormal}); // 1.548ms
  // tins.add({5, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{124, (int16_t)ME::GemmNormal}); // 1.55ms
  tins.add({6, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{81, (int16_t)ME::GemmNormal}); // 1.547ms
  // tins.add({6, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.55ms
  // tins.add({6, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{129, (int16_t)ME::GemmNormal}); // 1.55ms
  tins.add({7, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{112, (int16_t)ME::GemmNormal}); // 1.551ms
  // tins.add({7, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{113, (int16_t)ME::GemmNormal}); // 1.554ms
  // tins.add({7, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{86, (int16_t)ME::GemmNormal}); // 1.555ms
  tins.add({8, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{110, (int16_t)ME::GemmNormal}); // 1.546ms
  // tins.add({8, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.552ms
  // tins.add({8, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{129, (int16_t)ME::GemmNormal}); // 1.556ms
  tins.add({9, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{120, (int16_t)ME::GemmNormal}); // 1.553ms
  // tins.add({9, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{117, (int16_t)ME::GemmNormal}); // 1.554ms
  // tins.add({9, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{110, (int16_t)ME::GemmNormal}); // 1.555ms
  tins.add({10, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{120, (int16_t)ME::GemmNormal}); // 1.551ms
  // tins.add({10, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.56ms
  // tins.add({10, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{129, (int16_t)ME::GemmNormal}); // 1.561ms
  tins.add({11, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{129, (int16_t)ME::GemmNormal}); // 1.555ms
  // tins.add({11, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{120, (int16_t)ME::GemmNormal}); // 1.556ms
  // tins.add({11, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{125, (int16_t)ME::GemmNormal}); // 1.557ms
  tins.add({12, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{122, (int16_t)ME::GemmNormal}); // 1.552ms
  // tins.add({12, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{117, (int16_t)ME::GemmNormal}); // 1.556ms
  // tins.add({12, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{86, (int16_t)ME::GemmNormal}); // 1.558ms
  tins.add({13, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{110, (int16_t)ME::GemmNormal}); // 1.56ms
  // tins.add({13, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{74, (int16_t)ME::GemmNormal}); // 1.562ms
  // tins.add({13, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{116, (int16_t)ME::GemmNormal}); // 1.562ms
  tins.add({14, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{125, (int16_t)ME::GemmNormal}); // 1.557ms
  // tins.add({14, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{120, (int16_t)ME::GemmNormal}); // 1.561ms
  // tins.add({14, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{122, (int16_t)ME::GemmNormal}); // 1.562ms
  tins.add({15, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.555ms
  // tins.add({15, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{122, (int16_t)ME::GemmNormal}); // 1.56ms
  // tins.add({15, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{112, (int16_t)ME::GemmNormal}); // 1.564ms
  tins.add({16, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{110, (int16_t)ME::GemmNormal}); // 1.563ms
  // tins.add({16, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.564ms
  // tins.add({16, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{120, (int16_t)ME::GemmNormal}); // 1.567ms
  tins.add({17, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{124, (int16_t)ME::GemmNormal}); // 1.563ms
  // tins.add({17, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.565ms
  // tins.add({17, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{117, (int16_t)ME::GemmNormal}); // 1.565ms
  tins.add({18, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.567ms
  // tins.add({18, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{129, (int16_t)ME::GemmNormal}); // 1.569ms
  // tins.add({18, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{113, (int16_t)ME::GemmNormal}); // 1.57ms
  tins.add({19, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{120, (int16_t)ME::GemmNormal}); // 1.564ms
  // tins.add({19, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{125, (int16_t)ME::GemmNormal}); // 1.564ms
  // tins.add({19, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.565ms
  tins.add({20, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.566ms
  // tins.add({20, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{112, (int16_t)ME::GemmNormal}); // 1.568ms
  // tins.add({20, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{125, (int16_t)ME::GemmNormal}); // 1.569ms
  tins.add({21, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{110, (int16_t)ME::GemmNormal}); // 1.566ms
  // tins.add({21, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{74, (int16_t)ME::GemmNormal}); // 1.572ms
  // tins.add({21, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{122, (int16_t)ME::GemmNormal}); // 1.572ms
  tins.add({22, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{120, (int16_t)ME::GemmNormal}); // 1.571ms
  // tins.add({22, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.574ms
  // tins.add({22, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{110, (int16_t)ME::GemmNormal}); // 1.574ms
  tins.add({23, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{117, (int16_t)ME::GemmNormal}); // 1.574ms
  // tins.add({23, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{110, (int16_t)ME::GemmNormal}); // 1.574ms
  // tins.add({23, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.578ms
  tins.add({24, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.57ms
  // tins.add({24, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{110, (int16_t)ME::GemmNormal}); // 1.574ms
  // tins.add({24, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{113, (int16_t)ME::GemmNormal}); // 1.577ms
  tins.add({25, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.58ms
  // tins.add({25, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{74, (int16_t)ME::GemmNormal}); // 1.582ms
  // tins.add({25, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{117, (int16_t)ME::GemmNormal}); // 1.582ms
  tins.add({26, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{120, (int16_t)ME::GemmNormal}); // 1.574ms
  // tins.add({26, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.576ms
  // tins.add({26, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{122, (int16_t)ME::GemmNormal}); // 1.576ms
  tins.add({27, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.57ms
  // tins.add({27, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{125, (int16_t)ME::GemmNormal}); // 1.572ms
  // tins.add({27, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{124, (int16_t)ME::GemmNormal}); // 1.575ms
  tins.add({28, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.57ms
  // tins.add({28, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{125, (int16_t)ME::GemmNormal}); // 1.576ms
  // tins.add({28, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{122, (int16_t)ME::GemmNormal}); // 1.577ms
  tins.add({29, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{112, (int16_t)ME::GemmNormal}); // 1.577ms
  // tins.add({29, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{113, (int16_t)ME::GemmNormal}); // 1.579ms
  // tins.add({29, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{124, (int16_t)ME::GemmNormal}); // 1.579ms
  tins.add({30, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{108, (int16_t)ME::GemmNormal}); // 1.577ms
  // tins.add({30, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{117, (int16_t)ME::GemmNormal}); // 1.583ms
  // tins.add({30, 27648, 5120, (int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::BF16,(int16_t)ME::FP32,(int16_t)ME::RCR,(int16_t)ME::Sm80}, /*config*/{81, (int16_t)ME::GemmNormal}); // 1.583ms
  return 0;
}();
}// clang-format on