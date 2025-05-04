#include "flux/ops_impl/normal/gemm_v2_impl.h"
#include "flux/ops_impl/normal/gemm_v2_simt_impl.h"

namespace xop {
using namespace cutlass;
using ME = UnifiedMetaEnum;

static int tuned_normal_sm89 = []() {
  TunedConfigRegister& tins = TunedConfigRegister::instance();
  tins.add({1, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.584ms
  tins.add({2, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/39); // 1.578ms
  tins.add({3, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.581ms
  tins.add({4, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/39); // 1.587ms
  tins.add({5, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.588ms
  tins.add({6, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.594ms
  tins.add({7, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.59ms
  tins.add({8, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/41); // 1.602ms
  tins.add({9, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/44); // 1.6ms
  tins.add({10, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/26); // 1.605ms
  tins.add({11, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/26); // 1.611ms
  tins.add({12, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/36); // 1.609ms
  tins.add({13, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/36); // 1.61ms
  tins.add({14, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.599ms
  tins.add({15, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.608ms
  tins.add({16, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.614ms
  tins.add({17, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.6ms
  tins.add({18, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.614ms
  tins.add({19, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.608ms
  tins.add({20, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.608ms
  tins.add({21, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.614ms
  tins.add({22, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.621ms
  tins.add({23, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.616ms
  tins.add({24, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.612ms
  tins.add({25, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.626ms
  tins.add({26, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.62ms
  tins.add({27, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.615ms
  tins.add({28, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.619ms
  tins.add({29, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.624ms
  tins.add({30, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.623ms
  tins.add({31, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/44); // 1.629ms
  tins.add({32, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.62ms
  tins.add({33, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.632ms
  tins.add({34, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.62ms
  tins.add({35, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.631ms
  tins.add({36, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.634ms
  tins.add({37, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.627ms
  tins.add({38, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.629ms
  tins.add({39, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.628ms
  tins.add({40, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.635ms
  tins.add({41, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.641ms
  tins.add({42, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.63ms
  tins.add({43, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.64ms
  tins.add({44, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.632ms
  tins.add({45, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.638ms
  tins.add({46, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.649ms
  tins.add({47, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.641ms
  tins.add({48, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.649ms
  tins.add({49, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.645ms
  tins.add({50, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.638ms
  tins.add({51, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/45); // 1.644ms
  tins.add({52, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.654ms
  tins.add({53, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.655ms
  tins.add({54, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.643ms
  tins.add({55, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.646ms
  tins.add({56, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.646ms
  tins.add({57, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.645ms
  tins.add({58, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.645ms
  tins.add({59, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.658ms
  tins.add({60, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.666ms
  tins.add({61, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.662ms
  tins.add({62, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.657ms
  tins.add({63, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.655ms
  tins.add({64, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/38); // 1.665ms
  tins.add({65, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.724ms
  tins.add({66, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/2); // 1.726ms
  tins.add({67, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.732ms
  tins.add({68, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/5); // 1.73ms
  tins.add({69, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.745ms
  tins.add({70, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/5); // 1.724ms
  tins.add({71, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/5); // 1.726ms
  tins.add({72, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/5); // 1.731ms
  tins.add({73, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/5); // 1.728ms
  tins.add({74, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/5); // 1.739ms
  tins.add({75, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.743ms
  tins.add({76, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/5); // 1.739ms
  tins.add({77, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/5); // 1.739ms
  tins.add({78, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.742ms
  tins.add({79, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.745ms
  tins.add({80, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.82ms
  tins.add({81, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/2); // 1.832ms
  tins.add({82, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.841ms
  tins.add({83, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/5); // 1.784ms
  tins.add({84, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.784ms
  tins.add({85, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/1); // 1.876ms
  tins.add({86, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.842ms
  tins.add({87, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.92ms
  tins.add({88, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/5); // 1.809ms
  tins.add({89, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.801ms
  tins.add({90, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.802ms
  tins.add({91, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/1); // 1.867ms
  tins.add({92, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.79ms
  tins.add({93, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/2); // 1.976ms
  tins.add({94, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/2); // 1.862ms
  tins.add({95, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/8); // 1.927ms
  tins.add({96, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.809ms
  tins.add({97, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/2); // 1.81ms
  tins.add({98, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.811ms
  tins.add({99, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.782ms
  tins.add({100, 27648, 5120, (int8_t)ME::Normal,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP16,(int8_t)ME::FP32,(int8_t)ME::RCR,(int8_t)ME::Sm80}, /*id*/4); // 1.857ms
  return 0;
}();
}// clang-format on