
#pragma once

#include "xop/xop.h"

#include <cmath>

namespace xop {

struct Strategy {
  static int CoarseGrainedTuningM(int actual_m, int schema = 0) {
    int tuned_m = 0;
    if (schema == 0) {
      return actual_m;
    }
    else if (schema == 1) {
      if (actual_m <= 1) {
        tuned_m = 1;
      }
      else if (actual_m >= 16384) { // 4096, 8192, 16384, 32768, 65536
        tuned_m = 16384;
      }
      else {
        int exponent = static_cast<int>(std::floor(std::log2(actual_m)));
        tuned_m = std::pow(2, exponent);
      }
    }
    else {
      printf("Unsupported CoarseGrainedTuning schema: %d.\n", schema);
    }
    return tuned_m;
  }
};

}  // namespace xop
