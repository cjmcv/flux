
#pragma once

#include "xop/xop.h"

#include <cmath>

namespace xop {

struct Strategy {
  static int CoarseGrainedTuningM(int actual_m, int max_m = 16384) {
    int tuned_m = 0;
    if (max_m == -1) {
      return actual_m;
    }
    else {
      if (actual_m <= 1) {
        tuned_m = 1;
      }
      else if (actual_m >= max_m) { // 4096, 8192, 16384, 32768, 65536
        tuned_m = max_m;
      }
      else {
        int exponent = static_cast<int>(std::floor(std::log2(actual_m)));
        tuned_m = std::pow(2, exponent);
      }
    }
    return tuned_m;
  }

  static std::vector<int> SplitChunkM(const int m, const int chunk_size) {
    int quotient = m / chunk_size;
    int remainder = m % chunk_size;
    
    std::vector<int> result;
    for (int i = 0; i < quotient; ++i) {
        result.push_back(chunk_size);
    }
    if (remainder != 0) {
      if (!result.empty()) {
        result.back() += remainder;
      } else {
        result.push_back(remainder);
      }
    }
    return result;
  }
};

}  // namespace xop
