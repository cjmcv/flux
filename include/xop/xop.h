
#pragma once
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef XOP_LIKELY
#define XOP_LIKELY(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef XOP_UNLIKELY
#define XOP_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

namespace xop {

class CheckFail {
 public:
  CheckFail() = default;

  // Overload the stream insertion operator.
  template <typename T>
  CheckFail &
  operator<<(const T &value) {
    message_ << value;
    return *this;
  }

  // Destructor that throws an exception with the accumulated error message.
  ~CheckFail() noexcept(false) {
    std::cerr << message_.str() << std::endl;
    throw std::runtime_error(message_.str());
  }

 private:
  std::ostringstream message_;
};

// Macro to check a condition and stream a custom error message if the check fails.
// Note that this can only be used in host code
#define XOP_CHECK(condition)            \
  if (!(condition))                      \
  ::xop::CheckFail() << __FILE__ << ":" << __LINE__ << " Check failed: " #condition ". "

#define XOP_CHECK_BINOP(lhs, rhs, op)                                                            \
  if (auto x = (lhs), y = (decltype(x))(rhs); XOP_UNLIKELY(!(x op y)))                           \
  ::xop::CheckFail() << __FILE__ << ":" << __LINE__ << " Check failed: " << x \
                       << "(" #lhs ") " #op " " << y << "(" #rhs ")"

#define XOP_CHECK_EQ(lhs, rhs) XOP_CHECK_BINOP((lhs), (rhs), ==)
#define XOP_CHECK_NE(lhs, rhs) XOP_CHECK_BINOP((lhs), (rhs), !=)
#define XOP_CHECK_LT(lhs, rhs) XOP_CHECK_BINOP((lhs), (rhs), <)
#define XOP_CHECK_GT(lhs, rhs) XOP_CHECK_BINOP((lhs), (rhs), >)
#define XOP_CHECK_LE(lhs, rhs) XOP_CHECK_BINOP((lhs), (rhs), <=)
#define XOP_CHECK_GE(lhs, rhs) XOP_CHECK_BINOP((lhs), (rhs), >=)


/////////////////////////////////////////////////////
// Enum classes
/////////////////////////////////////////////////////
// 算子不区分v2/v3
enum class UnifiedMetaEnum : int8_t {
  GemmNormal = 0, GemmNormalSimt, GemmBlockScaleFp8,   // gemm type
  GemmGroupedBlockScaleFp8, GemmLt, 
  GemmAllreduce = 20,
  Void = 50, FP16, BF16, FP32, E4M3, E5M2, S8, S32,    // data type
  Sm80 = 60, Sm89, Sm90,                               // arch
  RRR = 70, RCR, RCC                                   // layout
};

// inline std::string MetaEnumToString(UnifiedMetaEnum value) {
//   switch (value) {
//     case UnifiedMetaEnum::GemmNormal:
//       return "GemmNormal";
//     case UnifiedMetaEnum::Void:
//       return "Void";
//     case UnifiedMetaEnum::FP16:
//       return "FP16";
//     case UnifiedMetaEnum::BF16:
//       return "BF16";
//     case UnifiedMetaEnum::FP32:
//       return "FP32";
//     case UnifiedMetaEnum::E4M3:
//       return "E4M3";
//     case UnifiedMetaEnum::E5M2:
//       return "E5M2";
//     case UnifiedMetaEnum::S8:
//       return "S8";
//     case UnifiedMetaEnum::S32:
//       return "S32";
//     case UnifiedMetaEnum::Sm80:
//       return "Sm80";
//     case UnifiedMetaEnum::Sm89:
//       return "Sm89";
//     case UnifiedMetaEnum::Sm90:
//       return "Sm90";
//     case UnifiedMetaEnum::RRR:
//       return "RRR";
//     case UnifiedMetaEnum::RCR:
//       return "RCR";
//     case UnifiedMetaEnum::RCC:
//       return "RCC";
//     default:
//       return "Unknown";
//   }
// }

enum class UnifiedHParamEnum : int8_t {
  GemmV2, GemmV3,                    // version
  Identity, StreamK,                 // swizzle
  SK, DP,                            // 
  Heuristic, AlongM, AlongN,         // type
  Cooperative, PingPong              // type
};

}  // namespace xop
