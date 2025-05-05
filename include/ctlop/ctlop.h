
#pragma once
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef CTLOP_LIKELY
#define CTLOP_LIKELY(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef CTLOP_UNLIKELY
#define CTLOP_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

namespace ctlop {

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
#define CTLOP_CHECK(condition)            \
  if (!(condition))                      \
  ::ctlop::CheckFail() << __FILE__ << ":" << __LINE__ << " Check failed: " #condition ". "

#define CTLOP_CHECK_BINOP(lhs, rhs, op)                                                            \
  if (auto x = (lhs), y = (decltype(x))(rhs); CTLOP_UNLIKELY(!(x op y)))                           \
  ::ctlop::CheckFail() << __FILE__ << ":" << __LINE__ << " Check failed: " << x \
                       << "(" #lhs ") " #op " " << y << "(" #rhs ")"

#define CTLOP_CHECK_EQ(lhs, rhs) CTLOP_CHECK_BINOP((lhs), (rhs), ==)
#define CTLOP_CHECK_NE(lhs, rhs) CTLOP_CHECK_BINOP((lhs), (rhs), !=)
#define CTLOP_CHECK_LT(lhs, rhs) CTLOP_CHECK_BINOP((lhs), (rhs), <)
#define CTLOP_CHECK_GT(lhs, rhs) CTLOP_CHECK_BINOP((lhs), (rhs), >)
#define CTLOP_CHECK_LE(lhs, rhs) CTLOP_CHECK_BINOP((lhs), (rhs), <=)
#define CTLOP_CHECK_GE(lhs, rhs) CTLOP_CHECK_BINOP((lhs), (rhs), >=)


/////////////////////////////////////////////////////
// Enum classes
/////////////////////////////////////////////////////
enum class UnifiedMetaEnum : int8_t {
  GemmNormal = 0, GemmNormalSimt,                      // meta type
  Void = 10, FP16, BF16, FP32, E4M3, E5M2, S8, S32,    // data type
  Sm80 = 20, Sm89, Sm90,                               // arch
  RRR = 30, RCR, RCC                                   // layout
};

inline std::string MetaEnumToString(UnifiedMetaEnum value) {
  switch (value) {
    case UnifiedMetaEnum::GemmNormal:
      return "GemmNormal";
    case UnifiedMetaEnum::Void:
      return "Void";
    case UnifiedMetaEnum::FP16:
      return "FP16";
    case UnifiedMetaEnum::BF16:
      return "BF16";
    case UnifiedMetaEnum::FP32:
      return "FP32";
    case UnifiedMetaEnum::E4M3:
      return "E4M3";
    case UnifiedMetaEnum::E5M2:
      return "E5M2";
    case UnifiedMetaEnum::S8:
      return "S8";
    case UnifiedMetaEnum::S32:
      return "S32";
    case UnifiedMetaEnum::Sm80:
      return "Sm80";
    case UnifiedMetaEnum::Sm89:
      return "Sm89";
    case UnifiedMetaEnum::Sm90:
      return "Sm90";
    case UnifiedMetaEnum::RRR:
      return "RRR";
    case UnifiedMetaEnum::RCR:
      return "RCR";
    case UnifiedMetaEnum::RCC:
      return "RCC";
    default:
      return "Unknown";
  }
}

enum class UnifiedHParamEnum : int8_t {
  GemmV2, GemmV3,                    // version
  Identity, StreamK,                 // swizzle
  SK, DP,                            // 
  Heuristic, AlongM, AlongN,         // type
  Cooperative, PingPong              // type
};

}  // namespace ctlop
