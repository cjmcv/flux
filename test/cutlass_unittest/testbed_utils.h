/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Tests for device-wide GEMM interface
*/

#pragma once
#include <iostream>
#include "cutlass/cutlass.h"

#define CUTLASS_TEST_UNIT_ENABLE_WARNINGS 1
// #define EXPECT_TRUE std::ostream

// 定义一个断言失败时的处理函数
class AssertionFailure {
  public:
      AssertionFailure(const std::string& file, int line, const std::string& message)
          : file_(file), line_(line), message_(message) {}
  
      void Report() const {
          std::cerr << "Assertion failed at " << file_ << ":" << line_ << ": " << message_ << std::endl;
      }
  
  private:
      std::string file_;
      int line_;
      std::string message_;
  };
  
  // 定义一个流式断言类
  class ExpectTrue {
  public:
      ExpectTrue(bool condition, const std::string& file, int line)
          : condition_(condition), file_(file), line_(line) {}
  
      template <typename T>
      ExpectTrue& operator<<(const T& value) {
          if (!condition_) {
              std::ostringstream oss;
              oss << value;
              message_ += oss.str();
          }
          return *this;
      }
  
      ~ExpectTrue() {
          if (!condition_) {
              AssertionFailure(file_, line_, message_).Report();
          }
      }
  
  private:
      bool condition_;
      std::string file_;
      int line_;
      std::string message_;
  };
  
  // 定义 EXPECT_TRUE 宏
  #define EXPECT_TRUE(expression) \
      ExpectTrue((expression), __FILE__, __LINE__)
  
#define EXPECT_GT(a, b) \
  ExpectTrue(((a) > (b)), __FILE__, __LINE__)

inline char const *to_string(cutlass::Status status) {

  switch (status) {
    case cutlass::Status::kSuccess: return "kSuccess";
    case cutlass::Status::kErrorMisalignedOperand: return "kErrorMisalignedOperand";
    case cutlass::Status::kErrorInvalidLayout: return "kErrorInvalidLayout";
    case cutlass::Status::kErrorInvalidProblem: return "kErrorInvalidProblem";
    case cutlass::Status::kErrorNotSupported: return "kErrorNotSupported";
    case cutlass::Status::kErrorWorkspaceNull: return "kErrorWorkspaceNull";
    case cutlass::Status::kErrorInternal: return "kErrorInternal";
    case cutlass::Status::kInvalid: return "kInvalid";
    default: break;
  }
  return "invalid";
}
