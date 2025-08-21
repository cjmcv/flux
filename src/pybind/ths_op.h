
#pragma once
#include "c10/util/Optional.h"
#include <torch/torch.h>
#include <ATen/core/ivalue.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/utils/pybind.h>

// #define ENABLE_GEMM_NORMAL 1
#define ENABLE_GEMM_COMM 1
// #define ENABLE_MARLIN_KERNEL 1
// #define ENABLE_FLASH_ATTEN 1
// #define ENABLE_ALLREDUCE_CUSTOM 1

#define XOP_TORCH_EXTENSION_NAME xop_pybind

namespace xop {
// <NT> torch::CustomClassHolder 是 PyTorch 提供的基类，它能让自定义类在 Python 和 C++ 之间顺利交互。
// 先自定义类，通过 TorchClassWrapper 模板结构体对其进行包装 如TorchClassWrapper<MyCustomClass>，
// 然后利用 ThsOpsInitRegistry 将 TorchClassWrapper<MyCustomClass> 注册到 PyTorch 库。
template <typename T>
struct TorchClassWrapper : public torch::CustomClassHolder, T {
 public:
  using T::T;
};

// Registry of functions that register
// functions into module
class ThsOpsInitRegistry {
 public:
  using OpInitFunc = std::function<void(py::module &)>;
  static ThsOpsInitRegistry &instance();
  void register_one(std::string name, OpInitFunc &&func);
  void initialize_all(py::module &m) const;

 private:
  std::map<std::string, OpInitFunc> registry_;
  mutable std::mutex register_mutex_;

  ThsOpsInitRegistry() {}
  ThsOpsInitRegistry(const ThsOpsInitRegistry &) = delete;
  ThsOpsInitRegistry &operator=(const ThsOpsInitRegistry &) = delete;
};

}  // namespace xop
