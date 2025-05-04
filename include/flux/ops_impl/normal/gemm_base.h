#pragma once
#include <map>
#include <string>
#include <stdexcept>
#include <iostream>
#include <vector>

#include "cute/algorithm/tuple_algorithms.hpp"
#include "cute/config.hpp"
#include "cute/container/tuple.hpp"
#include "cute/int_tuple.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/util/type_traits.hpp"

namespace xop {

/////////////////////////////////////////////////////////////
template <typename... Enums>
using EnumTuple = std::tuple<Enums...>;

struct RtParams {
  int m;
  int n;
  int k;

  float alpha;
  float beta;

  void *ptr_A;
  void *ptr_B;
  void *ptr_C;
  void *ptr_D;

  int stride_a;
  int stride_b;
  int stride_c;
  int stride_d;
};

class GemmBase {
public:
  virtual void initialize(RtParams &rt_params, void *stream = nullptr) = 0;
  virtual void run(void *stream = nullptr) = 0;
};

// 定义工厂函数类型
using GemmFactory = std::function<GemmBase*()>;

// 单例类来管理 gemm_map
class GemmConfigRegister {
private:
  std::map<std::vector<int8_t>, GemmFactory> gemm_map;
  std::map<std::vector<int8_t>, GemmBase*> created_instances;

  // 私有构造函数，防止外部实例化
  GemmConfigRegister() = default;

  // 防止拷贝构造和赋值操作
  GemmConfigRegister(const GemmConfigRegister&) = delete;
  GemmConfigRegister& operator=(const GemmConfigRegister&) = delete;

public:
  // 获取单例实例
  static GemmConfigRegister& instance() {
      static GemmConfigRegister instance;
      return instance;
  }

  void PrintRegistered(std::string loc_tag) {
    printf("\nloc_tag: %s, map.size() = %zd \n", loc_tag.c_str(), gemm_map.size());
    std::cout << "Registed gemm:" << std::endl;
    for (const auto& pair : gemm_map) {
      printf("  size: %zd: ", pair.first.size());
      for (int i=0; i<pair.first.size(); i++) {
        printf("%d, ", pair.first[i]);
      }
    }
  }

  // 注册函数
  void add(const std::vector<int8_t> &key, GemmFactory factory) {
    gemm_map[key] = factory;
  }

  GemmBase* createGemm(const std::vector<int8_t> &key, bool is_tuning = false) {
    // printf("name: %s.\n", name.c_str());
    // std::cout << key << std::endl;
    auto it = gemm_map.find(key);
    if (it != gemm_map.end()) {
      GemmBase* instance = it->second();
      created_instances[key] = instance;
      return instance;
    }
    
    if (is_tuning == false) {
      printf("Gemm type {");
      for (int i=0; i<key.size(); i++)
        printf("%d-", key[i]);
      printf("} not found.\n");
      throw std::runtime_error("Gemm type not found.");
    }

    return nullptr;
  }

  // 获取 Gemm 实例，如果已存在则直接返回，不存在则创建
  GemmBase* getGemm(const std::vector<int8_t> &key, bool is_tuning = false) {
    auto it = created_instances.find(key);
    if (it != created_instances.end()) {
        return it->second;
    }
    return createGemm(key, is_tuning);
  }

  // 析构时释放所有创建的实例
  ~GemmConfigRegister() {
    for (auto& pair : created_instances) {
      delete pair.second;
    }
  }
};

class TunedConfigRegister {
private:
  std::map<std::vector<int32_t>, int32_t> tuned_map;

  TunedConfigRegister() = default;

  // 防止拷贝构造和赋值操作
  TunedConfigRegister(const TunedConfigRegister&) = delete;
  TunedConfigRegister& operator=(const TunedConfigRegister&) = delete;

public:
  static TunedConfigRegister& instance() {
      static TunedConfigRegister instance;
      return instance;
  }

  void add(const std::vector<int32_t> &key, int selected_id) {
    tuned_map[key] = selected_id;
  }

  int32_t GetSelectedId(const std::vector<int32_t> &key) {
    auto it = tuned_map.find(key);
    if (it != tuned_map.end()) {
      return it->second;
    }
    return 0;  // auto
  }

  ~TunedConfigRegister() {}
};

} // namespace xop