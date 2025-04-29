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


/////////////////////////////////////////////////////
// Enum classes
/////////////////////////////////////////////////////
enum class DataTypeEnum : int8_t { Void, FP16, BF16, FP32, E4M3, E5M2, S8, S32 };
enum class ArchEnum : int { Sm80 = 80, Sm89 = 89, Sm90 = 90 };

enum class GemmLayoutEnum : int8_t { RRR, RCR, RCC };
enum class ImplEnum : int8_t { GemmV2, GemmV3 };

enum class GemmKindEnum : int8_t { GemmDefault, GemmStreamK };

enum class GemmStreamkModeEnum : int8_t { SK, DP };
enum class GemmRasterOrderEnum : int8_t { Heuristic, AlongM, AlongN };

enum class GemmKernelScheduleEnum : int8_t { Cooperative, PingPong };

/////////////////////////////////////////////////////
// Aliases for constant types
/////////////////////////////////////////////////////
using _GemmDefault = cute::C<GemmKindEnum::GemmDefault>;
using _GemmStreamK = cute::C<GemmKindEnum::GemmStreamK>;

using _Void = cute::C<DataTypeEnum::Void>;
using _FP32 = cute::C<DataTypeEnum::FP32>;
using _FP16 = cute::C<DataTypeEnum::FP16>;
using _BF16 = cute::C<DataTypeEnum::BF16>;
using _E4M3 = cute::C<DataTypeEnum::E4M3>;
using _E5M2 = cute::C<DataTypeEnum::E5M2>;
using _S32 = cute::C<DataTypeEnum::S32>;
using _S8 = cute::C<DataTypeEnum::S8>;

using _RRR = cute::C<GemmLayoutEnum::RRR>;
using _RCR = cute::C<GemmLayoutEnum::RCR>;
using _RCC = cute::C<GemmLayoutEnum::RCC>;

using _Sm80 = cute::C<ArchEnum::Sm80>;
using _Sm89 = cute::C<ArchEnum::Sm89>;
using _Sm90 = cute::C<ArchEnum::Sm90>;

using _GemmV2 = cute::C<ImplEnum::GemmV2>;
using _GemmV3 = cute::C<ImplEnum::GemmV3>;

using _True = cute::C<true>;
using _False = cute::C<false>;

using _StreamkSK = cute::C<GemmStreamkModeEnum::SK>;
using _StreamkDP = cute::C<GemmStreamkModeEnum::DP>;
using _RasterHeuristic = cute::C<GemmRasterOrderEnum::Heuristic>;
using _RasterAlongM = cute::C<GemmRasterOrderEnum::AlongM>;
using _RasterAlongN = cute::C<GemmRasterOrderEnum::AlongN>;

using _Cooperative = cute::C<GemmKernelScheduleEnum::Cooperative>;
using _PingPong = cute::C<GemmKernelScheduleEnum::PingPong>;

// struct Auto : cute::tuple<> {};
// struct None : cute::tuple<> {};

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
  virtual void initialize(RtParams &rt_params) = 0;
  virtual void run() = 0;
};

// 定义工厂函数类型
using GemmFactory = std::function<GemmBase*()>;

// 单例类来管理 gemm_map
class GemmConfigRegister {
private:
  std::map<std::vector<int>, GemmFactory> gemm_map;
  std::map<std::vector<int>, GemmBase*> created_instances;

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
    printf("\nloc_tag: %s, map.size() = %ud \n", loc_tag.c_str(), gemm_map.size());
    std::cout << "Registed gemm:" << std::endl;
    for (const auto& pair : gemm_map) {
      printf("  size: %ud: ", pair.first.size());
      for (int i=0; i<pair.first.size(); i++) {
        printf("%d, ", pair.first[i]);
      }
    }
  }

  // 注册函数
  void add(const std::vector<int> key, GemmFactory factory) {
    gemm_map[key] = factory;
  }

  GemmBase* createGemm(const std::vector<int> &key) {
    // printf("name: %s.\n", name.c_str());
    // std::cout << key << std::endl;
    auto it = gemm_map.find(key);
    if (it != gemm_map.end()) {
      GemmBase* instance = it->second();
      created_instances[key] = instance;
      return instance;
    }
    throw std::runtime_error("Gemm type not found.");
  }

  // 获取 Gemm 实例，如果已存在则直接返回，不存在则创建
  GemmBase* getGemm(const std::vector<int> key) {
    auto it = created_instances.find(key);
    if (it != created_instances.end()) {
        return it->second;
    }
    return createGemm(key);
  }

  // 析构时释放所有创建的实例
  ~GemmConfigRegister() {
    for (auto& pair : created_instances) {
      delete pair.second;
    }
  }
};

} // namespace xop