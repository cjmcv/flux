#pragma once
#include <map>
#include <string>
#include <stdexcept>
#include <iostream>

namespace xop {

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
    std::map<std::string, GemmFactory> gemm_map;
    std::map<std::string, GemmBase*> created_instances;

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

    // 注册函数
    void add(const std::string& name, GemmFactory factory) {
        gemm_map[name] = factory;
    }
    GemmBase* createGemm(const std::string& name) {
      auto it = gemm_map.find(name);
      if (it != gemm_map.end()) {
        GemmBase* instance = it->second();
        created_instances[name] = instance;
        return instance;
      }
      else {
        std::cout << "Registed gemm:" << std::endl;
        for (const auto& pair : gemm_map) {
          std::cout << pair.first << std::endl;
        }
      }
      throw std::runtime_error("Gemm type not found.");
    }
    // 获取 Gemm 实例，如果已存在则直接返回，不存在则创建
    GemmBase* getGemm(const std::string& name) {
        auto it = created_instances.find(name);
        if (it != created_instances.end()) {
            return it->second;
        }
        return createGemm(name);
    }

    // 析构时释放所有创建的实例
    ~GemmConfigRegister() {
        for (auto& pair : created_instances) {
            delete pair.second;
        }
    }
};

} // namespace xop