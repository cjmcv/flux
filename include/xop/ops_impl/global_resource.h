#pragma once
#include <map>
#include <string>
#include <stdexcept>
#include <iostream>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "cute/algorithm/tuple_algorithms.hpp"
#include "cute/config.hpp"
#include "cute/container/tuple.hpp"
#include "cute/int_tuple.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/util/type_traits.hpp"

#include "xop/xop.h"
#include "arguments.h"

namespace xop {

// GemmDeviceBase
class GemmBase {
public:
  virtual void initialize(RtArgumentsBase *rt_args, void *fusion_args = nullptr, void *stream = nullptr) = 0;
  virtual void run(void *stream = nullptr) = 0;
};


class GlobalBuffer {
private:
  GlobalBuffer() = default;
  GlobalBuffer(const GlobalBuffer&) = delete;
  GlobalBuffer& operator=(const GlobalBuffer&) = delete;

  cutlass::device_memory::allocation<uint8_t> device_buffer_;
  cutlass::device_memory::allocation<uint8_t> device_buffer2_;
  std::vector<uint8_t> host_buffer_;

public:
  static GlobalBuffer& instance() {
    static GlobalBuffer instance;
    return instance;
  }

  uint8_t* ResizeDeviceBufferIfNeeded(size_t workspace_size) {
    workspace_size = (workspace_size + 127) / 128 * 128;
    if (device_buffer_.size() < workspace_size) {
      device_buffer_.reset(workspace_size);
    }
    return device_buffer_.get();
  }

  uint8_t* ResizeDeviceBuffer2IfNeeded(size_t size) {
    size = (size + 127) / 128 * 128;
    if (device_buffer2_.size() < size) {
      device_buffer2_.reset(size);
    }
    return device_buffer2_.get();
  }

  uint8_t* ResizeHostBufferIfNeeded(size_t size) {
    size = (size + 127) / 128 * 128;
    if (host_buffer_.size() < size) {
      host_buffer_.resize(size);
    }
    return host_buffer_.data();
  }
};

// 定义工厂函数类型
using GemmFactory = std::function<GemmBase*()>;

// 单例类来管理 gemm_map
class GemmConfigRegister {
private:
  std::map<std::vector<int16_t>, GemmFactory> gemm_map;
  std::map<std::vector<int16_t>, GemmBase*> created_instances;

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
      printf("\n");
    }
  }

  // 注册函数
  void add(const std::vector<int16_t> &key, GemmFactory factory) {
    gemm_map[key] = factory;
  }

  GemmBase* CreateOp(const std::vector<int16_t> &key, bool is_tuning = false) {
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
      PrintRegistered("notfound");
      throw std::runtime_error("Gemm type not found.");
    }

    return nullptr;
  }

  // 获取 Gemm 实例，如果已存在则直接返回，不存在则创建
  GemmBase* GetOp(const std::vector<int16_t> &key, bool is_tuning = false) {
    // printf("Looking for Gemm type {");
    // for (int i=0; i<key.size(); i++)
    //   printf("%d-", key[i]);

    auto it = created_instances.find(key);
    if (it != created_instances.end()) {
      return it->second;
    }
    return CreateOp(key, is_tuning);
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
  std::map<std::vector<int32_t>, std::vector<int16_t>> normal_tuned_map;
  std::map<std::vector<int32_t>, std::vector<int16_t>> comm_tuned_map;

  TunedConfigRegister() = default;
  TunedConfigRegister(const TunedConfigRegister&) = delete;
  TunedConfigRegister& operator=(const TunedConfigRegister&) = delete;

public:
  static TunedConfigRegister& instance() {
      static TunedConfigRegister instance;
      return instance;
  }

  void add(const std::vector<int32_t> &key, const std::vector<int16_t> &select_config) {
    normal_tuned_map[key] = select_config;
  }
  void GetSelectedConfig(const std::vector<int32_t> &key, int16_t *selected_id, int16_t *schema_id, uint64_t *cublaslt_algo = nullptr) {
    auto it = normal_tuned_map.find(key);
    if (it != normal_tuned_map.end()) {
      *selected_id = it->second[0];
      *schema_id = it->second[1];
      if (cublaslt_algo != nullptr && *schema_id == (int16_t)UnifiedMetaEnum::GemmLt) {
        memcpy(cublaslt_algo, &it->second[2], sizeof(uint64_t) * 8);
      }
      return;
    }
    // else {
    //   printf("total normal_tuned_map.size: %d.\n", normal_tuned_map.size());
    //   for (auto it = normal_tuned_map.begin(); it != normal_tuned_map.end(); ++it) {
    //     const auto& key = it->first;
    //     const auto& value = it->second;

    //     std::cout << "Key: ";
    //     for (int32_t k : key) std::cout << k << " ";
    //       std::cout << " => Value: ";
    //     for (int16_t v : value) std::cout << v << " ";
    //       std::cout << "\n";
    //   }
    // }
  }  
  
  void Add2Comm(const std::vector<int32_t> &key, const std::vector<int16_t> &select_config) {
    comm_tuned_map[key] = select_config;
  }
  void GetCommSelectedConfig(const std::vector<int32_t> &key, int16_t *selected_id, int16_t *schema_id, uint64_t *cublaslt_algo = nullptr) {
    auto it = comm_tuned_map.find(key);
    if (it != comm_tuned_map.end()) {
      *selected_id = it->second[0];
      *schema_id = it->second[1];
      return;
    }
  }

  ~TunedConfigRegister() {}
};

} // namespace xop