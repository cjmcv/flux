#pragma once
#include <vector>
namespace xop {

struct RtArguments {
  int m;
  int n;
  int k;
  int l; // batch

  void *ptr_A;
  void *ptr_B;
  void *ptr_C;
  void *ptr_D;  

  float alpha;
  float beta;

  virtual ~RtArguments() {}
};

struct RtArgumentsV2 : public RtArguments {
  int stride_a;
  int stride_b;
  int stride_c;
  int stride_d;
};

struct RtBlockScaleFp8ArgumentsV3 : public RtArguments {
  std::vector<int32_t> problem_sizes; // mnk,mnk,mnk...
  int groups;
  
  float scale_a = 1.f, scale_b = 1.f, scale_c = 1.f, scale_d = 1.f, scale_aux = 1.f;

  void *d_blockscale_A;
  void *d_blockscale_B;

  // debug
  bool save_aux;
  bool save_amax;
  void *d_tensor_aux;
  void *d_abs_max_aux;
  void *d_abs_max_D;
};

struct RtGroupedBlockScaleFp8ArgumentsV3 : public RtArguments {
  int groups;
  std::vector<int32_t> problem_sizes; // mnk,mnk,mnk...

  std::vector<void const *> ptr_A;
  std::vector<void const *> ptr_B;
  std::vector<void const *> ptr_C;
  std::vector<void *> ptr_D;  

  std::vector<void const *> ptr_blockscale_A;
  std::vector<void const *> ptr_blockscale_B;
};

} // namespace xop