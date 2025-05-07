#pragma once

namespace ctlop {

struct RtArguments {
  int m;
  int n;
  int k;
  int l; // batch

  void *ptr_A;
  void *ptr_B;
  void *ptr_C;
  void *ptr_D;  

  virtual ~RtArguments() {}
};

struct RtArgumentsV2 : public RtArguments {
  float alpha;
  float beta;

  int stride_a;
  int stride_b;
  int stride_c;
  int stride_d;
};

struct RtBlockScaleFp8ArgumentsV3 : public RtArguments {

  float alpha;
  float beta;

  float scale_a = 1.f, scale_b = 1.f, scale_c = 1.f, scale_d = 1.f, scale_aux = 1.f;

  void *d_blockscale_A; // blockscale_tensor_A.device_data(),
  void *d_blockscale_B; // blockscale_tensor_B.device_data()

  // debug
  bool save_aux;
  bool save_amax;
  void *d_tensor_aux;
  void *d_abs_max_aux;
  void *d_abs_max_D;
};


} // namespace ctlop