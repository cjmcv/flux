#pragma once

namespace ctlop {

struct RtArguments {
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

} // namespace ctlop