//===- gemm_meta.h ------------------------------------------------ C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <type_traits>
#include <variant>
#include "cute/container/tuple.hpp"
#include "flux/flux.h"

namespace bytedance::flux {

template <class... Ts>
struct GemmDTypeConfig : public FluxNamedTupleBase<GemmDTypeConfig, Ts...> {
 public:
  using Base = FluxNamedTupleBase<GemmDTypeConfig, Ts...>;
  static constexpr char const *Name = "GemmDTypeConfig";
  static constexpr char const *LowerName = "gemm_dtype_config";
  static constexpr std::array<char const *, 6> Fields = {"a", "b", "c", "d", "acc", "blockscale"};
  FLUX_NAMED_TUPLE_DEFINE_FIELD(a, 0)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(b, 1)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(c, 2)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(d, 3)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(acc, 4)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(blockscale, 5)

  constexpr GemmDTypeConfig() : Base() { check_type(); }
  constexpr GemmDTypeConfig(cute::tuple<Ts...> const &tup) : Base(tup) { check_type(); }

  constexpr bool
  is_input_fp8() const {
    return is_fp8_dtype(a()) && is_fp8_dtype(b());
  }

  constexpr bool
  is_input_s8() const {
    return is_s8_dtype(a()) && is_s8_dtype(b());
  }

  friend GemmDTypeConfig<
      DataTypeEnum,
      DataTypeEnum,
      DataTypeEnum,
      DataTypeEnum,
      DataTypeEnum,
      DataTypeEnum>
  unify_type(GemmDTypeConfig const &obj) {
    return cute::make_tuple(
        unify_type(obj.a()),
        unify_type(obj.b()),
        unify_type(obj.c()),
        unify_type(obj.d()),
        unify_type(obj.acc()),
        unify_type(obj.blockscale()));
  }

 protected:
  constexpr void
  check_type() const {
    static_assert(is_of_type_v<decltype(a()), DataTypeEnum>, "a() requires DataTypeEnum");
    static_assert(is_of_type_v<decltype(b()), DataTypeEnum>, "b() requires DataTypeEnum");
    static_assert(is_of_type_v<decltype(c()), DataTypeEnum>, "c() requires DataTypeEnum");
    static_assert(is_of_type_v<decltype(d()), DataTypeEnum>, "d() requires DataTypeEnum");
    static_assert(is_of_type_v<decltype(acc()), DataTypeEnum>, "acc() requires DataTypeEnum");
    static_assert(
        is_of_type_v<decltype(blockscale()), DataTypeEnum>, "blockscale() requires DataTypeEnum");
  }
};

namespace detail {
template <class T>
struct is_gemm_dtype_config : std::false_type {};

template <class... Ts>
struct is_gemm_dtype_config<GemmDTypeConfig<Ts...>> : std::true_type {};
}  // namespace detail

template <class T>
inline constexpr bool is_gemm_dtype_config_v =
    detail::is_gemm_dtype_config<decay_and_strip_t<T>>::value;

template <
    class DTypeA,
    class DTypeB,
    class DTypeC,
    class DTypeD,
    class DTypeAcc = _FP32,
    class DTypeBlockScale = _FP32>
constexpr GemmDTypeConfig<DTypeA, DTypeB, DTypeC, DTypeD, DTypeAcc, DTypeBlockScale>
make_gemm_dtype_config(
    DTypeA const &dtype_a,
    DTypeB const &dtype_b,
    DTypeC const &dtype_c,
    DTypeD const &dtype_d,
    DTypeAcc const &dtype_acc = _FP32{},
    DTypeBlockScale const &dtype_blockscale = _FP32{}) {
  return {cute::make_tuple(dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc, dtype_blockscale)};
}

template <class DTypeOrDTypeConfig>
constexpr auto
make_gemm_dtype_config(DTypeOrDTypeConfig const &x) {
  if constexpr (is_gemm_dtype_config_v<DTypeOrDTypeConfig>) {
    return x;
  } else {
    static_assert(is_of_type_v<DTypeOrDTypeConfig, DataTypeEnum>, " requires DataTypeEnum.");
    return make_gemm_dtype_config(x, x, x, x, _FP32{}, _FP32{});
  }
}

template <class... Ts>
constexpr GemmDTypeConfig<Ts...>
to_gemm_dtype_config(cute::tuple<Ts...> const &tup) {
  return {tup};
}

/////////////////////////////////////////////////////
// Impl-specific meta
/////////////////////////////////////////////////////
template <class... Ts>
struct GemmV2Meta : FluxNamedTupleBase<GemmV2Meta, Ts...> {
  using Base = FluxNamedTupleBase<GemmV2Meta, Ts...>;
  using Base::Base;

  static constexpr char const *Name = "GemmV2Meta";
  static constexpr char const *LowerName = "gemm_v2_meta";
  static constexpr std::array<char const *, 1> Fields = {"fast_accum"};

  FLUX_NAMED_TUPLE_DEFINE_FIELD(fast_accum, 0)

  constexpr GemmV2Meta(cute::tuple<Ts...> const &tup) : Base(tup) {}

  friend GemmV2Meta<bool>
  unify_type(GemmV2Meta const &obj) {
    return cute::make_tuple(bool(obj.fast_accum()));
  }
};

template <class FastAccum>
constexpr GemmV2Meta<FastAccum>
make_gemm_v2_meta(FastAccum const &fast_accum) {
  return {fast_accum};
}

inline constexpr GemmV2Meta<_False>
to_gemm_v2_meta(None const &) {
  return {_False{}};
}

template <class... Ts, __CUTE_REQUIRES(sizeof...(Ts) > 0)>
constexpr GemmV2Meta<Ts...>
to_gemm_v2_meta(cute::tuple<Ts...> const &tup) {
  return {tup};
}

using UnifiedImplMeta = std::variant<None, unified_type_t<GemmV2Meta>>;

/////////////////////////////////////////////////////
// GemmMeta: params does not change
/////////////////////////////////////////////////////
template <class... Ts>
struct GemmMeta : FluxNamedTupleBase<GemmMeta, Ts...> {
 public:
  using Base = FluxNamedTupleBase<GemmMeta, Ts...>;

  static constexpr const char *Name = "GemmMeta";
  static constexpr const char *LowerName = "gemm_meta";
  static constexpr std::array<const char *, 5> Fields = {
      "dtype", "arch", "gemm_layout", "impl", "impl_spec"};
  FLUX_NAMED_TUPLE_DEFINE_FIELD(dtype, 0)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(arch, 1)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(gemm_layout, 2)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(impl, 3)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(impl_spec, 4)

  constexpr GemmMeta() : Base() { check_type(); }
  constexpr GemmMeta(cute::tuple<Ts...> const &tup) : Base(tup) { check_type(); }

  using UnifiedDTConf = unified_type_t<GemmDTypeConfig>;
  using UnifiedGemmMeta = GemmMeta<
      UnifiedDTConf,
      ArchEnum,
      GemmLayoutEnum,
      ImplEnum,
      UnifiedImplMeta>;

  friend UnifiedGemmMeta
  unify_type(GemmMeta const &obj) {
    UnifiedImplMeta impl_spec = unify_type(obj.impl_spec());
    if ((obj.impl() == _GemmV2{}) and std::holds_alternative<None>(impl_spec)) {
      impl_spec = unify_type(to_gemm_v2_meta(std::get<None>(impl_spec)));
    }
    return cute::make_tuple(
        unify_type(make_gemm_dtype_config(obj.dtype())),
        unify_type(obj.arch()),
        unify_type(obj.gemm_layout()),
        unify_type(obj.impl()),
        cute::move(impl_spec));
  }

 protected:
  constexpr auto
  check_type() {
    static_assert(
        is_of_type_v<decltype(this->dtype()), DataTypeEnum> or
            is_gemm_dtype_config_v<decltype(this->dtype())>,
        "dtype() requires to be DataTypeEnum or GemmDTypeConfig.");
    static_assert(is_of_type_v<decltype(this->arch()), ArchEnum>, "arch() requires ArchEnum.");
    static_assert(
        is_of_type_v<decltype(this->gemm_layout()), GemmLayoutEnum>,
        "gemm_layout() requires GemmLayoutEnum.");
    static_assert(is_of_type_v<decltype(this->impl()), ImplEnum>, "impl() requires ImplEnum.");
  }
};

using UnifiedGemmMeta = unified_type_t<GemmMeta>;

template <
    class DataType,
    class Arch,
    class GemmLayout,
    class Impl,
    class ImplSpec = None>
constexpr GemmMeta<DataType, Arch, GemmLayout, Impl, ImplSpec>
make_gemm_meta(
    DataType const &data_type,
    Arch const &arch,
    GemmLayout const &gemm_layout,
    Impl const &impl,
    ImplSpec const &impl_spec = None{}) {
  return {cute::make_tuple(data_type, arch, gemm_layout, impl, impl_spec)};
}

template <class... Ts>
constexpr GemmMeta<Ts...>
to_gemm_meta(cute::tuple<Ts...> const &tup) {
  return {tup};
}

namespace detail {
template <class... Ts>
constexpr bool
filter_fast_accum(GemmMeta<Ts...> meta) {
  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  if constexpr (dt_conf.is_input_fp8()) {
    if (meta.impl() == _GemmV2{}) {
      return meta.arch() == _Sm89{};
    }
  } else if constexpr (meta.impl() == _GemmV3{}) {
    // FastAccum does not matter for non FP8 dtype, thus filter out True cases.
    return meta.impl_spec().fast_accum() == _False{};
  }
  return true;
}

template <class... Ts>
constexpr bool
filter_layout(GemmMeta<Ts...> meta) {
  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  // Hardware cannot transpose 8bit data, thus cannot support RRR layout
  if constexpr (dt_conf.is_input_fp8()) {
    return meta.gemm_layout() == _RRR{};
  }
  return false;
}

}  // namespace detail

/////////////////////////////////////////////////////
// Create a tuple of GemmMeta by cartesian product
// of given sets of elements
template <
    class... DTypes,
    class... Archs,
    class... GemmLayouts,
    class... Impls,
    class ImplSpecs = cute::tuple<None>>
constexpr auto
make_space_gemm_meta(
    cute::tuple<DTypes...> const &dtypes,
    cute::tuple<Archs...> const &archs,
    cute::tuple<GemmLayouts...> const &gemm_layouts,
    cute::tuple<Impls...> const &impls,
    ImplSpecs const &impl_specs = cute::make_tuple(None{})) {
  auto gemm_meta_space = tuple_transform(
      tuple_cartesian_product(
          dtypes, archs, gemm_layouts, impls, impl_specs),
      [](auto tup) { return to_gemm_meta(tup); });
  return tuple_filter(gemm_meta_space, [](auto const tup) {
    auto meta = to_gemm_meta(tup);
    if constexpr (not detail::filter_fast_accum(meta)) {
      return false;
    }
    if constexpr (detail::filter_layout(meta)) {
      return false;
    }
    return true;
  });
}
}  // namespace bytedance::flux
