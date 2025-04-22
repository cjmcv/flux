//===- gemm_hparams.h --------------------------------------------- C++ ---===//
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
#include "cute/container/tuple.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "cute/layout.hpp"

namespace bytedance::flux {
using UnifiedTileShape = cute::tuple<int64_t, int64_t, int64_t>;
template <class T, __CUTE_REQUIRES(cute::is_tuple<T>::value)>
constexpr UnifiedTileShape
unify_type(T const &tile_shape) {
  static_assert(cute::tuple_size_v<T> == 3, "tile_shape requires tuple_size == 3");
  return cute::make_tuple(
      static_cast<int64_t>(cute::size<0>(tile_shape)),
      static_cast<int64_t>(cute::size<1>(tile_shape)),
      static_cast<int64_t>(cute::size<2>(tile_shape)));
}

/////////////////////////////////////////////////////
// Impl specific gemm hparams
/////////////////////////////////////////////////////
template <class... Ts>
struct GemmV2HParams : public FluxNamedTupleBase<GemmV2HParams, Ts...> {
  using Base = FluxNamedTupleBase<GemmV2HParams, Ts...>;
  using Base::Base;
  static constexpr char const *Name = "GemmV2HParams";
  static constexpr char const *LowerName = "gemm_v2_hparams";
  static constexpr std::array<char const *, 3> Fields = {
      "warp_shape", "instruction_shape", "streamk_mode"};

  FLUX_NAMED_TUPLE_DEFINE_FIELD(warp_shape, 0)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(instruction_shape, 1)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(streamk_mode, 2)

  friend GemmV2HParams<UnifiedTileShape, UnifiedTileShape, GemmStreamkModeEnum>
  unify_type(GemmV2HParams const &obj) {
    return cute::make_tuple(
        unify_type(obj.warp_shape()),
        unify_type(obj.instruction_shape()),
        unify_type(obj.streamk_mode()));
  }
};

template <class WarpShape, class InstructionShape, class StreamkMode = _StreamkSK>
constexpr GemmV2HParams<WarpShape, InstructionShape, StreamkMode>
make_gemm_v2_hparams(
    WarpShape const &warp_shape,
    InstructionShape const &instruction_shape,
    StreamkMode const &streamk_mode = _StreamkSK{}) {
  return {cute::make_tuple(warp_shape, instruction_shape, streamk_mode)};
}

template <class... Ts>
constexpr GemmV2HParams<Ts...>
to_gemm_v2_hparams(cute::tuple<Ts...> const &tuple) {
  return {tuple};
}

/////////////////////////////////////////////////////
// GemmHParams: params can change for better
// better performance
/////////////////////////////////////////////////////
using UnifiedImplHParams =
    std::variant<None, unified_type_t<GemmV2HParams>>;

template <class... Ts>
struct GemmHParams : FluxNamedTupleBase<GemmHParams, Ts...> {
 public:
  using Base = FluxNamedTupleBase<GemmHParams, Ts...>;
  using Base::Base;
  static constexpr char const *Name = "GemmHParams";
  static constexpr char const *LowerName = "gemm_hparams";
  static constexpr std::array<char const *, 5> Fields = {
      "impl_spec", "tile_shape", "gemm_kind", "mainloop_stage", "raster_order"};
  FLUX_NAMED_TUPLE_DEFINE_FIELD(impl_spec, 0)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(tile_shape, 1)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(gemm_kind, 2)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(mainloop_stage, 3)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(raster_order, 4)

  constexpr bool
  is_materialized() const noexcept {
    bool has_auto = is_auto_v<decltype(impl_spec())> or 
                    is_auto_v<decltype(tile_shape())> or is_auto_v<decltype(gemm_kind())> or
                    is_auto_v<decltype(mainloop_stage())> or is_auto_v<decltype(raster_order())>;
    return not has_auto;
  }

  using UnifiedGemmHParams = GemmHParams<
      UnifiedImplHParams,
      UnifiedTileShape,
      GemmKindEnum,
      int,
      GemmRasterOrderEnum>;

  friend UnifiedGemmHParams
  unify_type(GemmHParams const &obj) {
    return cute::make_tuple(
        UnifiedImplHParams(unify_type(obj.impl_spec())),
        unify_type(obj.tile_shape()),
        unify_type(obj.gemm_kind()),
        int(obj.mainloop_stage()),
        unify_type(obj.raster_order()));
  }
};

using UnifiedGemmHParams = unified_type_t<GemmHParams>;

template <
    class ImplSpecific,
    class TileShape,
    class GemmKind = _GemmDefault,
    class MainloopStage = cute::_0,
    class RasterOrder = _RasterHeuristic>
constexpr GemmHParams<ImplSpecific, TileShape, GemmKind, MainloopStage, RasterOrder>
make_gemm_hparams(
    ImplSpecific const &impl_spec = Auto{},
    TileShape const &tile_shape = Auto{},
    GemmKind const &gemm_kind = _GemmDefault{},
    MainloopStage const &mainloop_stage = cute::_0{},
    RasterOrder const &raster_order = _RasterHeuristic{}) {
  return {
      cute::make_tuple(impl_spec, tile_shape, gemm_kind, mainloop_stage, raster_order)};
}

template <class... Ts>
constexpr GemmHParams<Ts...>
to_gemm_hparams(cute::tuple<Ts...> const &tup) {
  return {tup};
}

using _AutoHParams = GemmHParams<Auto, Auto, Auto, Auto, Auto>;

// Create a tuple of GemmHParams by cartesian product
// of given sets of elements
template <
    class ImplSpecifics = cute::tuple<Auto>,
    class TileShapes = cute::tuple<Auto>,
    class GemmKinds = cute::tuple<Auto>,
    class MainloopStages = cute::tuple<Auto>,
    class RasterOrders = cute::tuple<Auto>>
constexpr auto
make_space_gemm_hparams(
    ImplSpecifics const &impl_specs = cute::make_tuple(Auto{}),
    TileShapes const &tile_shapes = cute::make_tuple(Auto{}),
    GemmKinds const &gemm_kinds = cute::make_tuple(Auto{}),
    MainloopStages const &mainloop_stages = cute::make_tuple(Auto{}),
    RasterOrders const &raster_orders = cute::make_tuple(Auto{})) {
  auto gemm_hparams_spaces = tuple_transform(
      tuple_cartesian_product(
          impl_specs, tile_shapes, gemm_kinds, mainloop_stages, raster_orders),
      [](auto tup) { return to_gemm_hparams(tup); });
  return gemm_hparams_spaces;
}

/////////////////////////////////////////////////////
// Materialization of GemmHParams.
// Convert all Auto fields to specific values.
/////////////////////////////////////////////////////
namespace detail {
using namespace cute;

template <class... Ts>
constexpr auto
auto_impl_spec(GemmMeta<Ts...> meta) {
  if constexpr (meta.impl() == _GemmV2{}) {
    auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
    if constexpr (meta.arch() == _Sm89{} && dt_conf.is_input_fp8()) {
      return make_gemm_v2_hparams(Shape<_64, _32, _64>{}, Shape<_16, _8, _32>{});
    } else if constexpr (meta.arch() == _Sm80{} && dt_conf.is_input_s8()) {
      return make_gemm_v2_hparams(Shape<_64, _32, _128>{}, Shape<_16, _8, _32>{});
    } else {
      return make_gemm_v2_hparams(Shape<_64, _64, _32>{}, Shape<_16, _8, _16>{});
    }
  } else {
    static_assert(cutlass::detail::dependent_false<decltype(meta.impl())>, "unsupported impl");
  }
}

template <class... Ts, class ImplHParams, class TileShape>
constexpr auto
materialize_tile_shape_m(GemmMeta<Ts...> meta, ImplHParams impl_hparams, TileShape tile_shape) {
  static_assert(is_auto_v<TileShape> or cute::rank_v<TileShape> == 3);
  constexpr bool is_tile_m_auto = []() {
    if constexpr (is_auto_v<TileShape>) {
      return true;
    } else {
      return is_auto_v<decltype(cute::get<0>(TileShape{}))>;
    }
  }();

  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  if constexpr (!is_tile_m_auto) {
    return get<0>(tile_shape);
  } else {
    constexpr bool is_tile_n_auto = []() {
      if constexpr (is_auto_v<TileShape>) {
        return true;
      } else {
        return is_auto_v<decltype(cute::get<1>(TileShape{}))>;
      }
    }();

    if constexpr (meta.impl() == _GemmV2{}) {
      return 128;
    } else {
      static_assert(cutlass::detail::dependent_false<decltype(meta.impl())>, "unsupported impl");
    }
  }
}

template <class... Ts, class ImplHParams, class TileShape>
constexpr auto
materialize_tile_shape_n(GemmMeta<Ts...> meta, ImplHParams impl_hparams, TileShape tile_shape) {
  static_assert(is_auto_v<TileShape> or cute::rank_v<TileShape> == 3);
  constexpr bool is_tile_n_auto = []() {
    if constexpr (is_auto_v<TileShape>) {
      return true;
    } else {
      return is_auto_v<decltype(cute::get<1>(TileShape{}))>;
    }
  }();

  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  if constexpr (!is_tile_n_auto) {
    return get<1>(tile_shape);
  } else {
    constexpr bool is_tile_m_auto = []() {
      if constexpr (is_auto_v<TileShape>) {
        return true;
      } else {
        return is_auto_v<decltype(cute::get<0>(TileShape{}))>;
      }
    }();

    if constexpr (meta.impl() == _GemmV2{}) {
      return ((meta.arch() == _Sm89{} && dt_conf.is_input_fp8()) or
              (meta.arch() == _Sm80{} && dt_conf.is_input_s8()))
                 ? 64
                 : 128;
    } else {
      static_assert(cutlass::detail::dependent_false<decltype(meta.impl())>, "unsupported impl");
    }
  }
}

template <class... Ts, class ImplHParams, class TileShape>
constexpr int
materialize_tile_shape_k(GemmMeta<Ts...> meta, ImplHParams impl_hparams, TileShape tile_shape) {
  static_assert(is_auto_v<TileShape> or cute::rank_v<TileShape> == 3);
  constexpr bool is_tile_k_auto = []() {
    if constexpr (is_auto_v<TileShape>) {
      return true;
    } else {
      return is_auto_v<decltype(cute::get<2>(TileShape{}))>;
    }
  }();

  if constexpr (!is_tile_k_auto) {
    return get<2>(tile_shape);
  } else {
    auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
    if constexpr (meta.impl() == _GemmV2{}) {
      return (meta.arch() == _Sm89{} && dt_conf.is_input_fp8())  ? 64
             : (meta.arch() == _Sm80{} && dt_conf.is_input_s8()) ? 128
                                                                 : 32;
    } else if constexpr (meta.impl() == _GemmV3{}) {
      return meta.arch() == _Sm80{}
                 ? 32
                 : 128 / cute::max(sizeof_dtype(dt_conf.a()), sizeof_dtype(dt_conf.b()));
    } else {
      static_assert(cutlass::detail::dependent_false<decltype(meta.impl())>, "unsupported impl");
    }
  }
}

template <class... Ts, class ImplHParams, class TileShape>
constexpr auto
materialize_tile_shape(GemmMeta<Ts...> meta, ImplHParams impl_hparams, TileShape tile_shape) {
  return make_shape(
      Int<materialize_tile_shape_m(meta, impl_hparams, tile_shape)>{},
      Int<materialize_tile_shape_n(meta, impl_hparams, tile_shape)>{},
      Int<materialize_tile_shape_k(meta, impl_hparams, tile_shape)>{});
};

template <class... Ts>
constexpr auto
auto_gemm_kind(GemmMeta<Ts...> meta) {
  if constexpr (meta.impl() == _GemmV2{}) {
    return _GemmStreamK{};
  } else {
    return _GemmDefault{};
  }
};

template <class TileShape, class... Ts>
constexpr auto
auto_mainloop_stage(GemmMeta<Ts...> meta, TileShape const &) {
  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  if constexpr (meta.arch() == _Sm89{}) {
    if constexpr (dt_conf.is_input_fp8()) {
      return cute::_3{};
    } else {
      return cute::_4{};
    }
  } else if constexpr (meta.arch() == _Sm80{} && dt_conf.is_input_s8()) {
    return cute::_3{};
  } else {
    return cute::_4{};
  }
}

template <class... Ts>
constexpr auto
auto_raster_order(GemmMeta<Ts...> meta) {
  if constexpr (meta.arch() == _Sm89{}) {
    return _RasterAlongN{};
  } else {
    return _RasterHeuristic{};
  }
}

}  // namespace detail

template <class... Ts, class... Us>
constexpr auto
materialize_hparams(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  auto is_auto_or = [](auto v, auto w) {
    if constexpr (not is_auto_v<decltype(v)>) {
      return v;
    } else {
      return w;
    }
  };

  auto auto_impl_spec = detail::auto_impl_spec(meta);
  auto impl_spec = is_auto_or(hparams.impl_spec(), auto_impl_spec);
  auto tile_shape = detail::materialize_tile_shape(meta, impl_spec, hparams.tile_shape());
  auto auto_gemm_kind = detail::auto_gemm_kind(meta);
  auto auto_mainloop_stage = detail::auto_mainloop_stage(meta, tile_shape);
  auto auto_raster_order = detail::auto_raster_order(meta);

  return make_gemm_hparams(
      impl_spec,
      tile_shape,
      is_auto_or(hparams.gemm_kind(), auto_gemm_kind),
      is_auto_or(hparams.mainloop_stage(), auto_mainloop_stage),
      is_auto_or(hparams.raster_order(), auto_raster_order));
}

namespace detail {

template <class... Ts, class... Us>
constexpr bool
filter_smem(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  auto [tile_m, tile_n, tile_k] = hparams.tile_shape();
  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  int expect_min_smem = ((sizeof_dtype(dt_conf.a()) * tile_m * tile_k) +
                         (sizeof_dtype(dt_conf.b()) * tile_n * tile_k)) *
                        hparams.mainloop_stage();
  // print("!!!!!!!!!!!! expect min smem : %d\n", expect_min_smem);
  if (meta.arch() == _Sm80{} and expect_min_smem > 163 * 1024) {
    return false;
  }
  if (meta.arch() == _Sm89{} and expect_min_smem > 99 * 1024) {
    return false;
  }
  if (meta.arch() == _Sm90{} and expect_min_smem > 227 * 1024) {
    return false;
  }
  return true;
}


}  // namespace detail

}  // namespace bytedance::flux
