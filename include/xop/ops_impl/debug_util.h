
#pragma once

#include <cute/config.hpp>                     // CUTE_HOST_DEVICE
#include <cute/layout.hpp>                     // cute::Shape
#include <cute/tensor.hpp>
// #include <cutlass/cutlass.h>
#include <cutlass/util/device_memory.h>
// 3rdparty/cutlass/tools/util/include/cutlass/util/device_memory.h

namespace xop {

// synclog / CUTLASS_HOST_TRACE      => -DCUTLASS_DEBUG_TRACE_LEVEL=1
// cutlass::debug::dump_fragment     => tools/util/include/cutlass/util/device_dump.h
// cutlass::debug::dump_shmem
// cutlass::detail::TensorViewWrite  => tools/util/include/cutlass/util/tensor_view_io.h
//
// cute::print              => include/cute/atom/mma_atom.hpp
// cute::print_latex
// cute::print_latex_mma
// cute::print_layout_mma
// cute::print_layout       => include/cute/layout.hpp
// cute::print_tensor       => include/cute/tensor_impl.hpp


template<class Engine, class Layout>
void print_device_tensor(cute::Tensor<Engine, Layout> const& t) {
  // Assumes size = cosize, i.e. compact tensor
  std::vector<typename Engine::value_type> data_host(t.size());
  cutlass::device_memory::copy_to_host(data_host.data(), t.data(), t.size());
  auto t_host = cute::make_tensor(data_host.data(), t.layout());
  cute::print_tensor(t_host);
}


template <class Engine, class Layout>
CUTE_HOST_DEVICE void print_tensor_shape(const char* name, cute::Tensor<Engine,Layout> const& tensor) {  
  printf("%s: ", name); 
  auto tshape = cute::shape(tensor.layout());
  cute::print(tshape); 
  printf("\n");
}


template <class Engine, class Layout>
CUTE_HOST_DEVICE bool is_tensor_aligned(cute::Tensor<Engine,Layout> const& tensor, size_t alignment) {
  auto ptr = tensor.data();
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  return (addr % alignment) == 0;
}

CUTE_HOST_DEVICE void print_2xbfloat16(const char* name, uint32_t src) {
  const uint16_t* bytes = reinterpret_cast<const uint16_t*>(&src);
  printf("%s: %f, %f\n", name, 
    static_cast<float>(*reinterpret_cast<const cutlass::bfloat16_t*>(&bytes[0])), 
    static_cast<float>(*reinterpret_cast<const cutlass::bfloat16_t*>(&bytes[1])));
}

CUTE_HOST_DEVICE void print_2xhalf(const char* name, uint32_t src) {
  const uint16_t* bytes = reinterpret_cast<const uint16_t*>(&src);
  printf("%s: %f, %f\n", name, 
    static_cast<float>(*reinterpret_cast<const cutlass::half_t*>(&bytes[0])), 
    static_cast<float>(*reinterpret_cast<const cutlass::half_t*>(&bytes[1])));
}

CUTE_HOST_DEVICE void print_4xfp8e4m3(const char* name, uint32_t src) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&src);
  printf("%s: %f, %f, %f, %f\n", name, 
    static_cast<float>(*reinterpret_cast<const cutlass::float_e4m3_t*>(&bytes[0])), 
    static_cast<float>(*reinterpret_cast<const cutlass::float_e4m3_t*>(&bytes[1])),
    static_cast<float>(*reinterpret_cast<const cutlass::float_e4m3_t*>(&bytes[2])),
    static_cast<float>(*reinterpret_cast<const cutlass::float_e4m3_t*>(&bytes[3])));
}

// cute::print_tensor
template <class Engine, class Layout>
CUTE_HOST_DEVICE void print_tensor(const char* name, cute::Tensor<Engine,Layout> const& tensor, 
                                   bool is_tid0 = true, bool print_data = true) {
  // ((_4,_2,_2),_1,_4):((_1,_4,_8),_0,_16)
  // => shape  ((_4,_2,_2),_1,_4)
  // => stride ((_1,_4,_8),_0,_16)
  if (is_tid0 && !cute::thread0()) { return; }

  printf("%s ", name); 
  using ElementType = typename cute::Tensor<Engine,Layout>::value_type;
  if constexpr (std::is_same_v<ElementType, float>) {
    printf("(float): ");
  } else if constexpr (__is_same(ElementType, cutlass::half_t)) {
    printf("(half_t): ");
  } else if constexpr (__is_same(ElementType, cutlass::bfloat16_t)) {
    printf("(bfloat16_t): ");
  } else if constexpr (__is_same(ElementType, cutlass::float_e4m3_t)) {
    printf("(float_e4m3_t): ");
  }

  if (print_data)
    cute::print_tensor(tensor);
}

}  // namespace cute
