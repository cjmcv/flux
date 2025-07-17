
#pragma once

#include <cute/config.hpp>                     // CUTE_HOST_DEVICE
#include <cute/layout.hpp>                     // cute::Shape
#include <cute/tensor.hpp>
// #include <cutlass/cutlass.h>
#include <cutlass/util/device_memory.h>
// 3rdparty/cutlass/tools/util/include/cutlass/util/device_memory.h
namespace cute {

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
void print_device_tensor(cute::Tensor<Engine, Layout> const& t)
{
  // Assumes size = cosize, i.e. compact tensor
  std::vector<typename Engine::value_type> data_host(t.size());
  cutlass::device_memory::copy_to_host(data_host.data(), t.data(), t.size());
  auto t_host = cute::make_tensor(data_host.data(), t.layout());
  cute::print_tensor(t_host);
}


template <class Engine, class Layout>
CUTE_HOST_DEVICE void print_tensor_shape(const char* name, Tensor<Engine,Layout> const& tensor)
{  
  print("%s: ", name); 
  auto tshape = shape(tensor.layout());
  print(tshape); 
  print("\n");
}

// cute::print_tensor
template <class Engine, class Layout>
CUTE_HOST_DEVICE void print_tensor(const char* name, Tensor<Engine,Layout> const& tensor, int tensor_type = 0, bool print_type = true)
{
  // ((_4,_2,_2),_1,_4):((_1,_4,_8),_0,_16)
  // => shape  ((_4,_2,_2),_1,_4)
  // => stride ((_1,_4,_8),_0,_16)
  print("%s: ", name); 
  if (print_type) {
    print(tensor); print(":\n");
  }

  if constexpr (Layout::rank == 1)
  {
    // print("dim[%d].\n", size(tensor));
    for (int m = 0; m < size(tensor); ++m) {
      pretty_print(tensor(m));
      printf("\n");
    }
  } else
  if constexpr (Layout::rank == 2)
  {
    // print("dim[%d, %d].\n", size<0>(tensor), size<1>(tensor));
    for (int m = 0; m < size<0>(tensor); ++m) {
      for (int n = 0; n < size<1>(tensor); ++n) {
        pretty_print(tensor(m,n));
      }
      printf("\n");
    }
  } else
  if constexpr (Layout::rank == 3)
  {
    // if (tensor_type == 0) {
    //   print_tensor(tensor(0,_,_), false);
    // }
    // else {
      print_tensor(tensor(_,_,0), false);
      for (int k = 1; k < size<2>(tensor); ++k) {
        for (int i = 0; i < 5*size<1>(tensor); ++i) { print("-"); } print("\n");
        print_tensor(tensor(_,_,k), false);
      }      
    // }
  } else
  if constexpr (Layout::rank == 4)
  {
    // print("dim[%d, %d, %d, %d].\n", size<0>(tensor) , size<1>(tensor), size<2>(tensor), size<3>(tensor));
    print_tensor(tensor(_,_,_,0), false);
    for (int p = 1; p < size<3>(tensor); ++p) {
      for (int i = 0; i < 5*size<1>(tensor); ++i) { print("="); } print("\n");
      print_tensor(tensor(_,_,_,p), false);
    }
  }
}

}  // namespace cute
