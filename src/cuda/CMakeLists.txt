cmake_minimum_required(VERSION 3.17)
if (WITH_PROTOBUF)
  add_compile_options("-DWITH_PROTOBUF")
  set(flux_proto_lib flux_proto ${Protobuf_LIBRARIES})
  set(flux_proto_inc ${Protobuf_INCLUDE_DIRS})
else ()
  set(flux_proto_lib)
  set(flux_proto_inc)
endif()

set(LIB_FILES
  op_registry.cu

)
  # cuda_stub.cc
  # nvml_stub.cc
set(_VER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/version.ld)

add_library(flux_cuda SHARED ${LIB_FILES})
set_target_properties(flux_cuda PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINK_DEPENDS ${_VER_FILE}
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(flux_cuda
  PUBLIC CUDA::cudart CUDA::cuda_driver ${FLUX_CUDA_OP_TARGETS}
  PRIVATE ${flux_proto_lib}
)

target_include_directories(flux_cuda PRIVATE ${flux_proto_inc})
target_link_options(flux_cuda
  PRIVATE
  $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs=libflux_proto:libprotobuf
                         LINKER:--version-script=${_VER_FILE}
                         LINKER:--no-as-needed>
)
install(TARGETS flux_cuda
    PUBLIC_HEADER DESTINATION include
)
