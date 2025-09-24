cmake_minimum_required(VERSION 3.17)

# append 'xop/src/' to include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set_property(GLOBAL PROPERTY XOP_THS_TARGETS)
function (xop_add_ths_op_target TARGET)
  get_property(XOP_THS_TARGETS_LOCAL GLOBAL PROPERTY XOP_THS_TARGETS)
  set_property(GLOBAL PROPERTY XOP_THS_TARGETS "${TARGET};${XOP_THS_TARGETS_LOCAL}")
endfunction()

#######################################################
set(XOP_THS_OP_FILES "") # ths op source files of subdirectory

# add_subdirectory(ops/gemm_normal)
# add_subdirectory(ops/marlin)
add_subdirectory(ops/gemm_comm)
add_subdirectory(ops/allreduce_normal)
# add_subdirectory(ops/flash_attn)

set(XOP_THS_FILES ${XOP_THS_OP_FILES})
message(STATUS "ths_files: ${XOP_THS_FILES}")

add_library(xop SHARED ${XOP_THS_FILES})
target_link_libraries(xop PUBLIC CUDA::cudart CUDA::cuda_driver ${TORCH_LIBRARIES})

install(TARGETS xop
    PUBLIC_HEADER DESTINATION include
)
#########################################

include(GNUInstallDirs)
install(
  DIRECTORY ./ "${PROJECT_SOURCE_DIR}/include/xop/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xop/
  FILES_MATCHING PATTERN "*/ops_impl/*.h")

file(GLOB XOP_HEADERS ${PROJECT_SOURCE_DIR}/include/xop/*.h)
install(
  FILES ${XOP_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xop/
)


