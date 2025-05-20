cmake_minimum_required(VERSION 3.17)

# append 'ctlop/src/' to include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set_property(GLOBAL PROPERTY CTLOP_THS_TARGETS)
function (ctlop_add_ths_op_target TARGET)
  get_property(CTLOP_THS_TARGETS_LOCAL GLOBAL PROPERTY CTLOP_THS_TARGETS)
  set_property(GLOBAL PROPERTY CTLOP_THS_TARGETS "${TARGET};${CTLOP_THS_TARGETS_LOCAL}")
endfunction()

#######################################################
set(CTLOP_THS_OP_FILES "") # ths op source files of subdirectory

add_subdirectory(ops/gemm_normal)
add_subdirectory(ops/allreduce_normal)

set(CTLOP_THS_FILES ${CTLOP_THS_OP_FILES})
message(STATUS "ths_files: ${CTLOP_THS_FILES}")

add_library(ctlop SHARED ${CTLOP_THS_FILES})
target_link_libraries(ctlop PUBLIC CUDA::cudart CUDA::cuda_driver ${TORCH_LIBRARIES})

install(TARGETS ctlop
    PUBLIC_HEADER DESTINATION include
)
#########################################

include(GNUInstallDirs)
install(
  DIRECTORY ./ "${PROJECT_SOURCE_DIR}/include/ctlop/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ctlop/
  FILES_MATCHING PATTERN "*/ops_impl/*.h")

file(GLOB CTLOP_HEADERS ${PROJECT_SOURCE_DIR}/include/ctlop/*.h)
install(
  FILES ${CTLOP_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ctlop/
)


