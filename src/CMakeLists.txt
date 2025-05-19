cmake_minimum_required(VERSION 3.17)

# append 'ctlop/src/' to include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# create an object library for each op subfolder
function(ctlop_add_op_cu_obj_lib TARGET FILES)
  add_library(${TARGET} OBJECT ${FILES})
  set_target_properties(${TARGET} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(${TARGET} PUBLIC CUDA::cudart ${CMAKE_THREAD_LIBS_INIT} )
  set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
endfunction()

set_property(GLOBAL PROPERTY CTLOP_THS_TARGETS)
function (ctlop_add_ths_op_target TARGET)
  get_property(CTLOP_THS_TARGETS_LOCAL GLOBAL PROPERTY CTLOP_THS_TARGETS)
  set_property(GLOBAL PROPERTY CTLOP_THS_TARGETS "${TARGET};${CTLOP_THS_TARGETS_LOCAL}")
endfunction()

#######################################################
set(CTLOP_CUDA_OP_TARGETS "") # cu op targets of subdirectory
set(CTLOP_THS_OP_FILES "") # ths op source files of subdirectory

add_subdirectory(ops/gemm_normal)
add_subdirectory(ops/allreduce_normal)
message(STATUS "op_targets: ${CTLOP_CUDA_OP_TARGETS}")

set(CTLOP_THS_FILES ${CTLOP_THS_OP_FILES})
message(STATUS "ths_files: ${CTLOP_THS_FILES}")

add_library(ctlop SHARED ${CTLOP_THS_FILES})
target_link_libraries(ctlop PUBLIC CUDA::cudart CUDA::cuda_driver ${CTLOP_CUDA_OP_TARGETS} ${TORCH_LIBRARIES})

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


