cmake_minimum_required(VERSION 3.17)

set(OP_REGS "")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/registers")
# execute_process(
#   COMMAND ${FLUX_GENERATOR_BINARY_DIR}/gen_single_gemm "--dir=./registers" "--archs=${CUDAARCHS}"
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#   COMMAND_ERROR_IS_FATAL ANY
#   COMMAND_ECHO STDOUT
# )

# file(GLOB OP_REGS ${CMAKE_CURRENT_BINARY_DIR}/registers/*.cu)
# message(STATUS "OP_REGS: ${OP_REGS}")

file(GLOB TUNING_CONFIGS tuning_config/*.cu)
message(STATUS "TUNING_CONFIGS: ${TUNING_CONFIGS}")

set(CU_FILES
  ${OP_REGS}
  ${TUNING_CONFIGS}
)

set(LIB_NAME "xop_normal_gemm")
flux_add_op_cu_obj_lib(${LIB_NAME} "${CU_FILES}")
set(FLUX_CUDA_OP_TARGETS ${FLUX_CUDA_OP_TARGETS})
list(APPEND FLUX_CUDA_OP_TARGETS ${LIB_NAME})
set(FLUX_CUDA_OP_TARGETS ${FLUX_CUDA_OP_TARGETS} PARENT_SCOPE)

if (BUILD_THS)
  file(GLOB THS_FILES ./*.cpp)
  set(FLUX_THS_OP_FILES ${FLUX_THS_OP_FILES})
  list(APPEND FLUX_THS_OP_FILES ${THS_FILES})
  set(FLUX_THS_OP_FILES ${FLUX_THS_OP_FILES} PARENT_SCOPE)
  flux_add_ths_op_target("normal_gemm")
endif()