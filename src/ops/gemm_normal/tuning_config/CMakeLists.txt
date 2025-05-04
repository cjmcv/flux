cmake_minimum_required(VERSION 3.17)

set(OP_REGS "")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/registers")

file(GLOB TUNING_CONFIGS tuning_config/*.cu)
message(STATUS "TUNING_CONFIGS: ${TUNING_CONFIGS}")

set(CU_FILES
  ${OP_REGS}
  ${TUNING_CONFIGS}
)

set(LIB_NAME "ctlop_normal_gemm")
ctlop_add_op_cu_obj_lib(${LIB_NAME} "${CU_FILES}")
set(ctlop_CUDA_OP_TARGETS ${ctlop_CUDA_OP_TARGETS})
list(APPEND ctlop_CUDA_OP_TARGETS ${LIB_NAME})
set(ctlop_CUDA_OP_TARGETS ${ctlop_CUDA_OP_TARGETS} PARENT_SCOPE)

#############################
file(GLOB THS_FILES ./*.cpp)
set(ctlop_THS_OP_FILES ${ctlop_THS_OP_FILES})
list(APPEND ctlop_THS_OP_FILES ${THS_FILES})
set(ctlop_THS_OP_FILES ${ctlop_THS_OP_FILES} PARENT_SCOPE)
ctlop_add_ths_op_target("normal_gemm")
