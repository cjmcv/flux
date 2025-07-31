cmake_minimum_required(VERSION 3.17)

file(GLOB TUNING_CONFIGS tuning_config/*.cu)
message(STATUS "TUNING_CONFIGS: ${TUNING_CONFIGS}")

set(CU_FILES
  ${TUNING_CONFIGS}
)

#########################
file(GLOB THS_FILES ./*.cpp)
set(XOP_THS_OP_FILES ${XOP_THS_OP_FILES})
list(APPEND XOP_THS_OP_FILES ${THS_FILES} "${CU_FILES}")
set(XOP_THS_OP_FILES ${XOP_THS_OP_FILES} PARENT_SCOPE)
xop_add_ths_op_target("gemm_normal")
