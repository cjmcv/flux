cmake_minimum_required(VERSION 3.17)

file(GLOB CU_FILES instantiations/*.cu)
message(STATUS "CU_FILES: ${CU_FILES}")

#########################
file(GLOB THS_FILES ./*.cpp)
set(CTLOP_THS_OP_FILES ${CTLOP_THS_OP_FILES})
list(APPEND CTLOP_THS_OP_FILES ${THS_FILES} "${CU_FILES}")
set(CTLOP_THS_OP_FILES ${CTLOP_THS_OP_FILES} PARENT_SCOPE)
ctlop_add_ths_op_target("flash_attn")
