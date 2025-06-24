cmake_minimum_required(VERSION 3.17)

file(GLOB CU_FILES instantiations/*.cu *.cu)
# file(GLOB CU_FILES instantiations/flash_fwd_hdim64_256_bf16_packgqa_sm90.cu *.cu)
# file(GLOB CU_FILES instantiations/flash_fwd_hdim64_bf16_sm80.cu *.cu)

message(STATUS "CU_FILES: ${CU_FILES}")

#########################
file(GLOB THS_FILES ./*.cpp)
set(XOP_THS_OP_FILES ${XOP_THS_OP_FILES})
list(APPEND XOP_THS_OP_FILES ${THS_FILES} "${CU_FILES}")
set(XOP_THS_OP_FILES ${XOP_THS_OP_FILES} PARENT_SCOPE)
xop_add_ths_op_target("flash_attn")
