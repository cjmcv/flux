cmake_minimum_required(VERSION 3.17)

file(GLOB CU_FILES *.cu)
message(STATUS "CU_FILES: ${CU_FILES}")

# set(LIB_NAME "ctlop_allreduce_normal")
# ctlop_add_op_cu_obj_lib(${LIB_NAME} )
# set(CTLOP_CUDA_OP_TARGETS ${CTLOP_CUDA_OP_TARGETS})
# list(APPEND CTLOP_CUDA_OP_TARGETS ${LIB_NAME})
# set(CTLOP_CUDA_OP_TARGETS ${CTLOP_CUDA_OP_TARGETS} PARENT_SCOPE)

#########################
file(GLOB THS_FILES ./*.cpp)
set(CTLOP_THS_OP_FILES ${CTLOP_THS_OP_FILES})
list(APPEND CTLOP_THS_OP_FILES ${THS_FILES} "${CU_FILES}")
set(CTLOP_THS_OP_FILES ${CTLOP_THS_OP_FILES} PARENT_SCOPE)
ctlop_add_ths_op_target("allreduce_normal")
