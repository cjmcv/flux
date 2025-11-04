cmake_minimum_required(VERSION 3.17)

# file(GLOB TUNING_CONFIGS tuning_config/*.cu)
set(TUNING_CONFIGS
    ${CMAKE_CURRENT_LIST_DIR}/tuning_config/search_space_gemmsm80.cu
    ${CMAKE_CURRENT_LIST_DIR}/tuning_config/search_space_gemmsimtsm80.cu
    ${CMAKE_CURRENT_LIST_DIR}/tuning_config/search_space_gemmsm90.cu 
    #
    ${CMAKE_CURRENT_LIST_DIR}/tuning_config/search_space_gemmblockscalefp8sm89.cu
    ${CMAKE_CURRENT_LIST_DIR}/tuning_config/search_space_gemmblockscalefp8sm90.cu
    #
    ${CMAKE_CURRENT_LIST_DIR}/tuning_config/search_space_gemmgroupedsm80.cu
    ${CMAKE_CURRENT_LIST_DIR}/tuning_config/search_space_gemmgroupedblockscalefp8sm90.cu
    #
    ${CMAKE_CURRENT_LIST_DIR}/tuning_config/search_space_gemmw4a16sm90.cu
    ### tuning ###
    # ${CMAKE_CURRENT_LIST_DIR}/tuning_config/tuned_config_gemmsm80.cu
    # ${CMAKE_CURRENT_LIST_DIR}/tuning_config/tuned_config_gemmsm90.cu
    # ${CMAKE_CURRENT_LIST_DIR}/tuning_config/tuned_config_gemmw4a16sm90.cu
)
message(STATUS "TUNING_CONFIGS: ${TUNING_CONFIGS}")

set(CU_FILES
  ${TUNING_CONFIGS}
  ${CMAKE_CURRENT_LIST_DIR}/gemm_preprocess.cu
)

#########################
file(GLOB THS_FILES ./*.cpp)
set(XOP_THS_OP_FILES ${XOP_THS_OP_FILES})
list(APPEND XOP_THS_OP_FILES ${THS_FILES} "${CU_FILES}")
set(XOP_THS_OP_FILES ${XOP_THS_OP_FILES} PARENT_SCOPE)
xop_add_ths_op_target("gemm_normal")
