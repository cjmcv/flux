cmake_minimum_required(VERSION 3.17)

link_libraries(CUDA::cudart CUDA::nvml ${CMAKE_THREAD_LIBS_INIT})

include_directories(${COMMON_HEADER_DIRS})

# add_executable(test_flux_templates test_flux_templates.cc)
# target_link_libraries(test_flux_templates PUBLIC ${CMAKE_THREAD_LIBS_INIT})

# add_executable(test_tuning test_tuning.cc)
# target_link_libraries(test_tuning PUBLIC flux_cuda_ths_op)

# add_executable(test_single_gemm test_single_gemm.cc)
# target_link_libraries(test_single_gemm PUBLIC flux_cuda_ths_op)

# add_executable(test_gemv_only test_gemv_only.cu)
# target_link_libraries(test_gemv_only PUBLIC flux_cuda_ths_op)

# add_executable(test_gemm_simt test_gemm_simt.cu)
# target_link_libraries(test_gemm_simt PUBLIC flux_cuda_ths_op)

# add_executable(test_dual_gemm dual_gemm/dual_gemm.cu)
# target_link_libraries(test_dual_gemm PUBLIC flux_cuda_ths_op)

# ## fused_two_gemms
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/two_tensor_op_fusion)

# # add_executable(test_fused_two_gemms_grouped_f16_sm80_rf two_tensor_op_fusion/fused_two_gemms_grouped_f16_sm80_rf.cu)
# # target_link_libraries(test_fused_two_gemms_grouped_f16_sm80_rf PUBLIC flux_cuda_ths_op)

# add_executable(test_fused_two_gemms_f16_sm80_rf two_tensor_op_fusion/fused_two_gemms_f16_sm80_rf.cu)
# target_link_libraries(test_fused_two_gemms_f16_sm80_rf PUBLIC flux_cuda_ths_op)

# add_executable(test_fused_two_gemms_f16_sm80_shmem two_tensor_op_fusion/fused_two_gemms_f16_sm80_shmem.cu)
# target_link_libraries(test_fused_two_gemms_f16_sm80_shmem PUBLIC flux_cuda_ths_op)

# # add_executable(test_fused_two_gemms_s8_sm80_rf two_tensor_op_fusion/fused_two_gemms_s8_sm80_rf.cu)
# # target_link_libraries(test_fused_two_gemms_s8_sm80_rf PUBLIC flux_cuda_ths_op)

# # add_executable(test_fused_two_gemms_s8_sm80_shmem two_tensor_op_fusion/fused_two_gemms_s8_sm80_shmem.cu)
# # target_link_libraries(test_fused_two_gemms_s8_sm80_shmem PUBLIC flux_cuda_ths_op)
# ## 

# ##
# add_executable(67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling 67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu)
# target_link_libraries(67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling PUBLIC flux_cuda_ths_op)

# add_executable(68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling 68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling.cu)
# target_link_libraries(68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling PUBLIC flux_cuda_ths_op)
# #
# add_executable(37_gemm_layernorm_gemm_fusion 37_gemm_layernorm_gemm_fusion/gemm_layernorm.cu)
# target_link_libraries(37_gemm_layernorm_gemm_fusion PUBLIC flux_cuda_ths_op)
##

add_executable(pure_test pure_test.cu)# config_normal_gemm_sm89.cu
target_link_libraries(pure_test PUBLIC flux_cuda_ths_op)