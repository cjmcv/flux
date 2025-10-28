cmake_minimum_required(VERSION 3.17)

link_libraries(CUDA::cudart CUDA::nvml ${CMAKE_THREAD_LIBS_INIT})

include_directories(${COMMON_HEADER_DIRS})

# add_executable(test_tuning test_tuning.cc)
# target_link_libraries(test_tuning PUBLIC xop)

# add_executable(test_single_gemm test_single_gemm.cc)
# target_link_libraries(test_single_gemm PUBLIC xop)

# add_executable(test_gemv_only test_gemv_only.cu)
# target_link_libraries(test_gemv_only PUBLIC xop)

# add_executable(test_gemm_simt test_gemm_simt.cu)
# target_link_libraries(test_gemm_simt PUBLIC xop)

# add_executable(test_dual_gemm dual_gemm/dual_gemm.cu)
# target_link_libraries(test_dual_gemm PUBLIC xop)

# # ## fused_two_gemms 对shape有严格限制，ThreadblockShape::kN = problem_size.N，不符合实际应用情况
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/13_two_tensor_op_fusion)

# add_executable(test_fused_two_gemms_f16_sm80_rf 13_two_tensor_op_fusion/fused_two_gemms_f16_sm80_rf.cu)
# target_link_libraries(test_fused_two_gemms_f16_sm80_rf PUBLIC xop)

# add_executable(test_fused_two_gemms_f16_sm80_shmem 13_two_tensor_op_fusion/fused_two_gemms_f16_sm80_shmem.cu)
# target_link_libraries(test_fused_two_gemms_f16_sm80_shmem PUBLIC xop)
# # ## 

# ##

# add_executable(gemm_multi_stage cute_practice/gemm-multi-stage.cu)
# target_link_libraries(gemm_multi_stage PUBLIC xop)
# add_executable(gemm-simple cute_practice/gemm-simple.cu)
# target_link_libraries(gemm-simple PUBLIC xop)

# add_executable(mma_layout cute_practice/mma_layout.cu)
# target_link_libraries(mma_layout PUBLIC xop)
# add_executable(sgemm_sm80 cute_practice/sgemm_sm80.cu)
# target_link_libraries(sgemm_sm80 PUBLIC xop)
# add_executable(tiled_copy cute_practice/tiled_copy.cu)
# target_link_libraries(tiled_copy PUBLIC xop)
# add_executable(my_sgemm cute_practice/my_sgemm.cu)
# target_link_libraries(my_sgemm PUBLIC xop)

# add_executable(cutlass_unittest cutlass_unittest/minitest.cu)
# target_link_libraries(cutlass_unittest PUBLIC xop)

# add_executable(71_blackwell_gemm 71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu)
# target_link_libraries(71_blackwell_gemm PUBLIC xop)

# add_executable(79_blackwell_geforce_gemm 79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
# set_target_properties(79_blackwell_geforce_gemm PROPERTIES
#     CUDA_ARCHITECTURES "120"
# )
# target_link_libraries(79_blackwell_geforce_gemm PUBLIC xop)

# add_executable(49_hopper_gemm 49_hopper_gemm_with_collective_builder/49_collective_builder.cu)
# target_link_libraries(49_hopper_gemm PUBLIC xop)

# add_executable(58_ada_fp8_gemm 58_ada_fp8_gemm/ada_fp8_gemm.cu)
# target_link_libraries(58_ada_fp8_gemm PUBLIC xop)

# add_executable(ada_blockwise_gemm ada_blockwise_gemm/ada_fp8_gemm.cu)
# target_link_libraries(ada_blockwise_gemm PUBLIC xop)

# add_executable(65_distributed_gemm 65_distributed_gemm/65_distributed_gemm.cu)
# target_link_libraries(65_distributed_gemm PUBLIC xop)

# add_executable(67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling 67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu)
# target_link_libraries(67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling PUBLIC xop)

# add_executable(68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling 68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling.cu)
# target_link_libraries(68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling PUBLIC xop)

# add_executable(68_mini_test 68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling/68_mini_test.cu)
# target_link_libraries(68_mini_test PUBLIC xop)
# #
# add_executable(37_gemm_layernorm_gemm_fusion 37_gemm_layernorm_gemm_fusion/gemm_layernorm.cu)
# target_link_libraries(37_gemm_layernorm_gemm_fusion PUBLIC xop)

# add_executable(gemm_grouped 24_gemm_grouped/gemm_grouped.cu)
# target_link_libraries(gemm_grouped PUBLIC xop)

add_executable(55_hopper_int4_bf16_gemm 55_hopper_mixed_dtype_gemm/55_quant_dequant.cu) # 55_hopper_int4_bf16_gemm
target_link_libraries(55_hopper_int4_bf16_gemm PUBLIC xop)

##

# add_executable(pure_test pure_test.cu)# config_normal_gemm_sm89.cu
# target_link_libraries(pure_test PUBLIC xop)