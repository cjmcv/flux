cmake_minimum_required(VERSION 3.17)

link_libraries(CUDA::cudart CUDA::nvml ${CMAKE_THREAD_LIBS_INIT})

include_directories(${COMMON_HEADER_DIRS})

add_executable(test_flux_templates test_flux_templates.cc)
target_link_libraries(test_flux_templates PUBLIC ${CMAKE_THREAD_LIBS_INIT})

add_executable(test_tuning test_tuning.cc)
target_link_libraries(test_tuning PUBLIC flux_cuda_ths_op)

add_executable(test_single_gemm test_single_gemm.cc)
target_link_libraries(test_single_gemm PUBLIC flux_cuda_ths_op)

add_executable(test_gemv_only test_gemv_only.cu)
target_link_libraries(test_gemv_only PUBLIC flux_cuda_ths_op)

add_executable(test_gemm_simt test_gemm_simt.cu)
target_link_libraries(test_gemm_simt PUBLIC flux_cuda_ths_op)

add_executable(test_dual_gemm dual_gemm/dual_gemm.cu)
target_link_libraries(test_dual_gemm PUBLIC flux_cuda_ths_op)