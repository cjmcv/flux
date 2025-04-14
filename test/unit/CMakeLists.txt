cmake_minimum_required(VERSION 3.17)

link_libraries(CUDA::cudart CUDA::nvml ${CMAKE_THREAD_LIBS_INIT})

include_directories(${COMMON_HEADER_DIRS})

add_executable(test_flux_templates test_flux_templates.cc)
target_link_libraries(test_flux_templates PUBLIC ${CMAKE_THREAD_LIBS_INIT})

add_executable(test_tuning test_tuning.cc)
target_link_libraries(test_tuning PUBLIC flux_cuda)

add_executable(test_gemm_only test_gemm_only.cc)
target_link_libraries(test_gemm_only PUBLIC flux_cuda)