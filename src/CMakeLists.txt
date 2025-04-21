cmake_minimum_required(VERSION 3.17)

# append 'flux/src/' to include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# create an object library for each op subfolder
function(flux_add_op_cu_obj_lib TARGET FILES)
  add_library(${TARGET} OBJECT ${FILES})
  set_target_properties(${TARGET} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(${TARGET} PUBLIC CUDA::cudart ${CMAKE_THREAD_LIBS_INIT} )
  set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
endfunction()

set_property(GLOBAL PROPERTY FLUX_THS_TARGETS)
function (flux_add_ths_op_target TARGET)
  get_property(FLUX_THS_TARGETS_LOCAL GLOBAL PROPERTY FLUX_THS_TARGETS)
  set_property(GLOBAL PROPERTY FLUX_THS_TARGETS "${TARGET};${FLUX_THS_TARGETS_LOCAL}")
endfunction()

set(FLUX_CUDA_OP_TARGETS "") # cu op targets of subdirectory
if (BUILD_THS)
  set(FLUX_THS_OP_FILES "") # ths op source files of subdirectory
endif()

include(ExternalProject)
ExternalProject_Add(
  FLUX_GENERATOR
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generator
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""  # Prevents automatic build during build phase
  INSTALL_COMMAND ""  # Prevents automatic install
)
ExternalProject_Get_property(FLUX_GENERATOR SOURCE_DIR)
ExternalProject_Get_property(FLUX_GENERATOR BINARY_DIR)
set(FLUX_GENERATOR_BINARY_DIR ${BINARY_DIR})
set(FLUX_GENERATOR_SOURCE_DIR ${SOURCE_DIR})

execute_process(
  COMMAND ${CMAKE_COMMAND} ${FLUX_GENERATOR_SOURCE_DIR}
  WORKING_DIRECTORY ${FLUX_GENERATOR_BINARY_DIR}
  COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND ${CMAKE_COMMAND} --build . --parallel 8
  WORKING_DIRECTORY ${FLUX_GENERATOR_BINARY_DIR}
  COMMAND_ERROR_IS_FATAL ANY
)

add_subdirectory(comm_none)
message(STATUS "op_targets: ${FLUX_CUDA_OP_TARGETS}")

# this must be after all op subdirectory added
# add_subdirectory(cuda)

# if (BUILD_THS)
#   add_subdirectory(ths_op)
# endif()

############ flux_cuda_ths_op ################
set(FLUX_THS_FILES ${FLUX_THS_OP_FILES})
message(STATUS "ths_files: ${FLUX_THS_FILES}") # comm_none/gemm_only.cc

add_library(flux_cuda_ths_op SHARED ${FLUX_THS_FILES})
target_link_libraries(flux_cuda_ths_op PUBLIC CUDA::cudart CUDA::cuda_driver ${FLUX_CUDA_OP_TARGETS} ${TORCH_LIBRARIES})

# Write the unchached variable to a file
get_property(FLUX_THS_TARGETS GLOBAL PROPERTY FLUX_THS_TARGETS)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/flux_ths_targets.txt" "FLUX_THS_TARGETS=ths_op;${FLUX_THS_TARGETS}\n")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

install(TARGETS flux_cuda_ths_op
    PUBLIC_HEADER DESTINATION include
)
#########################################

include(GNUInstallDirs)
install(
  DIRECTORY ./ "${PROJECT_SOURCE_DIR}/include/flux/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flux/
  FILES_MATCHING PATTERN "*/ths_op/*.h")

file(GLOB FLUX_HEADERS ${PROJECT_SOURCE_DIR}/include/flux/*.h)
install(
  FILES ${FLUX_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flux/
)


