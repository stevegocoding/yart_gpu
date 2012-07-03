set(TARGET test_assimp_scene)
project(${TARGET})

set(CORE_FILES
	test_assimp_scene.cpp
	test_assimp_scene.h
	)

set(GPU_FILES
	test_assimp_scene.cu
	)

source_group("host" FILES ${CORE_FILES})
source_group("gpu" FILES ${GPU_FILES})
set(SRC_FILES 
	${CORE_FILES} 
	${GPU_FILES}
	)

add_definitions(-DUNICODE)

link_directories(${CMAKE_BINARY_DIR}/lib)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_20)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -ftz=true)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -prec-div=false)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -prec-sqrt=false)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -G0)

cuda_add_executable(${TARGET} 
		 ${SRC_FILES}
		 OPTIONS ${GENCODE_SM20} ${CUDA_DEBUG})
		 
set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX "_debug")
target_link_libraries(${TARGET} debug libyartgpu_d.lib)
target_link_libraries(${TARGET} optimized libyartgpu.lib)
target_link_libraries(${TARGET} debug cuda.lib)
target_link_libraries(${TARGET} optimized cuda.lib)
target_link_libraries(${TARGET} debug cudart.lib)
target_link_libraries(${TARGET} optimized cudart.lib)
target_link_libraries(${TARGET} debug assimp_d.lib)
target_link_libraries(${TARGET} optimized assimp.lib)

# Installation
install(TARGETS ${TARGET} RUNTIME DESTINATION bin/${TARGET})
install(DIRECTORY ${DEP_DLL_DEBUG_DIR} ${DEP_DLL_REL_DIR} DESTINATION bin/${TARGET})