set(TARGET yart_gpu_core)
set(LIB_TARGET libyartgpu)

set(DEP_LIBS_DEBUG
	assimp_d.lib
	cutil32D.lib
	shrUtils32D.lib
	)

set(DEP_LIBS_RELEASE
	assimp.lib
	cutil32.lib
	shrUtils32.lib
	)

set(CORE_FILES
	scene.cpp
	scene.h
	)

set(GPU_FILES
	vec_add.cu
	)

source_group("core" FILES ${CORE_FILES})
source_group("gpu" FILES ${GPU_FILES})

set(SRC_FILES 
	${CORE_FILES} 
	${GPU_FILES}
	)

cuda_add_library(${LIB_TARGET} ${SRC_FILES} STATIC)
set_target_properties(${LIB_TARGET} PROPERTIES DEBUG_POSTFIX "_d") 
target_link_libraries(${LIB_TARGET} debug "${DEP_LIBS_DEBUG}")
target_link_libraries(${LIB_TARGET} optimized "${DEP_LIBS_RELEASE}")
install(TARGETS ${LIB_TARGET} ARCHIVE DESTINATION lib/)