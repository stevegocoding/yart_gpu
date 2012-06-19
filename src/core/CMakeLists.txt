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

set(CUDA_MT_FILES
	cuda_mt/MersenneTwister_kernel.cu
	cuda_mt/MersenneTwister.dat
	cuda_mt/MersenneTwister.h
	)

set(CORE_FILES
	camera.cpp
	camera.h
	cuda_defs.h
	cuda_mem_pool.cpp
	cuda_mem_pool.h
	cuda_rng.cpp
	cuda_rng.h
	cuda_utils.cpp
	cuda_utils.h
	math_utils.cpp
	math_utils.h
	matrix44f.cpp
	matrix44f.h
	normal3f.cpp
	normal3f.h
	point3f.cpp
	point3f.h
	vector3f.cpp
	vector3f.h
	prerequisites.h
	ray.cpp
	ray.h
	ray_tracer.cpp
	ray_tracer.h
	render_target.cpp
	render_target.h
	sampler.cpp
	sampler.h
	scene.cpp
	scene.h
	transform.cpp
	transform.h
	main.cpp
	)

set(GPU_FILES
	ray_gen.cu
	vec_add.cu
	)

source_group("cuda_mt" FILES ${CUDA_MT_FILES})
source_group("core" FILES ${CORE_FILES})
source_group("gpu" FILES ${GPU_FILES})

set(SRC_FILES 
	${CUDA_MT_FILES}
	${CORE_FILES} 
	${GPU_FILES}
	)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

#cuda_add_library(${LIB_TARGET} ${LIB_TYPE} ${SRC_FILES})
cuda_add_executable(${LIB_TARGET} ${SRC_FILES})

set_target_properties(${LIB_TARGET} PROPERTIES DEBUG_POSTFIX "_d") 
target_link_libraries(${LIB_TARGET} debug "${DEP_LIBS_DEBUG}")
target_link_libraries(${LIB_TARGET} optimized "${DEP_LIBS_RELEASE}")
#install(TARGETS ${LIB_TARGET} ARCHIVE DESTINATION lib/)