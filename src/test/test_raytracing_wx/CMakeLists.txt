set(TARGET test_raytracing_wx)
project(${TARGET})

set(UI_FILES
	app.cpp
	app.h
	cuda_canvas.cpp
	cuda_canvas.h
	main_frame.cpp
	main_frame.h
	wx_utils.cpp
	wx_utils.h
	)

if(WIN32)
  set(RES_FILES minimal.rc)
endif(WIN32) 

source_group("source" FILES ${UI_FILES})
source_group("res" FILES ${RES_FILES})

add_definitions(-DUNICODE)

include(${wxWidgets_USE_FILE})

set(SRC_FILES 
	${UI_FILES} 
	${RES_FILES}
	)

link_directories(${CMAKE_BINARY_DIR}/lib)

add_executable(${TARGET} WIN32 ${SRC_FILES})
set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX "_debug")

target_link_libraries(${TARGET} debug libyartgpu_d.lib ${wxWidgets_LIBRARIES})
target_link_libraries(${TARGET} optimized libyartgpu.lib ${wxWidgets_LIBRARIES})
target_link_libraries(${TARGET} debug assimp_d.lib)
target_link_libraries(${TARGET} optimized assimp.lib)
target_link_libraries(${TARGET} debug glew32.lib)
target_link_libraries(${TARGET} optimized glew32.lib)
target_link_libraries(${TARGET} debug DevIL.lib)
target_link_libraries(${TARGET} optimized DevIL.lib)
target_link_libraries(${TARGET} debug ILU.lib)
target_link_libraries(${TARGET} optimized ILU.lib)
target_link_libraries(${TARGET} debug ILUT.lib)
target_link_libraries(${TARGET} optimized ILUT.lib)
target_link_libraries(${TARGET} debug cuda.lib)
target_link_libraries(${TARGET} optimized cuda.lib)
target_link_libraries(${TARGET} debug cudart.lib)
target_link_libraries(${TARGET} optimized cudart.lib)

# Installation
install(TARGETS ${TARGET} RUNTIME DESTINATION bin/${TARGET})
install(DIRECTORY ${DEP_DLL_DEBUG_DIR} ${DEP_DLL_REL_DIR} DESTINATION bin/${TARGET})