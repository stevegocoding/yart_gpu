set(TARGET test_glew)
project(${TARGET})

set(UI_FILES
	main.cpp
	)

source_group("source" FILES ${UI_FILES})
source_group("res" FILES ${RES_FILES})

add_definitions(-DUNICODE)

set(SRC_FILES 
	${UI_FILES} 
	)

link_directories(${CMAKE_BINARY_DIR}/lib)

add_executable(${TARGET} ${SRC_FILES})
set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX "_debug")

target_link_libraries(${TARGET} debug glew32.lib)
target_link_libraries(${TARGET} optimized glew32.lib)

# Installation
install(TARGETS ${TARGET} RUNTIME DESTINATION bin/${TARGET})
install(DIRECTORY ${DEP_DLL_DEBUG_DIR} ${DEP_DLL_REL_DIR} DESTINATION bin/${TARGET})