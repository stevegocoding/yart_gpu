set(TARGET test_raygen)
project(${TARGET})

set(SRCS test_raygen.cpp)

add_definitions(-DUNICODE)

link_directories(${CMAKE_BINARY_DIR}/lib)


add_executable(${TARGET} ${SRCS})
set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX "_debug")
target_link_libraries(${TARGET} debug libyartgpu_d.lib)
target_link_libraries(${TARGET} optimized libyartgpu.lib)
target_link_libraries(${TARGET} debug cuda.lib)
target_link_libraries(${TARGET} optimized cuda.lib)
target_link_libraries(${TARGET} debug cudart.lib)
target_link_libraries(${TARGET} optimized cudart.lib)

# Installation
install(TARGETS ${TARGET} RUNTIME DESTINATION bin/${TARGET})