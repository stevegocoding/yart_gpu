PROJECT(yart_gpu)

cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_SOURCE_DIR}/src/core/)
include_directories(${PROJECT_SOURCE_DIR}/deps/include/)

link_directories(${PROJECT_SOURCE_DIR}/deps/lib/debug)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib/release)

SET(BOOST_ROOT_DIR "" CACHE PATH "")
SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} ${BOOST_ROOT_DIR})
SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${BOOST_ROOT_DIR}/lib)

find_package(BOOST)
if (Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIR})
	add_definitions("-DHAS_BOOST")
endif()

#Searching CUDA
find_package(CUDA)
if (CUDA_FOUND)
	include_directories(${CUDA_SDK_ROOT_DIR}/shared/inc)
	include_directories(${CUDA_SDK_ROOT_DIR}/C/common/inc)
	include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

	CUDA_INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/include)
	CUDA_INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/shared/inc)
	CUDA_INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/C/common/inc)

	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32)
endif()

FIND_PACKAGE(wxWidgets COMPONENTS core base gl)
if (wxWidgets_FOUND)
	include_directories(${wxWidgets_INCLUDE_DIRS})
	SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${wxWidgets_LIB_DIR})
endif()

set(DEP_DLL_DEBUG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/bin/debug/)
set(DEP_DLL_REL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/bin/release/)



#Adding src directory to the project
add_subdirectory(data)
add_subdirectory(src)