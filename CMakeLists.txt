#The name of the project
PROJECT(yart_gpu)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#Searching CUDA
FIND_PACKAGE(CUDA)

FIND_PACKAGE(wxWidgets COMPONENTS core base gl)
if (wxWidgets_FOUND)
	include_directories(${wxWidgets_INCLUDE_DIRS})
	SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${wxWidgets_LIB_DIR})
endif()

set(DEP_DLL_DEBUG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/bin/debug/)
set(DEP_DLL_REL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/bin/release/)

include_directories(${CMAKE_SOURCE_DIR}/src/core/)
include_directories(${PROJECT_SOURCE_DIR}/deps/include/)
include_directories(${CUDA_SDK_ROOT_DIR}/shared/inc)
include_directories(${CUDA_SDK_ROOT_DIR}/C/common/inc)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

link_directories(${PROJECT_SOURCE_DIR}/deps/lib/debug)
link_directories(${PROJECT_SOURCE_DIR}/deps/lib/release)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32)

CUDA_INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/include)
CUDA_INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/shared/inc)
CUDA_INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/C/common/inc)

#Adding src directory to the project
ADD_SUBDIRECTORY(src)