cmake_minimum_required(VERSION 3.24)

project(Fluora CXX CUDA C)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++20 for host code
set(CMAKE_CXX_STANDARD 20)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
endif()

########################################
# CUDA Setup
########################################
find_package(CUDA 10 REQUIRED)
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
set(CUDA_VERBOSE_BUILD ON)

if(WIN32)
    # Set up include and lib paths
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
endif(WIN32)
########################################

find_package(OpenGL REQUIRED)

if(UNIX)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY})
else(UNIX)
    set(EXTERNAL "external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

file(GLOB_RECURSE headers 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

set(sources
    src/main.cpp
    src/stb.cpp
    src/image.cpp
    src/glslUtility.cpp
    src/mathUtils.cu
    src/bsdf.cu
    src/color.cu
    src/pathtrace.cu
    src/camera.cu
    src/materials.cu
    src/microfacet.cu
    src/scene.cpp
    src/preview.cpp
    src/utilities.cpp
    src/bvh.cpp
    src/spectrum.cu
    src/intersections.cu
    src/sampling.cu
    src/media.cu
	src/mikktspace/mikktspace.c
    src/ImGui/imgui.cpp 
    src/ImGui/imgui_demo.cpp 
    src/ImGui/imgui_draw.cpp 
    src/ImGui/imgui_impl_glfw.cpp 
    src/ImGui/imgui_impl_opengl3.cpp 
    src/ImGui/imgui_tables.cpp 
    src/ImGui/imgui_widgets.cpp 
    src/SpectrumConsts/rgbspectrum_aces.cpp
    src/SpectrumConsts/rgbspectrum_dci_p3.cpp
    src/SpectrumConsts/rgbspectrum_rec2020.cpp
    src/SpectrumConsts/rgbspectrum_srgb.cpp
    )


set (NANOVDB_INCLUDE src/openvdb/nanovdb)
include_directories(${NANOVDB_INCLUDE})

list(SORT headers)
list(SORT sources)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Headers" FILES ${headers})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Sources" FILES ${sources})


link_directories(${PROJECT_SOURCE_DIR}/external/lib)
find_package(CUDAToolkit REQUIRED)
add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES native)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${LIBRARIES}
    cudadevrt
    OpenImageDenoise
)

if (WIN32)
    file(GLOB DLL_FILES 
        "${PROJECT_SOURCE_DIR}/external/bin/*.dll"
    )
    foreach(DLL_FILE ${DLL_FILES})
        add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${DLL_FILE} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>)
    endforeach()
endif()