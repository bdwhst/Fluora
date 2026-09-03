cmake_minimum_required(VERSION 3.24)

# CUDA is enabled below only on non-Apple platforms; macOS builds the Metal
# vertical slice (FluoraMini) instead. See docs/metal-rhi-design.md.
project(Fluora CXX C)

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

# (Debug device-debug flags are appended after enable_language(CUDA) below:
# setting CMAKE_CUDA_FLAGS before it shadows the compiler-detected defaults.)

########################################
# macOS: Metal vertical slice only (milestone M1)
########################################
if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 20)
    # Dear ImGui + its Metal renderer backend power the FluoraMini overlay GUI.
    # The portable widget/controller code lives in src/core/gui and is shared
    # with the (future) Windows preview; only the backend wiring is here.
    set(FLUORA_IMGUI_SOURCES
        src/ImGui/imgui.cpp
        src/ImGui/imgui_draw.cpp
        src/ImGui/imgui_tables.cpp
        src/ImGui/imgui_widgets.cpp
        src/ImGui/imgui_impl_metal.mm)
    add_executable(FluoraMini
        src/mini/mini_main.cpp
        src/core/scene_loader.cpp
        src/core/mesh_loader.cpp
        src/core/bvh_builder.cpp
        src/core/image_loader.cpp
        src/core/spectra.cpp
        src/core/gui/gui.cpp
        src/SpectrumConsts/rgbspectrum_srgb.cpp
        src/rhi/rhi_metal.mm
        src/stb.cpp
        ${FLUORA_IMGUI_SOURCES}
    )
    target_include_directories(FluoraMini PRIVATE external/include src/ImGui)
    # MSL sources are loaded from the source tree at runtime (dev convenience;
    # packaging deferred to M5)
    target_compile_definitions(FluoraMini PRIVATE
        MINI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/mini"
        CORE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/core"
        RHI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/rhi"
        FLUORA_SCENES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/scenes"
        RHI_ENABLE_IMGUI)
    # rhi_metal.mm and the ImGui Metal backend are Objective-C++ under ARC.
    set_source_files_properties(src/rhi/rhi_metal.mm src/ImGui/imgui_impl_metal.mm
        PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(FluoraMini "-framework Metal" "-framework Foundation"
        "-framework AppKit" "-framework QuartzCore")

    # RHI primitive unit tests (reduce/scan/compact/sort/work queue)
    add_executable(RhiTest
        src/test/rhi_test.cpp
        src/rhi/rhi_algorithms.cpp
        src/rhi/rhi_metal.mm
    )
    target_compile_definitions(RhiTest PRIVATE RHI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/rhi")
    target_link_libraries(RhiTest "-framework Metal" "-framework Foundation"
        "-framework AppKit" "-framework QuartzCore")

    # Dual-personality value test for the portable device-code shim: the same
    # shared headers compiled as host C++ (the CUDA-side spelling) and as MSL,
    # asserting numeric agreement (docs/portable-device-code.md).
    add_executable(SharedHostTest
        src/test/shared_host_test.cpp
        src/core/spectra.cpp
        src/SpectrumConsts/rgbspectrum_srgb.cpp
        src/rhi/rhi_metal.mm
    )
    target_include_directories(SharedHostTest PRIVATE external/include)
    target_compile_definitions(SharedHostTest PRIVATE
        CORE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/core"
        RHI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/rhi"
        TEST_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/test")
    target_link_libraries(SharedHostTest "-framework Metal" "-framework Foundation"
        "-framework AppKit" "-framework QuartzCore")
    return()
endif()

########################################
# CUDA Setup
########################################
enable_language(CUDA)
find_package(CUDA 10 REQUIRED)
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
set(CUDA_VERBOSE_BUILD ON)

# CUDA 13's CCCL headers require MSVC's standards-conforming preprocessor.
# Apply /Zc:preprocessor to both host C++ code and nvcc's host-compiler pass.
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()

# Stop <windows.h> from defining min/max macros that collide with
# std::numeric_limits<T>::max(), Thrust internals, and std::min/std::max.
if(WIN32)
    add_compile_definitions(NOMINMAX)
endif()

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

# zlib (required because media.h enables NANOVDB_USE_ZIP for compressed .nvdb files)
find_library(ZLIB_STATIC_LIB
    NAMES zlibstatic zlibstaticd zlib
    PATHS "${PROJECT_SOURCE_DIR}/external/lib"
    NO_DEFAULT_PATH
)
if(NOT ZLIB_STATIC_LIB)
    message(FATAL_ERROR "zlib static library not found in external/lib. "
        "Drop zlibstatic.lib into external/lib and zlib.h + zconf.h into external/include.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/include/zlib.h")
    message(FATAL_ERROR "zlib.h not found in external/include. "
        "Drop zlib.h and zconf.h into external/include.")
endif()
list(APPEND LIBRARIES ${ZLIB_STATIC_LIB})

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
    src/lights.cu
    src/lightSamplers.cu
    src/naiveIntegrator.cu
    src/misIntegrator.cu
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
    src/SpectrumConsts/spectrum_data.cu
    )


set (NANOVDB_INCLUDE src/openvdb/nanovdb)
include_directories(${NANOVDB_INCLUDE})

list(SORT headers)
list(SORT sources)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Headers" FILES ${headers})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Sources" FILES ${sources})


link_directories(${PROJECT_SOURCE_DIR}/external/lib)
find_package(CUDAToolkit REQUIRED)

# Thrust headers live under the CUDA Toolkit include dir. nvcc gets this
# automatically; the host C++ compiler does not, so expose it explicitly
# for the .cpp files that pull in CUDA-aware headers (e.g. bsdf.h -> thrust/random.h).
include_directories(${CUDAToolkit_INCLUDE_DIRS})
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

########################################
# RHI targets on the CUDA backend (milestone M4, docs/metal-rhi-design.md):
# the same portable renderer core, tests, and single-source kernels the macOS
# build compiles for Metal, here compiled by nvcc against src/rhi/rhi_cuda.cu.
# Every target needs relocatable device code + cudadevrt for the dynamic-
# parallelism indirect dispatch (rhi_cuda.h).
########################################
set(FLUORA_RHI_GL_LIBS ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
set(FLUORA_IMGUI_GL_SOURCES
    src/ImGui/imgui.cpp
    src/ImGui/imgui_draw.cpp
    src/ImGui/imgui_tables.cpp
    src/ImGui/imgui_widgets.cpp
    src/ImGui/imgui_impl_glfw.cpp
    src/ImGui/imgui_impl_opengl3.cpp)
set(FLUORA_RHI_CUDA_SOURCES
    src/rhi/rhi_cuda.cu
    src/rhi/rhi_cuda_present.cpp)
set(FLUORA_SHADER_DIR_DEFINES
    MINI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/mini"
    CORE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/core"
    RHI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/rhi"
    TEST_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/test"
    FLUORA_SCENES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/scenes")

function(fluora_rhi_target name)
    add_executable(${name} ${ARGN})
    target_include_directories(${name} PRIVATE external/include src/ImGui)
    target_compile_definitions(${name} PRIVATE ${FLUORA_SHADER_DIR_DEFINES})
    set_property(TARGET ${name} PROPERTY CUDA_ARCHITECTURES native)
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(${name} ${FLUORA_RHI_GL_LIBS} CUDA::cudart cudadevrt)
endfunction()

fluora_rhi_target(FluoraMini
    src/mini/mini_main.cpp
    src/mini/mini_kernels.cu
    src/core/scene_loader.cpp
    src/core/mesh_loader.cpp
    src/core/bvh_builder.cpp
    src/core/image_loader.cpp
    src/core/spectra.cpp
    src/core/gui/gui.cpp
    src/SpectrumConsts/rgbspectrum_srgb.cpp
    src/stb.cpp
    ${FLUORA_RHI_CUDA_SOURCES}
    ${FLUORA_IMGUI_GL_SOURCES})
target_compile_definitions(FluoraMini PRIVATE RHI_ENABLE_IMGUI)

fluora_rhi_target(RhiTest
    src/test/rhi_test.cpp
    src/test/rhi_test_kernels.cu
    src/rhi/rhi_algorithms.cpp
    ${FLUORA_RHI_CUDA_SOURCES})

fluora_rhi_target(SharedHostTest
    src/test/shared_host_test.cpp
    src/test/shared_probe_kernels.cu
    src/core/spectra.cpp
    src/SpectrumConsts/rgbspectrum_srgb.cpp
    ${FLUORA_RHI_CUDA_SOURCES})