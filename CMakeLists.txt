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

# All four test suites run under CTest: `ctest --test-dir <build>` (add
# --output-on-failure; on multi-config generators also -C <Config>).
enable_testing()

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
        external/ImGui/imgui.cpp
        external/ImGui/imgui_draw.cpp
        external/ImGui/imgui_tables.cpp
        external/ImGui/imgui_widgets.cpp
        external/ImGui/imgui_impl_metal.mm)
    add_executable(FluoraMini
        src/mini/mini_main.cpp
        src/core/scene_loader.cpp
        src/core/mesh_loader.cpp
        src/core/bvh_builder.cpp
        src/core/image_loader.cpp
        src/core/spectra.cpp
        src/core/lights.cpp
        src/core/gui/gui.cpp
        external/SpectrumConsts/rgbspectrum_srgb.cpp
        src/rhi/rhi_metal.mm
        src/stb.cpp
        ${FLUORA_IMGUI_SOURCES}
    )
    target_include_directories(FluoraMini PRIVATE external/include external/ImGui)
    # MSL sources are loaded from the source tree at runtime (dev convenience;
    # packaging deferred to M5)
    target_compile_definitions(FluoraMini PRIVATE
        MINI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/mini"
        CORE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/core"
        RHI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/rhi"
        FLUORA_SCENES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/scenes"
        RHI_ENABLE_IMGUI)
    # rhi_metal.mm and the ImGui Metal backend are Objective-C++ under ARC.
    set_source_files_properties(src/rhi/rhi_metal.mm external/ImGui/imgui_impl_metal.mm
        PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(FluoraMini "-framework Metal" "-framework Foundation"
        "-framework AppKit" "-framework QuartzCore")

    # RHI primitive unit tests (reduce/scan/compact/sort/work queue)
    add_executable(RhiTest
        src/test/rhi_test.cpp
        src/rhi/rhi_algorithms.cpp
        src/rhi/rhi_metal.mm
    )
    target_compile_definitions(RhiTest PRIVATE
        RHI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/rhi"
        TEST_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/test")
    target_link_libraries(RhiTest "-framework Metal" "-framework Foundation"
        "-framework AppKit" "-framework QuartzCore")

    # Dual-personality value test for the portable device-code shim: the same
    # shared headers compiled as host C++ (the CUDA-side spelling) and as MSL,
    # asserting numeric agreement (docs/portable-device-code.md).
    add_executable(SharedHostTest
        src/test/shared_host_test.cpp
        src/core/lights.cpp
        src/core/spectra.cpp
        external/SpectrumConsts/rgbspectrum_srgb.cpp
        src/rhi/rhi_metal.mm
    )
    target_include_directories(SharedHostTest PRIVATE external/include)
    target_compile_definitions(SharedHostTest PRIVATE
        CORE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/core"
        RHI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/rhi"
        TEST_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/test")
    target_link_libraries(SharedHostTest "-framework Metal" "-framework Foundation"
        "-framework AppKit" "-framework QuartzCore")

    # Intersection/BVH tests: rt_* unit cases, host traversal vs brute force
    # over a real bvh_builder tree, and the same rays traced on the GPU.
    add_executable(RaytraceTest
        src/test/raytrace_test.cpp
        src/core/bvh_builder.cpp
        src/rhi/rhi_metal.mm
    )
    target_include_directories(RaytraceTest PRIVATE external/include)
    target_compile_definitions(RaytraceTest PRIVATE
        CORE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/core"
        RHI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/rhi"
        TEST_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/test")
    target_link_libraries(RaytraceTest "-framework Metal" "-framework Foundation"
        "-framework AppKit" "-framework QuartzCore")

    # White-furnace test: drives the FluoraMini binary headlessly (no GPU deps
    # of its own) and checks energy conservation via MINI_DUMP_ACCUM.
    add_executable(FurnaceTest src/test/furnace_test.cpp)
    add_dependencies(FurnaceTest FluoraMini)

    add_test(NAME RhiTest COMMAND RhiTest)
    add_test(NAME SharedHostTest COMMAND SharedHostTest)
    add_test(NAME RaytraceTest COMMAND RaytraceTest)
    add_test(NAME FurnaceTest COMMAND FurnaceTest)
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
	external/mikktspace/mikktspace.c
    external/ImGui/imgui.cpp 
    external/ImGui/imgui_demo.cpp 
    external/ImGui/imgui_draw.cpp 
    external/ImGui/imgui_impl_glfw.cpp 
    external/ImGui/imgui_impl_opengl3.cpp 
    external/ImGui/imgui_tables.cpp 
    external/ImGui/imgui_widgets.cpp 
    external/SpectrumConsts/rgbspectrum_aces.cpp
    external/SpectrumConsts/rgbspectrum_dci_p3.cpp
    external/SpectrumConsts/rgbspectrum_rec2020.cpp
    external/SpectrumConsts/rgbspectrum_srgb.cpp
    external/SpectrumConsts/spectrum_data.cu
    )


set (NANOVDB_INCLUDE external/openvdb/nanovdb)
include_directories(${NANOVDB_INCLUDE})

list(SORT headers)
list(SORT sources)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Headers" FILES ${headers})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Sources" FILES ${sources})


link_directories(${PROJECT_SOURCE_DIR}/external/lib)
find_package(CUDAToolkit REQUIRED)

# Thrust headers live under the CUDA Toolkit include dir. nvcc gets this
# automatically; the host C++ compiler does not, so expose it explicitly
# for the .cpp files that pull in CUDA-aware headers (e.g. bsdf.h -> thrust/random.h).
include_directories(${CUDAToolkit_INCLUDE_DIRS})
add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE external/ImGui)
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
    external/ImGui/imgui.cpp
    external/ImGui/imgui_draw.cpp
    external/ImGui/imgui_tables.cpp
    external/ImGui/imgui_widgets.cpp
    external/ImGui/imgui_impl_glfw.cpp
    external/ImGui/imgui_impl_opengl3.cpp)
set(FLUORA_RHI_CUDA_SOURCES
    src/rhi/rhi_cuda.cu
    src/rhi/rhi_cuda_present.cpp)
set(FLUORA_SHADER_DIR_DEFINES
    MINI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/mini"
    CORE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/core"
    RHI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/rhi"
    TEST_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/test"
    FLUORA_SCENES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/scenes")

# The CUDA lowering of DeviceDesc::safeMath. CUDA kernels are compiled
# offline, so the runtime flag cannot change their codegen the way Metal's
# runtime MSL compile can; this option is the build-time equivalent. nvcc
# never enables -use_fast_math here, but its default -fmad=true still lets
# ptxas contract the same shared expression differently in the megakernel vs
# a specialized wf_shade instantiation — the exact mechanism --safe-math
# suppresses on Metal. OFF by default for perf; configure with
# -DFLUORA_CUDA_SAFE_MATH=ON before trusting bitwise mega==wavefront checks
# (FurnaceTest, manual cmp) on this backend. Requesting --safe-math on a
# build without it logs a runtime warning (rhi_cuda.cu).
option(FLUORA_CUDA_SAFE_MATH
    "Compile RHI CUDA device code with -fmad=false (bitwise mega==wavefront)" OFF)

function(fluora_rhi_target name)
    add_executable(${name} ${ARGN})
    target_include_directories(${name} PRIVATE external/include external/ImGui)
    target_compile_definitions(${name} PRIVATE ${FLUORA_SHADER_DIR_DEFINES})
    set_property(TARGET ${name} PROPERTY CUDA_ARCHITECTURES native)
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(${name} ${FLUORA_RHI_GL_LIBS} CUDA::cudart cudadevrt)
    if (FLUORA_CUDA_SAFE_MATH)
        target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
        target_compile_definitions(${name} PRIVATE RHI_CUDA_SAFE_MATH)
    endif()
endfunction()

# Generates <build>/generated/<header-stem>_kernels.cu -- the CUDA registration
# TU for one single-source _gpu.h (RHI_CUDA_REGISTER_* per GPU_KERNEL /
# GPU_SPEC_INSTANCES found in it) -- and adds it to the target. Re-runs when
# the header or the generator changes. INCLUDES: headers to include first, for
# concat-style headers that do not include their own dependencies.
function(fluora_cuda_kernels target header)
    cmake_parse_arguments(FCK "" "" "INCLUDES" ${ARGN})
    get_filename_component(_abs "${header}" ABSOLUTE)
    get_filename_component(_stem "${header}" NAME_WE)
    set(_out "${CMAKE_BINARY_DIR}/generated/${target}_${_stem}_kernels.cu")
    set(_incs "")
    foreach(_i IN LISTS FCK_INCLUDES)
        get_filename_component(_i "${_i}" ABSOLUTE)
        list(APPEND _incs "${_i}")
    endforeach()
    set(_gen "${CMAKE_SOURCE_DIR}/cmake/GenerateCudaKernels.cmake")
    add_custom_command(
        OUTPUT "${_out}"
        COMMAND ${CMAKE_COMMAND}
            -DINPUT=${_abs} -DOUTPUT=${_out} "-DINCLUDES=${_incs}"
            -DREGISTRY_HEADER=${CMAKE_SOURCE_DIR}/src/rhi/rhi_cuda.h
            -P "${_gen}"
        DEPENDS "${_abs}" "${_gen}"
        COMMENT "Generating CUDA kernel registrations for ${header}"
        VERBATIM)
    target_sources(${target} PRIVATE "${_out}")
endfunction()

fluora_rhi_target(FluoraMini
    src/mini/mini_main.cpp
    src/core/scene_loader.cpp
    src/core/mesh_loader.cpp
    src/core/bvh_builder.cpp
    src/core/image_loader.cpp
    src/core/spectra.cpp
    src/core/lights.cpp
    src/core/gui/gui.cpp
    external/SpectrumConsts/rgbspectrum_srgb.cpp
    src/stb.cpp
    ${FLUORA_RHI_CUDA_SOURCES}
    ${FLUORA_IMGUI_GL_SOURCES})
target_compile_definitions(FluoraMini PRIVATE RHI_ENABLE_IMGUI)
fluora_cuda_kernels(FluoraMini src/mini/pathtrace_gpu.h)

fluora_rhi_target(RhiTest
    src/test/rhi_test.cpp
    src/rhi/rhi_algorithms.cpp
    ${FLUORA_RHI_CUDA_SOURCES})
fluora_cuda_kernels(RhiTest src/test/rhi_test_gpu.h)

fluora_rhi_target(SharedHostTest
    src/test/shared_host_test.cpp
    src/core/lights.cpp
    src/core/spectra.cpp
    external/SpectrumConsts/rgbspectrum_srgb.cpp
    ${FLUORA_RHI_CUDA_SOURCES})
fluora_cuda_kernels(SharedHostTest src/test/shared_probe.h INCLUDES
    src/rhi/gpu_portable.h src/rhi/primitives_shared.h src/core/spectrum_shared.h
    src/core/bsdf_shared.h src/core/envmap_shared.h src/core/light_shared.h
    src/core/tonemap_shared.h)

fluora_rhi_target(RaytraceTest
    src/test/raytrace_test.cpp
    src/core/bvh_builder.cpp
    ${FLUORA_RHI_CUDA_SOURCES})
fluora_cuda_kernels(RaytraceTest src/test/raytrace_test_gpu.h)

# White-furnace test: drives the FluoraMini binary headlessly (no GPU deps of
# its own) and checks energy conservation via MINI_DUMP_ACCUM.
add_executable(FurnaceTest src/test/furnace_test.cpp)
add_dependencies(FurnaceTest FluoraMini)

add_test(NAME RhiTest COMMAND RhiTest)
add_test(NAME SharedHostTest COMMAND SharedHostTest)
add_test(NAME RaytraceTest COMMAND RaytraceTest)
add_test(NAME FurnaceTest COMMAND FurnaceTest)