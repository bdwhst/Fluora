# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Fluora is a CUDA/C++20 spectral path tracer (originally UPenn CIS-5650 Project 3) supporting unidirectional path tracing with MIS, volume rendering (NanoVDB), spectral rendering, and an OpenGL/ImGui interactive preview.

## Building and running

**Two build paths**, selected automatically by platform in CMakeLists.txt:

**Windows (full CUDA renderer)** — requires an NVIDIA GPU + CUDA Toolkit and MSVC (`memoryUtils.h` uses `_aligned_malloc`; prebuilt `.lib` files live in `external/lib`). The full renderer cannot compile on macOS: CUDA-only changes can only be verified on a Windows CUDA machine.

```sh
git submodule update --init          # src/openvdb -> NanoVDB (feature/nanovdb branch)
cmake -B build
cmake --build build --config Release
```

Run with a scene file as the only argument: `Fluora.exe scenes/cornell-sphere.txt`. Controls: WASD moves the camera, left mouse rotates, `q` saves the current image.

**macOS (FluoraMini, Metal vertical slice)** — builds only the `FluoraMini` target (no CUDA/GL deps). See `docs/metal-rhi-design.md` for the migration plan and what the slice does/doesn't support.

```sh
cmake -B build-mac -DCMAKE_BUILD_TYPE=Release
cmake --build build-mac
./build-mac/bin/FluoraMini scenes/cornell-sphere.txt --spp 1000 --out out.png
./build-mac/bin/RhiTest    # unit tests for the RHI parallel primitives — run after touching src/rhi/
```

FluoraMini opens a live preview window by default (updates per iteration, freezes at the final frame until closed; q/Esc also close it). Pass `--no-preview` for headless/scripted renders — preview and headless output are bitwise identical, and the window shows the same orientation as the saved PNG.

Gotcha when comparing images: `saveImage()` (main.cpp) writes PNGs mirrored (`setPixel(width-1-x, y)`) relative to kernel pixel indexing; all `img/` references use that convention and FluoraMini matches it. Another replicated quirk: the camera's `pixelLength` uses `tan(fovy_degrees→radians)` un-halved (scene.cpp) — FOVY in scene files is effectively a half-angle.

Prerequisites the CMake build hard-fails without: a zlib static lib in `external/lib` plus `zlib.h`/`zconf.h` in `external/include` (needed because NanoVDB reads zip-compressed `.nvdb` volumes). OpenImageDenoise is linked from `external/lib`, with DLLs copied from `external/bin` post-build.

There are no tests and no linter. Verification is visual: render a scene and compare against `img/` references.

## Architecture

The renderer is heavily modeled on PBRT (pbr-book.org); when in doubt about a convention, PBRT's is the answer.

### GPU polymorphism via TaggedPointer

There is no virtual dispatch in device code. `taggedptr.h` implements PBRT-style tagged pointers: `MaterialPtr`, `BxDFPtr` (`bsdf.h`), `LightPtr` (`light.h`), `MediumPtr`/`PhaseFunctionPtr`/`RayMajorantIteratorPtr` (`medium.h`), and `LightSamplerPtr` (`lightSampler.h`) each subclass `TaggedPointer<T0, T1, ...>` and dispatch with a lambda over the concrete type. **To add a new material/light/medium**: write the concrete class, add it to the corresponding `TaggedPointer` type list, and handle it in scene loading. Concrete objects are allocated with the `Allocator` so device code can dereference them.

**Materials use a newer, preferred pattern** (`taggedindex.h`): persistent storage is a 32-bit `MaterialHandle` ({tag, index}) plus type-segregated pools (`MaterialPool` in `Scene`, uploaded to a POD `MaterialPoolView` in `SceneInfoDev`); kernels resolve handles to a transient `MaterialPtr` via `pool.resolve<MaterialPtr>(handle)`. This exists so device-visible data contains no raw host pointers (required by the Metal backend). When touching lights/media/spectra storage, prefer migrating them to this handle pattern over adding new `alloc.new_object` pointer graphs.

**Metal migration layout rule** (see `docs/metal-rhi-design.md`): portable host code goes in `src/core/`, backend seams/device code/primitives in `src/rhi/` (the Metal backend there is real and in the macOS build), and `src/mini/` is scaffolding that only shrinks — it may call core/rhi but never gain capability of its own.

### Memory model

`memoryUtils.h` defines `Allocator` over a `MemoryResourceBackend` hierarchy: `CUDAMemoryResourceBackend` uses `cudaMallocManaged` (unified memory — the same pointers are dereferenced on host and device), and `MonotonicBlockMemoryResourceBackend` bump-allocates from blocks on top of it. `main.cpp` owns the global backends. Flat bulk arrays (paths, intersections, geometry) are still allocated with raw `cudaMalloc` in `pathtrace.cu`.

### Frame flow

1. `main.cpp` `runCuda()` maps an OpenGL PBO to a CUDA pointer (`cudaGLMapBufferObject`) and calls `pathtrace()`.
2. `pathtrace.cu` owns all device-side state (`dev_*` statics, set up in `pathtraceInit`) and dispatches to an integrator, selected by the `mainIntegratorType` global at the top of `pathtrace.cu`: `naiveIntegrator.cu` (random walk) or `misIntegrator.cu` (NEE + MIS, the default).
3. Integrators run the loop: generate rays → intersect (BVH traversal in `intersections.cu`) → shade/scatter → stream-compact terminated paths (Thrust). Radiance accumulates into `RGBFilm` (spectral samples converted via `PixelSensor`, `spectrum.h`, `SpectrumConsts/`).
4. `sendImageToPBO` tonemaps (ACES) into the PBO; `preview.cpp` draws it with OpenGL + ImGui. Optional OIDN denoising lives in `main.cpp`.

### Scene and acceleration

`scene.cpp` parses the custom `.txt` scene format (and `.json` for volume scenes; see `scenes/` for examples), loading glTF/OBJ/PLY geometry and building the BVH on the CPU (`bvh.cpp`, SAH-based). Two stackless GPU traversal layouts exist, chosen by the `MTBVH` macro: the default is the multi-threaded BVH (six direction-ordered arrays with hit/miss links). Textures are uploaded as `cudaTextureObject_t` and looked up by name via maps in `scene.h`/`containers.h`.

### Compile-time feature toggles

Renderer options are `#define`s at the top of `sceneStructs.h`: `MTBVH`, `DENOISE`, `TONEMAPPING`, `DOF_ENABLED`, `STOCHASTIC_SAMPLING`, `FIRST_INTERSECTION_CACHING`, `SORT_BY_MATERIAL_TYPE`, etc. Changing behavior often means flipping one of these rather than a runtime setting.

### CUDA specifics to preserve

- Host/device-shared code is marked `__host__ __device__` (or `CPU_GPU_FUNC` from `defines.h`) — most headers under `src/` are compiled by both nvcc and MSVC, so keep them free of host-only constructs.
- `workqueue.h` provides `SOA<T>`-backed work queues with `cuda::atomic` and a generic `GPUParallelFor`; separable compilation is on (`CUDA_SEPARABLE_COMPILATION`), and device-side Thrust (sort/remove/random) is used throughout the integrators.

## Git conventions

When fixing a feature that is broken by a recent commit that has not been pushed, amend/fixup that commit (rebase if it is not HEAD) instead of stacking new "fix the fix" commits — keep one clean commit per feature.
