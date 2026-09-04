# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Fluora is a CUDA/C++20 spectral path tracer (originally forked from UPenn's CIS-5650 Project 3 base code) supporting unidirectional path tracing with MIS, volume rendering (NanoVDB), spectral rendering, and an OpenGL/ImGui interactive preview. It is a single-executable interactive renderer: it opens a window, renders progressively via CUDA-OpenGL interop, and lets you fly the camera with WASD + mouse while iterations accumulate.

## Building and running

**Two build paths**, selected automatically by platform in CMakeLists.txt:

**Windows (full CUDA renderer)** — requires an NVIDIA GPU + CUDA Toolkit and MSVC (`memoryUtils.h` uses `_aligned_malloc`; prebuilt `.lib` files live in `external/lib`). The full renderer cannot compile on macOS: CUDA-only changes can only be verified on a Windows CUDA machine.

```powershell
git submodule update --init                 # external/openvdb -> NanoVDB (feature/nanovdb branch)
cmake -S . -B build
cmake --build build --config Release        # or --config Debug
```

The executable lands at `build/bin/<Config>/Fluora.exe`. A scene file path is **required** — `main()` exits with usage if `argc < 2`. Working directory matters: the `.txt` scenes reference meshes/HDRIs as `../scenes/models/...` / `../scenes/environment/...`, i.e. relative to a first-level subdirectory of the repo, so launch from `build/` (or pass absolute paths):

```powershell
cd build
bin\Release\Fluora.exe ..\scenes\bunny-skybox.txt
```

Note that `cornell-sphere.txt` uses a `microfacet` material the main renderer's `.txt` loader does not map (it prints `Unsupported material:` and faults); FluoraMini renders it (as a conductor in PBRT reflectance mode).

The Windows build also produces the RHI targets on the **CUDA backend** (`src/rhi/rhi_cuda.cu`, milestone M4): `FluoraMini.exe`, `RhiTest.exe`, `SharedHostTest.exe`, `RaytraceTest.exe` in the same `build/bin/<Config>/`, with the same CLI/behavior as on macOS (`--no-preview`, `--spp`, `--mode wavefront|mega`, `--out`; output suffix `_cuda.png`). Run `RhiTest`, `SharedHostTest` and `RaytraceTest` after touching `src/rhi/` or any shared `_gpu.h`/`_shared.h`. The CUDA registration translation units are **generated at build time**, not checked in: `fluora_cuda_kernels(<target> <header>)` in CMakeLists.txt runs `cmake/GenerateCudaKernels.cmake`, which scans the single-source header for `GPU_KERNEL(name, GPU_TID_*)` and emits `build/generated/<target>_<header>_kernels.cu` with one `RHI_CUDA_REGISTER_KERNEL` per kernel. A kernel declared after `GPU_SPEC_CONST` must carry a `GPU_SPEC_INSTANCES(name, index, value, ...)` annotation listing the specialization values the app requests (see `wf_shade` in `pathtrace_gpu.h`); the generator emits one `RHI_CUDA_REGISTER_SPEC` per value and errors out if the annotation is missing. So adding a kernel is just adding it to the header (the primitives are registered by `rhi_cuda.cu` itself). Kernels have external linkage, so each `_gpu.h` must be compiled into exactly one `.cu` — `primitives_gpu.h` offers `GPU_PRIMITIVES_HELPERS_ONLY` for TUs that only need its helpers. Gotcha: the VS CUDA integration sometimes skips recompiling a `.cu` whose included headers changed; delete `build/generated/*.cu` (or `touch` them) if a launch-thunk edit in `rhi_cuda.h` seems not to take.

Notes on the Windows build setup:
- C++20 host, CUDA separable compilation, `CUDA_ARCHITECTURES native` (compiles for the local GPU; pin this if you need a portable build).
- Requires zlib (`zlibstatic.lib` + `zlib.h`/`zconf.h`) in `external/lib` and `external/include` — CMake fails fast if they are missing. This is because `media.h` enables `NANOVDB_USE_ZIP` for compressed `.nvdb` volumes.
- OpenImageDenoise is linked from `external/lib`. The post-build step copies every DLL from `external/bin/` (GLEW, OIDN, TBB, oneAPI/SYCL runtimes) next to the .exe — if you move the binary, take the DLLs with it.
- MSVC is forced into standards-conforming preprocessor mode (`/Zc:preprocessor`) for both C++ and nvcc host passes — CUDA 13's CCCL headers require it. `NOMINMAX` is defined globally; don't reintroduce `min`/`max` macros.
- Debug builds add `-G` to CUDA flags for cuda-gdb / Nsight device debugging.

**macOS (FluoraMini, Metal vertical slice)** — builds only the `FluoraMini` target (no CUDA/GL deps). See `docs/metal-rhi-design.md` for the migration plan and what the slice does/doesn't support.

```sh
cmake -B build-mac -DCMAKE_BUILD_TYPE=Release
cmake --build build-mac
./build-mac/bin/FluoraMini scenes/cornell-sphere.txt --spp 1000 --out out.png
./build-mac/bin/FluoraMini      # no scene arg -> opens the Cornell box (scenes/cornell-sphere.txt)
ctest --test-dir build-mac --output-on-failure   # all four test suites; on Windows add -C Release
./build-mac/bin/RhiTest         # RHI parallel-primitive tests — run after touching src/rhi/
./build-mac/bin/SharedHostTest  # host-C++ vs MSL value parity for shared device code
./build-mac/bin/RaytraceTest    # rt_* intersection units + BVH traversal vs brute force, host and GPU
./build-mac/bin/FurnaceTest     # white-furnace energy check (drives FluoraMini; catches both-mode integrator bugs)
```

Device code is single-source (`docs/portable-device-code.md`): shared `_gpu.h`/`_shared.h` files compile under MSL, CUDA, and host C++ through `src/rhi/gpu_portable.h` (gpu_* types, GPU_KERNEL macros, wave shims). Never fork a shader per backend; raw `kernel void`/`__global__` in renderer device code is a review flag (backend-private and test scaffolding kernels are exempt).

FluoraMini's integrator is the portable NEE + MIS path tracer (`src/mini/pathtrace_gpu.h` over `src/core/light_shared.h`, `bsdf_shared.h`, `spectrum_shared.h`): uniform light selection over emissive objects/triangles plus the env map, power-heuristic MIS, one shared `miniShadeVertex` for both modes, and a shadow-ray queue stage in wavefront mode. Per the M4 part-2 decision in `docs/metal-rhi-design.md`, remaining renderer features are ported into this core rather than the old renderer being migrated.

There are no tests, no linter, and no CI for the renderer itself (`RhiTest`/`SharedHostTest`/`RaytraceTest` cover the RHI seam, shared-device-code value parity, and intersection/BVH traversal, on both backends). Verification is otherwise visual: render a scene and compare against `img/` references. For FluoraMini, `--mode mega` and `--mode wavefront` must stay bitwise identical (`cmp` the two `--no-preview` PNGs) on every scene **with `--safe-math` passed to both runs** — that is the regression check used throughout M2–M4. `--safe-math` compiles the Metal shaders with `MTLMathModeSafe` (`DeviceDesc::safeMath`); under the default fast math the two modes can differ by a few ±1/255 pixels because the Metal compiler contracts/reassociates the same shared expressions differently in the megakernel vs the specialized `wf_shade` (the once-bisected `bsdf_tr_D` denominator is now pinned with `gpu_fma`, but fast math retains that license everywhere else, so the divergence persists). Fast math stays the default for interactive/perf runs (safe math costs mega ~40–50%, wavefront ~0–12%). On the CUDA backend kernels are compiled offline, so `--safe-math` cannot change codegen at runtime; its lowering is the `-DFLUORA_CUDA_SAFE_MATH=ON` CMake option (`-fmad=false` on the RHI targets — nvcc never enables `-use_fast_math`, but its default `-fmad=true` still lets ptxas contract shared expressions differently per kernel). Requesting `--safe-math` on a CUDA build without that option logs a warning, and the bitwise checks are only guaranteed with it.

### Runtime controls

Windows preview (`preview.cpp` / `main.cpp`):
- `W`/`A`/`S`/`D`: fly camera (first-person)
- Left mouse drag: rotate camera
- Right mouse drag (vertical): zoom
- Middle mouse drag: pan LOOKAT in X/Z
- Space: recenter on the scene's original LOOKAT
- `Q`: save PNG; Esc: save PNG and exit
- ImGui panel exposes integrator and runtime toggles

FluoraMini opens a live preview window by default with an interactive ImGui overlay: render stats, a fly camera (WASD move, E/C up-down, left-drag look, wheel dolly — accumulation restarts on move), and a dropdown that hot-swaps any sibling `.txt`/`.json` scene (the window resizes to the new scene's RES; it is never user-resizable, since the window size is the render size). q/Esc or the close button save the current image and exit. The portable GUI widget/camera code lives in `src/core/gui` (backend-neutral, shared with the future Windows preview); the Metal backend wiring (ImGui Metal renderer + a Cocoa→ImGui-IO event shim) is in `src/rhi/rhi_metal.mm` behind `RHI_ENABLE_IMGUI`. Pass `--no-preview` for headless/scripted renders — headless output stays bitwise identical to the pre-GUI renderer (verified), and the window shows the same orientation as the saved PNG.

Gotcha when comparing images: `saveImage()` (main.cpp) writes PNGs mirrored (`setPixel(width-1-x, y)`) relative to kernel pixel indexing; all `img/` references use that convention and FluoraMini matches it. Another replicated quirk: the camera's `pixelLength` uses `tan(fovy_degrees→radians)` un-halved (scene.cpp) — FOVY in scene files is effectively a half-angle.

## Architecture

The renderer is heavily modeled on PBRT (pbr-book.org); when in doubt about a convention, PBRT's is the answer.

### GPU polymorphism via TaggedPointer

There is no virtual dispatch in device code. `taggedptr.h` implements PBRT-style tagged pointers: `MaterialPtr`, `BxDFPtr` (`bsdf.h`), `LightPtr` (`light.h`), `MediumPtr`/`PhaseFunctionPtr`/`RayMajorantIteratorPtr` (`medium.h`), and `LightSamplerPtr` (`lightSampler.h`) each subclass `TaggedPointer<T0, T1, ...>` and dispatch with a lambda over the concrete type. **To add a new material/light/medium**: write the concrete class, add it to the corresponding `TaggedPointer` type list, and handle it in scene loading. Concrete objects are allocated with the `Allocator` so device code can dereference them.

**Materials use a newer, preferred pattern** (`taggedindex.h`): persistent storage is a 32-bit `MaterialHandle` ({tag, index}) plus type-segregated pools (`MaterialPool` in `Scene`, uploaded to a POD `MaterialPoolView` in `SceneInfoDev`); kernels resolve handles to a transient `MaterialPtr` via `pool.resolve<MaterialPtr>(handle)`. This exists so device-visible data contains no raw host pointers (required by the Metal backend). When touching lights/media/spectra storage, prefer migrating them to this handle pattern over adding new `alloc.new_object` pointer graphs.

**Metal migration layout rule** (see `docs/metal-rhi-design.md`): portable host code goes in `src/core/`, backend seams/device code/primitives in `src/rhi/` (the Metal backend there is real and in the macOS build), and `src/mini/` is scaffolding that only shrinks — it may call core/rhi but never gain capability of its own.

### Memory model

`memoryUtils.h` defines `Allocator` over a `MemoryResourceBackend` hierarchy: `CUDAMemoryResourceBackend` uses `cudaMallocManaged` (unified memory — the same pointers are dereferenced on host and device), and `MonotonicBlockMemoryResourceBackend` bump-allocates 256 KiB blocks on top of it. `main.cpp` owns the global backends (`baseBackend`, `mainBlockBackend`). Flat bulk arrays (paths, intersections, geometry) are still allocated with raw `cudaMalloc` in `pathtrace.cu`.

Allocate scene-lifetime GPU data through `Allocator(mainBlockBackend)` (bump, no per-object free); allocate things that must be freed individually (e.g. NanoVDB grids, which don't fit the bump allocator) through `Allocator(baseBackend)`.

### Frame flow

1. `main.cpp` sets up the backends, calls `initScene()`, then uploads scene assets in stages: `LoadAllTexturesToGPU`, `LoadAllMeshesToGPU`, `LoadAllMaterialsToGPU`, `LoadAllMediaToGPU`, `LoadAllLightsToGPU`. The BVH is built host-side and flattened for stackless GPU traversal here.
2. `runCuda()` maps an OpenGL PBO to a CUDA pointer (`cudaGLMapBufferObject`) and calls `pathtrace()`; `preview.cpp::mainLoop` drives one `runCuda()` (= one path-tracing iteration) per frame.
3. `pathtrace.cu` owns all device-side state (`dev_*` statics, set up in `pathtraceInit`) and dispatches to an integrator, selected by the `mainIntegratorType` global at the top of `pathtrace.cu`: `naiveIntegrator.cu` (random walk) or `misIntegrator.cu` (NEE + MIS, the default).
4. Integrators run the loop: generate camera rays → intersect (BVH traversal in `intersections.cu`) → shade/scatter (`bsdf.cu`, `materials.cu`, `microfacet.cu`) → optional stream compaction and material sort (Thrust) → continue. First-bounce intersections can be cached across iterations (`FIRST_INTERSECTION_CACHING`). Radiance accumulates into `RGBFilm` (spectral samples converted via `PixelSensor`, `spectrum.h`, `SpectrumConsts/`).
5. `sendImageToPBO` tonemaps (ACES) into the PBO; `preview.cpp` draws it with OpenGL + ImGui. Optional OIDN denoising (beauty + normal + albedo aux buffers) lives in `main.cpp`, gated by `DENOISE` and the ImGui panel.

### Scene and acceleration

`scene.cpp` parses the scene file, loading glTF (tinygltf), OBJ (tinyobjloader), and PLY (tinyply) geometry, HDR environment maps (tinyexr / stb_image), and building the BVH on the CPU (`bvh.cpp`, SAH-based). Textures are uploaded as `cudaTextureObject_t` and looked up by name via maps in `scene.h`/`containers.h`.

Two stackless GPU BVH layouts coexist in `src/bvh.{h,cpp}`, chosen by the `MTBVH` macro:
- **State-machine BVH** — single tree with parent links, traversed via FROM_PARENT / FROM_CHILD / FROM_SIBLING states.
- **MTBVH (Multi-Threaded BVH)** — six precomputed direction-ordered hit/miss chains, much faster but ~4–5× larger. This is the default (`MTBVH 1`).

The README's "BVH building and traversal" section has the algorithmic details and benchmarks.

### Materials, spectra, and media

- `src/materials.{h,cu}` defines material types dispatched through the tagged-pointer / tagged-index machinery above — `diffuse`, `dielectric`, `conductor`, `microfacet`, `emitting`, etc.
- `src/spectrum.{h,cu}` + `external/SpectrumConsts/` implement PBRT-v4-style sampled spectra and RGB↔spectrum tables (sRGB, ACES, DCI-P3, Rec2020). The renderer can run spectrally rather than RGB.
- `src/microfacet.{h,cu}` implements GGX VNDF sampling and the asymmetric microfacet model from the SIGGRAPH 2023 "Microfacet theory for non-uniform heightfields" paper (conductor + dielectric variants — see the README showcase).
- `src/media.{h,cu}` + `external/openvdb/nanovdb` implement homogeneous and heterogeneous (NanoVDB) volumes. `.nvdb` files may be gzip-compressed (hence the zlib requirement).

### Lights

`src/lights.{h,cu}` and `src/lightSamplers.{h,cu}`: area lights from emissive geometry, environment map lights with importance-sampled 2D conditional CDF, plus the `LightSampler` abstraction used by the MIS integrator.

### Other notable subsystems

- `external/openvdb/nanovdb` — vendored NanoVDB headers for volumetric data (git submodule).
- `external/mikktspace/` — MikkTSpace tangent generation for normal mapping (used when glTF meshes don't ship with tangents).
- `external/ImGui/` — vendored ImGui (OpenGL3 + GLFW backends on Windows, Metal backend on macOS).
- `src/soa.h`, `src/workqueue.h`, `src/containers.h` — SoA helpers and queue scaffolding intended for wavefront-style rewrites; not all integrators are fully wavefront yet.

### Compile-time feature toggles

Renderer options are `#define`s at the top of `src/sceneStructs.h` rather than runtime flags — `USE_BVH`, `MTBVH`, `MIS_POWER_2`, `DENOISE`, `TONEMAPPING`, `DOF_ENABLED`, `STOCHASTIC_SAMPLING`, `FIRST_INTERSECTION_CACHING`, `SORT_BY_MATERIAL_TYPE`, `WHITE_FURNANCE_TEST`, `MAX_DEPTH`, etc. Changing behavior often means flipping one of these rather than a runtime setting, and requires a rebuild. Check this header first before adding a new toggle — it's the canonical place.

### CUDA specifics to preserve

- `.cu` files are CUDA translation units, `.cpp` are pure host. Host/device-shared code is marked `__host__ __device__` (or `CPU_GPU_FUNC` from `defines.h`) — most headers under `src/` are compiled by both nvcc and MSVC, so keep them free of host-only constructs.
- CUDA-aware headers (anything pulling `thrust/...`) must be included from `.cu` files or from `.cpp` files whose translation unit explicitly includes the CUDA Toolkit headers — CMake exposes `${CUDAToolkit_INCLUDE_DIRS}` globally for this reason.
- `workqueue.h` provides `SOA<T>`-backed work queues with `cuda::atomic` and a generic `GPUParallelFor`; separable compilation is on (`CUDA_SEPARABLE_COMPILATION`), and device-side Thrust (sort/remove/random) is used throughout the integrators.
- The path tracer accumulates radiance into a film and divides by `iter` at display time. Don't reset the film without also resetting `iteration` in `main.cpp` — camera-change handling already does this via `camchanged`.

## Scene file formats

Two formats live side-by-side under `scenes/`; extension dispatch happens in `Scene::Scene` (`scene.cpp`) for the old renderer and in `loadScene()` (`src/core/scene_loader.cpp`) for FluoraMini, which reads both.
- `.txt` — the original CIS-5650 line-oriented format (MATERIAL / OBJECT / CAMERA blocks). Documented in `INSTRUCTION.md` and well-suited for quick test scenes. Loader: `Scene::loadMaterial` / `loadObject` / `loadCamera` in `scene.cpp`; `loadTxtScene` in the core.
- `.json` — newer format with richer material descriptions (spectral eta/k, named spectra from `SpectrumConsts/`), volumes, and proper camera DOF parameters. Loader: `Scene::loadJSON` in `scene.cpp`; `loadJsonScene` in the core (PLY/OBJ/inline meshes, EXR textures, env SCALE/MAXRGB; media, medium interfaces and DOF are parsed into `CoreScene` but not rendered yet — interface-only objects render as pass-through boundaries). Prefer this for new scenes. Paths inside both formats are joined with the scene's directory, so the `../scenes/...` form works from the repo root or `build/`.

## Git conventions

When fixing a feature that is broken by a recent commit that has not been pushed, amend/fixup that commit (rebase if it is not HEAD) instead of stacking new "fix the fix" commits — keep one clean commit per feature.
