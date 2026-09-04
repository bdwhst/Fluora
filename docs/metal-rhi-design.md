# Metal RHI: design and migration plan

Status: M0–M3 landed (FluoraMini renders full scenes spectrally on Metal);
M4 part 1 landed 2026-09-03 (real CUDA backend — FluoraMini, RhiTest and
SharedHostTest build and pass on Windows/CUDA, both render modes bitwise
identical; the main CUDA renderer builds again). M4 part 2 in progress: grow
the portable renderer instead of migrating pathtrace.cu — NEE+MIS, the
.json/PLY loader, homogeneous media and NanoVDB grids landed, DOF + denoise
next · Owner: bdwhst · Last updated: 2026-09-04

## 1. Context and goal

Fluora is a CUDA/C++ spectral path tracer that currently builds only on Windows with an
NVIDIA GPU. The goal is a **swappable GPU backend (RHI) covering compute and ray
tracing**, with Apple Metal as the second backend, so the renderer can run on Apple
Silicon Macs.

The forcing constraint: development is currently happening on a Mac with **no CUDA
device available**, so the plan must produce something buildable and testable on macOS
early, without needing the CUDA renderer to compile here. That drives the milestone
order below — a small, self-contained **vertical slice** (Cornell box rendered via the
RHI on Metal) comes first, and the existing CUDA renderer is migrated onto the same RHI
later, when a CUDA machine is available to verify parity.

## 2. Why this is a restructuring, not just an abstraction

An audit of the codebase (see git history around the `taggedindex.h` introduction)
found three CUDA assumptions that pervade the ~12k lines of device code:

1. **Unified virtual addressing.** `Allocator` sits on `cudaMallocManaged`; host code
   builds pointer graphs (`TaggedPointer` bit-packs raw 64-bit pointers) that kernels
   dereference directly. Metal has no shared CPU/GPU address space
   (`MTLBuffer.contents` ≠ `gpuAddress`).
2. **Single-source `__host__ __device__` C++** with glm and `thrust::random`
   throughout. MSL is a separate C++14-ish dialect that cannot include either.
3. **Host-lambda kernel launches** (`GPUParallelFor`, extended lambdas) and Thrust
   host-side algorithms (stream compaction, sort). Metal kernels are named, precompiled
   entry points; there is no Thrust.

## 3. Portability invariants

Every migrated subsystem must satisfy these; new code should not regress them.

- **I-1: No raw host pointers in device-visible data.** Persistent references are
  `{type tag, array index}` handles (`src/taggedindex.h`) resolved against per-dispatch
  pool views, or buffer-slot + offset pairs. Raw device pointers may exist only
  *transiently inside a kernel*. (Done for materials; lights/media/spectra pending.)
- **I-2: Kernels are named entry points** taking one POD parameter block plus a list of
  buffer resources bound to slots. No lambdas across the launch boundary.
- **I-3: Shared host/GPU structs are layout-audited PODs** defined once in a header
  compiled by both sides (see `src/mini/mini_shared.h` for the pattern: `float3`/
  `float4x4` under MSL, `simd_float3`/`simd_float4x4` under host clang — identical
  size/alignment).
- **I-4: The renderer core calls `rhi::` interfaces only** (`src/rhi/rhi.h`); backend
  headers (`cuda_runtime.h`, `Metal/Metal.h`) appear only inside backend
  implementations.

## 4. RHI shape (host side)

Defined in `src/rhi/rhi.h`; one implementation file per backend, selected by
`rhi::kNativeBackend`: `rhi_metal.mm` (macOS) and `rhi_cuda.cu` +
`rhi_cuda_present.cpp` (Windows/CUDA, landed in M4), with `rhi_cuda.h` as the
kernel-registration API a `.cu` file includes.

| Concept | CUDA backend | Metal backend |
|---|---|---|
| `Device` | context + kernel registry (`RHI_CUDA_REGISTER_KERNEL/SPEC`) | `MTLDevice` + `MTLLibrary` |
| `Buffer` | `cudaMalloc` / `cudaMallocManaged` | `MTLBuffer` (Private/Shared) |
| `Texture` | `cudaArray` + `cudaTextureObject_t`; heap = device array of objects + host shadow | `MTLTexture` + sampler, bindless heap index |
| `ComputePipeline` | registry entry (launch thunk deduced from the kernel signature) | `MTLComputePipelineState` |
| `CommandStream` | blocking `cudaStream_t` + event ring (2 submits in flight) | `MTLCommandBuffer` + compute encoders |
| indirect dispatch | 1-thread launcher kernel + dynamic parallelism | `dispatchThreadgroups(indirectBuffer:)` |
| `RayIntersector` | CPU-built threaded BVH upload (`core/bvh_builder`) | same (M3) or `MTLAccelerationStructure` (M5) |
| present | GLFW window, GL texture via CUDA-registered PBO, ImGui GLFW/GL3 | `CAMetalLayer` drawable + ImGui Metal |

**Dispatch/binding convention** (I-2): `dispatch(pipeline, gridGroups, groupSize,
params, paramsSize, {buffers...})`. The parameter block binds at Metal `buffer(0)` via
`setBytes`; resource *i* binds at `buffer(i+1)`. The CUDA thunk reconstructs its kernel
argument struct from the same (params, buffer-address list). Grid semantics equal CUDA's
(grid = number of groups, block = threads per group). Bindless via `DeviceAddress`
(`gpuAddress` / device pointer) is kept in the interface for later wavefront work but is
not used in M1 — slot binding avoids Metal residency bookkeeping while the buffer count
is small.

**Shader compilation**: M1 compiles MSL from source at device creation
(`newLibraryWithSource`, sources concatenated by the host: shared-structs header first,
then kernels — runtime MSL compilation cannot resolve `#include`s). Later milestones
should move to an offline `xcrun metal` → `.metallib` CMake step; runtime compile keeps
M1 free of a Metal-toolchain build dependency.

## 5. Ray tracing seam

Host side, `RayIntersector::build()` consumes the scene's primitives and returns an
opaque POD `TraversalView` blob that kernels receive in their parameter block. Device
side, integrator code calls only `rt::intersect(view, ray)` / `rt::occluded(view, ray)`,
compiled per backend:

- **Parity path (M3):** port the existing MTBVH stackless traversal
  (`intersections.cu`) to MSL. The CPU SAH builder (`bvh.cpp`) is backend-neutral and
  stays. This gives bit-comparable traversal on both backends — debuggable.
- **Fast path (M5):** `MTLAccelerationStructure` + MSL `intersection_query`, bypassing
  the CPU BVH entirely. Hardware RT on M3/M4-class GPUs; same `rt::` call sites.

M1 does not use `RayIntersector` at all: the Cornell scene is a handful of analytic
cubes/spheres, intersected by a brute-force loop in the kernel. That keeps the slice
free of BVH porting while still exercising Device/Buffer/Pipeline/CommandStream.

## 6. Milestones

Milestones are ordered **Mac-first**: the CUDA machine is unavailable until ~week of
2026-07-27, so M2/M3 proceed on macOS now and M4 is the CUDA catch-up pass that fixes
whatever the Mac-side work broke and verifies parity.

**M0 — handle groundwork** *(landed)*: `taggedindex.h`; materials converted from
pointer graphs to `MaterialHandle` + `MaterialPool` (satisfies I-1 for materials).
Unverified on CUDA until M4.

**M1 — Metal vertical slice: Cornell box on this Mac** *(landed)*:
- `rhi.h` finalized for compute; **real Metal backend** `src/rhi/rhi_metal.mm`
  implementing Device/Buffer/ComputePipeline/CommandStream (Texture, RayIntersector,
  present throw "unimplemented").
- `FluoraMini` target (`src/mini/`): parses the existing scene format subset
  (MATERIAL/CAMERA/OBJECT with cube/sphere), megakernel naive path tracer in MSL,
  accumulates into a shared buffer, CPU-side ACES tonemap (same curve as
  `sendImageToPBO`), PNG out via stb.
- Camera and transform conventions replicate the CUDA renderer exactly
  (`T·Rx·Ry·Rz·S`, `pixelLength` from `tan(fovy°)`, ray =
  `view − right·plx·(x−w/2+jx) − up·ply·(y−h/2+jy)`) so images are framed identically.
- Material mapping, deliberately RTIOW-grade for the slice: `diffuse` → Lambert
  (cosine-sampled), `emitting` → emitter, `frenselSpecular` → ideal glass (Schlick),
  `microfacet` → perfect mirror (roughness ignored), unknown → diffuse.
- **Acceptance:** `cmake -B build && cmake --build build` on macOS, then
  `./build/bin/FluoraMini scenes/cornell-sphere.txt --spp 200` writes a recognizable
  Cornell render (compare `img/REFERENCE_cornell.5000samp.png` for framing/orientation;
  exact radiometry will differ — the slice is RGB, not spectral).

**M2 — GPU primitives + wavefront pattern on Metal** *(Mac, in progress)*:
- Backend-agnostic parallel primitives as RHI algorithms (`src/rhi/rhi_algorithms.*`
  host-side, kernels named per I-2 so the CUDA backend can later register same-named
  equivalents): **reduce, exclusive scan, compact, LSD radix sort** in MSL, with a
  `RhiTest` unit-test target verifying each against CPU references.
- **Atomic work queue** with simdgroup-aggregated push (one `fetch_add` per simdgroup,
  not per thread) + `dispatchIndirect` so GPU-written queue counts drive dispatches
  without CPU readback.
- Decision: keep **both** queues and compact/sort. Queues are the primary bounce-loop
  mechanism; the primitives stay for A/B measurement and future needs. The ray-ordering
  loss from queues is judged minor — paths decorrelate after the first bounce in
  mostly-diffuse scenes anyway — but the A/B on CUDA in M4 confirms it.
- FluoraMini restructured into wavefront stages (raygen → per-bounce
  intersect/shade with ping-pong queues and indirect dispatch; terminated paths
  simply aren't re-enqueued). `--mode wavefront|mega` selects; both share every
  shading helper and produce **bitwise-identical PNGs** (verified at 500 spp).
- Measured on the Cornell toy scene (800×800×500spp): mega 0.89 s, wavefront
  3.15 s — per-sample dispatch overhead (~34 encoders) and 80 B/path queue
  traffic dominate when intersect/shade are trivial. Expected to invert once
  real BSDFs/BVH make stages expensive and divergent; re-measure in M3/M4
  before drawing conclusions.

**M3 — shared device code + full scenes on Metal** *(landed)*:
- *Landed:* mesh scenes render — OBJ loading (`src/core/mesh_loader`), CPU-built
  six-direction threaded BVH (`src/core/bvh_builder`; SAH split landed later, below), and
  stackless traversal behind the `rt_closest_hit` seam (`src/rhi/raytrace.metal`) with
  GPU residency owned by `RayIntersector`. Glass-bunny (144k tris) renders in both
  modes, bitwise identical; with real traversal cost the wavefront/mega gap collapses
  to 1.14× (4.9 s vs 5.6 s at 800×800×300spp) from 3.5× on the analytic scene.
- *Landed:* per-material-type shade queues (tier-1 of the `get_bxdf` plan) — intersect
  routes paths by material type and resolves emissive hits inline; shading is ONE
  `wf_shade` kernel specialized per type via `rhi::SpecConstant` (Metal function
  constants — the Metal analog of template instantiation), so the material branch
  folds at pipeline creation and each dispatch is divergence-free. First real BSDF
  port in `src/core/bsdf_shared.h`: GGX VNDF conductor (Heitz 2018, from
  microfacet.cu) — `microfacet` materials now honor roughness. Both modes remain
  bitwise identical on cornell and bunny.
- *Landed (pulled forward from M5):* live preview window behind the presentation
  seam. `Device::presentTarget(w,h)` creates a Cocoa window + `CAMetalLayer` and
  returns an RGBA8 buffer; a `present_tonemap` kernel (ACES from
  `src/core/tonemap_shared.h`, shared with the host PNG writer) writes the running
  average into it each iteration, and `Device::present()` pumps events and blits to
  the drawable (returns false on window close / q / Esc). Presentation shares the
  device's single `MTLCommandQueue` with all streams, so commit order + hazard
  tracking give present-after-render for free. `CommandStream::submit()` bounds
  in-flight command buffers (2): without backpressure the CPU encodes the whole
  render ahead of the GPU, which both queues seconds of GPU work (starving
  WindowServer compositing — OS-wide stutter on heavy scenes) and breaks preview
  pacing (wall clock stops tracking render progress, so only the first/last frames
  ever show). The mini loop submits per sample in preview mode and rate-limits
  presents to ~60 Hz (the drawable pool would otherwise throttle fast renders),
  freezes at the last frame until the window closes, and saves a partial image on
  early close.
  Preview is on by default; `--no-preview` keeps the headless path, which stays
  bitwise identical to the pre-GUI renderer. (Since the ImGui overlay landed the
  preview builds its camera basis from the fly camera — orthonormal, unlike the
  scene.cpp quirk headless replicates — so preview matches headless framing only
  for the default horizontal-view scenes.) Window size follows the scene
  since M4 part 2: `presentTarget(w, h)` with a new size resizes the window
  and replaces the target on both backends, and the scene swap reallocates
  the accumulation buffer and wavefront queues; the user cannot drag-resize
  (the render size is the window size).
- *Landed (pulled forward from M5):* interactive ImGui overlay GUI. `present()`
  draws a Dear ImGui pass over the blit (render pass, `loadAction=Load`) when
  `Device::enableGui(draw)` is set. Everything portable lives in `src/core/gui`
  (shared with the future Windows preview): the overlay widgets, the fly camera
  (WASD / drag-look / wheel-dolly, read from ImGui IO), the scene-directory scan,
  and `runPreview` — the interactive-loop *policy* (sample accounting, ~60 Hz
  present pacing, accumulation restart on camera-move / reset / scene-switch,
  command dispatch), which drives renderer-specific work through a `PreviewHooks`
  callback set so it never names a GPU type. Only the Metal renderer backend + a
  small Cocoa→ImGui-IO event shim are in `rhi_metal.mm`, guarded by
  `RHI_ENABLE_IMGUI` so the RHI test targets compile without ImGui; `src/mini`
  keeps just the backend glue (buildSceneGpu, dispatch, param packing). The scene
  dropdown hot-swaps any sibling `.txt` scene (drain GPU → rebuild the scene-tied
  buffers at the fixed session resolution → reset camera + accum); the bindless
  texture heap recycles slots on teardown so repeated swaps don't leak toward the
  1024-slot cap. The interactive loop replaces the freeze-at-last-frame behavior
  in preview mode; headless is untouched (bitwise-identical, verified). We vendor
  only `imgui_impl_metal` (stable) and write the platform shim ourselves to avoid
  `imgui_impl_osx` version drift against the pinned ImGui 1.89.
- *Landed:* bindless textures + environment maps. `Device::textureHeap()` is a
  buffer of 64-bit entries indexed by `Texture::shaderHandle()` — `MTLResourceID`
  on Metal (read directly as `texture2d<float>`, Metal 3 bindless, backend keeps
  every texture resident per dispatch), an array of `cudaTextureObject_t` on CUDA
  in M4 (same layout, same kernel code through the `tex_heap_sample` shim in
  `src/rhi/texture.metal`, fixed bilinear+wrap sampler matching the CUDA
  `cudaTextureDesc`). First consumer: SKYBOX equirect env maps (`src/core/`
  image_loader + envmap_shared) sampled on ray miss, resolved inline in
  `wf_intersect` like emissive hits. Replicated quirk: the CUDA renderer sets
  `stbi_set_flip_vertically_on_load` globally and its `v = asin(y)+0.5` mapping
  assumes bottom-up images — the core loader flips to match. Verified by a
  RhiTest bilinear-sampling unit test plus bunny-skybox rendering both modes
  bitwise identical; cornell output unchanged.
- *Landed:* mesh vertex attributes + OBJ/MTL materials. `core/mesh_loader`
  builds unified vertex arrays (positions/normals/uvs share indices, deduped on
  the OBJ index triple — BVH triangle reordering stays valid); `RtHit` carries
  barycentrics + triangle index and `rt_shading_normal`/`rt_interp_uv`
  interpolate behind the seam (`vnormal` smooth shading; zero normals fall back
  to geometric). `material -1` pulls the OBJ's MTL materials as textured
  diffuse (`MiniMaterial.texIdx` = bindless heap index; sRGB RGBA8 textures,
  deduped by path — 45 MTL entries binding one atlas must not become 45 GPU
  textures, that OOM'd the wavefront queues). The command stream now surfaces
  command-buffer faults instead of rendering black. Verified: dragon-skybox
  (871k tris, vnormal) and lost-empire (textured, per-face MTL materials)
  render in both modes bitwise identical; cornell and bunny byte-unchanged.
- *Landed:* the portable device-code shim (`src/rhi/gpu_portable.h`, designed
  in `docs/portable-device-code.md`) — shaders are now single-source across
  MSL/CUDA/host-C++: kernels declared via GPU_KERNEL signature macros,
  primitives on wave shims, BSDF/traversal/env/tonemap on gpu_* value types
  with 16-byte-true storage types (I-3). `pathtrace/primitives/raytrace`
  became `_gpu.h` files; only `texture.metal` stays per-backend.
  `SharedHostTest` proves host-C++ (the CUDA spelling) and MSL agree
  numerically; GPU renders verified bitwise-unchanged (the sole drift was the
  host PNG writer's ACES codegen, simd→glm: ≤1 LSB on a handful of channels,
  one-time). CUDA execution of the same sources lands in M4.
- *Landed:* the scene loader lives in the core. `src/core/scene_loader`
  parses the full .txt key set (including the spectral REFRIOR_NAMED /
  REFRIOR_RGB / REFRIOR_REAL_NAMED / REFRIOR_IMAG_NAMED parameters, carried
  in `CoreMaterial` for the spectral port) into a backend-neutral `CoreScene`;
  `mini_scene` is deleted and `mini_main` only maps `CoreScene` to its device
  PODs. In the same pass the core BVH builder gained bvh.cpp's bucketed SAH
  split (same constants and float expressions — M4 points the CUDA backend
  here and retires bvh.cpp's copy), and all core loader host math moved from
  Apple simd to glm (`utilityCore`'s exact transform composition,
  `glm::inverse`/`inverseTranspose`), so host math is identical across
  backend hosts in M4; storage types come from the gpu_portable shim.
  Scope notes: glTF did NOT migrate — it is commented-out dead code in
  scene.cpp; PLY migrates with the .json volume scenes in the spectral/volume
  port. One-time golden drift from both changes, quantified: SAH re-ties 5-9
  px per multi-MP image (≤ a few LSB); simd→glm host math ULP-shifts
  transforms/normals, which refraction amplifies (glass dragon: 6.5% of px,
  unbiased ±0, visually identical; others ≤0.1%). Mega and wavefront remain
  bitwise identical to each other on every scene.
- *Landed:* the spectral pipeline — FluoraMini is now a spectral renderer
  like the CUDA path (visually matches img/dragon_spec_0.png). Device side
  in `src/core/spectrum_shared.h` (see that header): float4 spectra,
  visible-wavelength sampling, dense spectra + the sRGB rgb2spec table as
  flat buffers, complex Fresnel, CIE pixel sensor; film matrix derived
  host-side in `src/core/spectra.cpp` from SpectrumConsts data
  (`spectrum_tables.inl`, split out of spectrum_data.cu). The live BxDFs
  ported spectrally into `bsdf_shared.h`: DiffuseBxDF (per-hit RGB->sigmoid
  albedo, textures included), smooth DielectricBxDF with real Fresnel and
  dispersion (named eta terminates secondary wavelengths, glass-Fake
  renders prismatic sparkle), ConductorBxDF on Trowbridge-Reitz with
  measured eta/k (metal-* spectra) — plus PBRT reflectance mode for RGB
  "microfacet" materials the CUDA loader rejects. Emitters and env-map
  texels spectralize as RGB illuminants (D65). Wavefront paths carry
  lambdaU + a dispersion flag (8 B) and recompute wavelengths per stage;
  WfPath grew 80->96 B. Both modes bitwise identical on every scene;
  spectral costs ~16% on the dragon. The "remaining BSDFs" of the earlier
  plan (rough dielectric, metallic workflow) turned out to be dead code in
  the CUDA renderer, like glTF — nothing real remains unported.
- *M3 is complete.* Renderer capability that was never in M3 — NEE/MIS
  integration, volumes/NanoVDB, DOF, denoise — arrives with the CUDA-side
  migration (M4) and Metal-native work (M5).

**Code layout rule:** portable host code (loaders, builders, eventually the renderer
core) lives in `src/core/`; backend seams, device-code files, and primitives in
`src/rhi/`; `src/mini/` is scaffolding that only shrinks and may only *call* the other
two. Adding capability inside `src/mini` is a review flag.

**M4 — CUDA catch-up and parity** *(Windows/CUDA machine; part 1 landed 2026-09-03)*:

- *Landed (part 1):* **the real CUDA backend**, `src/rhi/rhi_cuda.cu` +
  `rhi_cuda_present.cpp` + `rhi_cuda.h`. The Windows build now produces
  `FluoraMini`, `RhiTest` and `SharedHostTest` next to `Fluora`, from the same
  portable core, tests and single-source kernels the macOS build compiles for
  Metal (`rhi::kNativeBackend` picks the backend; `DeviceDesc::shaderSource`
  is only filled for Metal). Verified on an RTX 5080 / CUDA 13.2 / VS 2026:
  RhiTest passes all six primitives (reduce, scan, compact, radix sort,
  wave-aggregated queue push, bilinear heap sampling — the wave shims run on
  cooperative-groups coalesced groups unchanged), SharedHostTest passes host-C++
  vs CUDA value parity on all 21 slots, and FluoraMini renders cornell, bunny-
  skybox, dragon-skybox (871k tris, dispersive glass, 4k env map) and
  lost-empire (textured MTL materials) with **mega ≡ wavefront bitwise** on
  every scene; the interactive preview (GLFW window, ImGui overlay, fly
  camera, scene hot-swap) runs on the same `gui::runPreview` policy as Metal.
  - Kernel registration: `RHI_CUDA_REGISTER_KERNEL(k)` /
    `RHI_CUDA_REGISTER_SPEC(k, index, value)` once per entry point in one `.cu`
    per `_gpu.h` file. Those `.cu` files are generated at build time by
    `cmake/GenerateCudaKernels.cmake` (driven by `fluora_cuda_kernels()` in
    CMakeLists.txt, output under `build/generated/`) from the `GPU_KERNEL`
    declarations and the `GPU_SPEC_INSTANCES(k, index, values...)` annotation
    a specialized kernel carries; the backend registers the primitives
    itself. The thunk is deduced from the kernel's own signature
    (`KernelSig`), so the binding convention is enforced by the type system.
    `GPU_SPEC_CONST` lowers to a template parameter on CUDA (one constant per
    kernel).
  - Indirect dispatch is a per-kernel one-thread launcher + dynamic parallelism
    (`cudaStreamFireAndForget`; parent completion waits for the child, so
    stream order holds). Generated by the macro as a plain `__global__`: nvcc
    does not register `__global__` templates whose template argument is a
    kernel address (verified standalone), so the kernel identity lives in a
    `__device__` helper's template argument. Needs rdc + `cudadevrt` on every
    RHI target. Kernels have external linkage for the same reason, hence the
    `GPU_PRIMITIVES_HELPERS_ONLY` guard in `primitives_gpu.h`.
  - Memory: `Shared` = managed memory. On Windows the host may only touch it
    while no kernel runs — the same "not in flight" rule `Buffer::hostPtr()`
    documents — so the texture heap is a device array with a host shadow
    (written with synchronous H2D copies) rather than managed. Buffer frees are
    deferred behind events recorded on every live stream (rhi_algorithms'
    lifetime note). Streams are blocking streams so the legacy default stream
    (present copy, heap writes) orders after them, the CUDA analog of Metal's
    single queue; `submit()` bounds in-flight work at 2 via an event ring.
  - `texture.metal` → `texture_gpu.h`, the one shared file whose function body
    is per backend (`tex2D` on `cudaTextureObject_t` vs bindless
    `texture2d`), plus a host stub. Test kernels went single-source too
    (`rhi_test_gpu.h`, the probe kernel in `shared_probe.h`).
  - **Fixed what M0–M3 broke:** the main CUDA renderer did not compile —
    `TypedPoolView::kSizes` (a static constexpr array indexed in device code)
    is ODR-used, which nvcc rejects; now a pack-fold `typeSize()`. With that,
    `Fluora.exe` builds and renders bunny-skybox on the M0 handle pools
    (smoke-tested; note its scene paths are `../scenes/...`, so run it from a
    repo subdirectory such as `build/`). The material pool is now uploaded
    once in `Scene::LoadAllMaterialsToGPU` (`Scene::materialPoolDev`) instead
    of per camera move (§8b leak). CMake's Debug `-g -G` block moved after
    `enable_language(CUDA)` (§8b).
  - Shim follow-ups from `portable-device-code.md` §5 closed: NaN-consistent
    vector `min`/`max` (fmin/fmax overloads in the CUDA and host
    personalities), `gpu_load3` as one 16-byte load on CUDA, and the atomics
    note (an aligned volatile 32-bit access *is* the relaxed atomic load/store
    the API promises).
  - **A/B, queues vs megakernel on CUDA** (RTX 5080, headless):

    | scene | spp | mega | wavefront |
    |---|---|---|---|
    | cornell 800×800 | 500 | 0.38 s | 1.22 s (3.2×) |
    | bunny-skybox 1440×1440 | 300 | 1.13 s | 1.41 s (1.24×) |
    | dragon-skybox 1920×1300 | 100 | 1.44 s | 1.44 s (1.0×) |
    | lost-empire 1000×1000 | 100 | 0.30 s | 0.33 s (1.1×) |

    Same shape as the Metal measurements (M2/M3): the per-stage launch and
    queue traffic dominate on trivial scenes and vanish once traversal is the
    cost. Neither mode uses compaction; the M2 decision (queues as the bounce
    mechanism, primitives kept) stands. On CUDA the indirect launch adds one
    dynamic-parallelism hop per stage, which is inside the cornell gap.
  - Parity with Metal: not measured here (no Mac in reach from this machine).
    Mega ≡ wavefront bitwise on both backends is the strongest available
    evidence; the cross-backend golden diff (§9) stays open, with the known
    caveats of MSL fast-math vs nvcc `--fmad` and the shim's tolerance-based
    SharedHostTest.
- *Part 2 — decision (2026-09-03): converge on the portable renderer, do
  not migrate `pathtrace.cu` in place.* The old renderer is ~6k lines built on
  what the RHI forbids (30 `thrust::default_random_engine` sites,
  `TaggedPointer` graphs for lights/media/spectra/phase functions,
  `Distribution2D*` and `cudaTextureObject_t` inside light objects,
  `alloc.new_object` pointer graphs, Thrust scan/scatter/sort in the bounce
  loop); converting all of that would end where FluoraMini already is
  (spectral, single-source, handle-based, wavefront, two backends). What
  FluoraMini lacks is features, so those are ported into the core and the old
  renderer becomes the parity reference until it is retired. Order:
  1. Mac sanity pass on what M4 part 1 touched without a Metal build
     (`texture_gpu.h`, single-source test kernels, `GPU_SPEC_INSTANCES`)
     *(done 2026-09-04: Metal build verified on the owner's Mac, including
     the NEE stage and the final kernel-macro form)*.
  2. **NEE + MIS** *(landed 2026-09-03)*: `rt_occluded` on the ray seam, area
     lights from emissive triangles and analytic objects (the CUDA renderer's
     cube/sphere/triangle sampling schemes), env-map importance sampling with
     the 2D CDF as flat buffers (`core/light_shared.h` device side,
     `core/lights.cpp` host side), a uniform light sampler, BSDF eval/pdf in
     `bsdf_shared.h` (probe slots 21–22 in SharedHostTest), and a shadow-ray
     queue stage in the wavefront loop (`wf_prep_shadow` + `wf_shadow`, 48 B
     per ray, run before the next bounce's intersect). Both modes evaluate one
     shared `miniShadeVertex` (NEE draws, then BSDF draws) and add every
     contribution to the film on its own in the same per-bounce order, so mega
     ≡ wavefront stays **bitwise** on cornell, bunny-skybox, dragon-skybox and
     lost-empire; NEE renders match the naive ones in brightness at a fraction
     of the noise. Cost on CUDA: cornell 200 spp 0.17 → 0.34 s (mega), bunny
     100 spp 0.38 → 0.43 s. Parity with the CUDA MIS renderer is statistical,
     with one deliberate deviation: the env-map sampling pdf uses the equirect
     Jacobian `p(uv) / (2π² cos(latitude))` (PBRT), not the old renderer's
     `p(uv)/4π`. Lights are one-sided for sampling (back-side emission arrives
     via BSDF sampling with full weight), as in the CUDA renderer.
  3. **`.json` scenes and PLY into `core/scene_loader`** *(landed
     2026-09-04)*: `loadScene()` dispatches on the extension; `loadJsonScene`
     mirrors `Scene::loadJSON` key for key (named materials with const/named
     eta and k, REFL rgb/texture, NORMAL_MAP path, Background SCALE/MAXRGB
     baked into the env texels, Camera LENS_RADIUS/FOCAL_LEN/MEDIUM, Media,
     MediumInterfaces, `geometry_cube`/`geometry_sphere`, `model_inline`,
     `model_ply` dispatched on the real extension since scenes label OBJ
     files that way). `mesh_loader` gained tinyply PLY and inline meshes,
     `image_loader` EXR via tinyexr (linear base color sRGB-encoded into the
     8-bit heap format). Media and DOF are carried in `CoreScene` for steps
     4–5; a medium interface rides on a per-(material, inside, outside)
     material clone because triangles carry one id, and interface-only
     objects get a `CoreMaterialType::Interface` that FluoraMini maps to an
     index-matched dielectric (exactly pass-through, but it counts as a
     bounce and occludes shadow rays — both go away with the media step,
     where shadow rays need transmittance anyway). Verified on CUDA:
     bunny.json, matchbulb.json (73k PLY tris, EXR textures, Ag spectra, env
     scale 0.2) and volumetric-caustics.json render with mega ≡ wavefront
     bitwise; the .txt path is unchanged (cornell bitwise across modes,
     FurnaceTest passes).
  4. Volumes: homogeneous first, then a NanoVDB-under-MSL spike before
     committing to grid media on Metal. **Homogeneous media landed
     2026-09-04** (`core/medium_shared.h` + `miniTrace` in
     `pathtrace_gpu.h`): every path segment is traced through surfaces and
     media together — a homogeneous medium is delta-tracked at the hero
     wavelength (its majorant is sigma_t, so a sampled collision is a real
     absorb/scatter event), a real scatter becomes a phase-function vertex
     (HG, routed to a fourth `wf_shade` specialization `MINI_MAT_MEDIUM` in
     wavefront mode), and surfaceless `MINI_MAT_INTERFACE` hits switch the
     medium and continue without counting a bounce (PBRT rule: leaving along
     the geometric normal enters OUTSIDE; also applied after BSDF scatters on
     surfaces that carry an interface, e.g. the matchbulb glass). Spectral
     MIS is PBRT-v4's: paths carry `r`, the per-wavelength pdf ratio to the
     hero, and every contribution divides by its average; with no media `r`
     is exactly 1 and media-free renders stayed bitwise identical to the
     pre-change binary (cornell, bunny-skybox, bunny.json, both modes).
     Shadow rays carry the spectral contribution instead of film RGB and
     take the analytic Beer-Lambert transmittance through the media they
     cross (passing interfaces), so there is no ratio-tracking `r_l` term;
     media-free scenes keep the any-hit path. Light/BSDF MIS stays the
     scalar power heuristic (the old volume integrator used the balance
     form). The FurnaceTest gained a chromatic (sigma_s 0.2/0.4/0.8),
     anisotropic (g 0.3), scattering-only fog box that must vanish: it does
     (object/background 1.001/1.000/0.996 per channel at 256 spp), bitwise
     across modes. Not rendered: medium emission (LESCALE — the .json
     homogeneous media never carried an Le spectrum in the old loader
     either) and NanoVDB grids, which upload as empty media so their
     interfaces stay pass-through (matchbulb renders, flame missing).
     Camera MEDIUM is honoured; volumetric-caustics.json declares the camera
     inside `smoke_medium` while its box is INSIDE smoke / OUTSIDE vacuum, so
     under the PBRT rule paths that exit the box leave the fog — the scene
     file's inconsistency, not the tracker's.
     **NanoVDB grids landed 2026-09-04.** The MSL spike was settled by
     inspection rather than a port: the vendored NanoVDB (32.3) has CUDA,
     OpenCL and host personalities but no Metal one, and the runtime MSL
     build concatenates sources without resolving `#include`, so its 7k-line
     header cannot be compiled there. Grids are therefore read on the host
     with NanoVDB's own reader (`core/volume_loader.cpp`, `NANOVDB_USE_ZIP` —
     the scene files are ZIP-compressed, so FluoraMini links zlib on both
     platforms) and re-bricked into a pointer-free layout the same device
     code samples everywhere: 8^3 bricks (NanoVDB's leaf size; leaves copy
     across, tile values fill constant bricks) behind a dense brick table
     over the index-space bounding box, a per-brick majorant (max over the
     brick and its 26 neighbours, since trilinear samples reach one voxel
     across), two flat buffers (`vol.table` uints, `vol.data` floats) with
     offsets in `MediumGpu`. The tracker became PBRT-v4's general
     `SampleT_maj`: a majorant-segment iterator (one segment for a
     homogeneous medium, a 3D DDA over bricks for a grid) feeding a
     delta-tracking loop with null collisions, emission at every collision
     (temperature grid → peak-normalized blackbody × LESCALE), and ratio
     tracking with a per-shadow-ray forked RNG stream for transmittance
     through grids (homogeneous media keep the analytic exponential). The
     homogeneous path is a special case of the same loop, and the furnace
     ratios are unchanged to the last digit. Verified: teapot_cloud.json
     (1.9M active voxels, 10 MiB), ground_explosion.json (27M active voxels,
     129 MiB, 6900 K peak) and matchbulb.json (flame inside the glass bulb)
     render bitwise across modes — matchbulb only with
     `-DFLUORA_CUDA_SAFE_MATH=ON`; under `-fmad=true` about 0.8% of its
     pixels take a different collision branch, the documented fast-math
     caveat, while the other two happen to stay bitwise. Memory is the
     obvious next lever (float voxels; half or a coarser majorant grid would
     halve the explosion), and RGBGridMedium/GridMedium (`.json` types other
     than nanovdb/homogeneous) are still not parsed.
  5. DOF (camera ray generation) and denoise (OIDN behind the present seam on
     CUDA; Metal stays M5).
  6. Retire `pathtrace.cu`, the integrators, `bvh.cpp`, `main.cpp`/
     `preview.cpp` once every scene in `scenes/` renders through FluoraMini;
     `Fluora.exe` becomes FluoraMini.
  Still from the original list: cross-backend golden diffs (needs both
  machines plus fast-math/fma flag matching). Queues vs compaction is settled
  by the A/B above (queues stay).

**M5 — Metal-native features**: hardware RT path; preview upgrades (the basic
window landed in M3, the ImGui overlay in M3.x, scene-driven window resize in
M4 part 2); OIDN-on-Metal denoise.

## 7. Build integration

`CMakeLists.txt` branches at the top: on APPLE it builds only `FluoraMini`
(C++/Obj-C++, links Metal + Foundation, no CUDA/GL/GLFW dependencies) and returns.
Otherwise it builds the CUDA renderer `Fluora` plus, since M4, the same three RHI
targets on the CUDA backend through the `fluora_rhi_target()` helper (nvcc for the
`.cu` registration units, `CUDA_SEPARABLE_COMPILATION` + `cudadevrt` for the
dynamic-parallelism indirect dispatch, GLEW/GLFW/OpenGL for presentation). The mini
target compiles `src/stb.cpp` for PNG writing and defines the `*_SHADER_DIR` paths so
MSL sources are loaded from the source tree at runtime on Metal (dev-time convenience;
unused on CUDA). Gotcha: the Visual Studio CUDA integration does not always re-run
nvcc on a `.cu` whose *included headers* changed (seen with `rhi_cuda.h`); `touch`
the registration `.cu` files when a launch-thunk change seems not to take.

## 8. Risks / open questions

- **Struct layout drift** between MSL and host (I-3): mitigated by the single shared
  header + keeping fields 16-byte-friendly; add `static_assert` size checks host-side
  as structs grow.
- **Runtime MSL compile** has no include resolution; concatenation is fine for two
  files, will not scale past M1 → move to metallib build step in M3.
- **M4 bit-identical goal** may be too strict once compaction is replaced by queues
  (queue push order is nondeterministic, changing float accumulation order); fall back
  to statistical image comparison.
- **Spectral rendering on Metal** (M3): `SampledSpectrum` math is portable C++, but the
  dense spectra tables live in managed memory today — needs the handle/pool treatment.
- Old glm in `external/include` is not used by the Mac targets (Apple `simd` instead);
  revisit when M3 shims unify math types.

## 8b. Code-review follow-ups (2026-07-22)

Confirmed findings from the pre-push review of M0–M3, deferred to the milestones
that naturally own them (quick loader/robustness fixes were applied directly):

- **CUDA: material pool re-uploaded per camera move** *(resolved in M4)*:
  `Scene::LoadAllMaterialsToGPU` uploads once into `Scene::materialPoolDev`;
  `pathtraceInit` just copies the view.
- **CMake: Windows Debug drops default nvcc flags** *(resolved in M4)*: the
  early `set(CMAKE_CUDA_FLAGS ...)` block is gone; the post-`enable_language`
  `CMAKE_CUDA_FLAGS_DEBUG` append (`-G`, on top of CMake's default `-g`) is
  the only Debug device flag.
- **Shader-source concat lists**: the "move to metallib in M3" plan (§8) is
  overdue — mini_main carries a 10-file dependency-ordered list, RhiTest a
  second 3-file one, from absolute configure-time paths. Do alongside the M3
  loader migration: a build-time metallib step or one shared concat helper
  owned by the rhi backend.
- **Wavefront queue mapping restated ~6×** (`WF_COUNT_*`/`WF_ARG_*` defines,
  prep kernels' hardcoded zeroing/twin arrays, wf_intersect's if/else + buffer
  slots, mini_main's shadePasses): derive counter/slot as BASE + matType and
  loop over a type count before adding the next BSDF queue; `WfCtl.argSlot` is
  dead — remove it.
- **Duplicate BVH builder** *(resolved in M3)*: `core/bvh_builder` now carries
  bvh.cpp's bucketed SAH split (ported, since `sceneStructs.h` pulling
  `cuda_runtime.h` blocks literal reuse); M4 swaps the CUDA renderer onto the
  core builder and deletes bvh.cpp's copy.
- **`mini_scene` capability creep** *(resolved in M3)*: `mini_scene` is
  deleted; parsing, texture-path dedupe, MTL→material mapping, and SKYBOX
  handling live in `src/core/scene_loader`.

## 9. Verification

- M1: visual check against `img/REFERENCE_cornell.5000samp.png` (framing, colors,
  shadows); CPU smoke test of pool machinery exists from M0.
- M2+: golden-image diffs on `scenes/` (cornell, bunny, sponza) between backends and
  against pre-RHI renders; per-stage buffer dumps when they diverge.
