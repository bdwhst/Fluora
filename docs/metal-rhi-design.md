# Metal RHI: design and migration plan

Status: M0–M3 landed (FluoraMini renders full scenes spectrally on Metal);
next is M4, the CUDA catch-up · Owner: bdwhst · Last updated: 2026-09-01

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

Defined in `src/rhi/rhi.h`; one implementation file per backend
(`rhi_metal.mm` real, `rhi_cuda.h` sketch until the CUDA migration starts).

| Concept | CUDA backend | Metal backend |
|---|---|---|
| `Device` | context + kernel registry | `MTLDevice` + `MTLLibrary` |
| `Buffer` | `cudaMalloc`/`cudaMallocManaged` | `MTLBuffer` (Private/Shared) |
| `Texture` | `cudaTextureObject_t` | `MTLTexture` + sampler, bindless heap index (landed) |
| `ComputePipeline` | registered launch thunk (named) | `MTLComputePipelineState` |
| `CommandStream` | `cudaStream_t` | `MTLCommandBuffer` + compute encoders |
| `RayIntersector` | CPU-built MTBVH upload | same (M3) or `MTLAccelerationStructure` (M5) |
| present | GL PBO interop (as today) | `CAMetalLayer` drawable (landed) |

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
  bitwise identical to preview-mode output. ImGui and resize stay in M5.
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

**M4 — CUDA catch-up and parity** *(needs the Windows/CUDA machine, ~2026-07-27+)*:
build the CUDA renderer and fix anything M0–M3 broke (starting with the unverified
materials handle conversion); implement `rhi_cuda` for real; migrate `pathtrace.cu`
onto the RHI (convert lights/media/spectra to handle pools, replace Thrust with the
M2 queues/primitives); A/B queues vs compaction on CUDA; golden-image parity between
backends and against pre-migration renders.

**M5 — Metal-native features**: hardware RT path; preview upgrades (ImGui Metal
backend, window resize — the basic window landed in M3); OIDN-on-Metal denoise.

## 7. Build integration

`CMakeLists.txt` branches at the top: on APPLE it builds only `FluoraMini`
(C++/Obj-C++, links Metal + Foundation, no CUDA/GL/GLFW dependencies) and returns; the
CUDA renderer path is untouched otherwise (`enable_language(CUDA)` moved out of
`project()`). The mini target compiles `src/stb.cpp` for PNG writing and defines
`MINI_SHADER_DIR` so MSL sources are loaded from the source tree at runtime (dev-time
convenience; fine until M4 packaging).

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

- **CUDA: material pool re-uploaded per camera move** (`pathtrace.cu`,
  `materialPool.upload(alloc)` in `pathtraceInit`): every `camchanged` reset
  bump-allocates a fresh device pool that the monotonic allocator never
  reclaims — unbounded growth while dragging the camera. Fix in M4 (first CUDA
  build): upload once at scene load, or before `pathtraceInit` re-runs.
- **CMake: Windows Debug drops default nvcc flags**: the Debug-mode
  `set(CMAKE_CUDA_FLAGS ... "-g -G")` near the top now runs before
  `enable_language(CUDA)` (CUDA left `project()` for the Apple branch); under
  CMP0126 the normal variable shadows the compiler-detected defaults on a fresh
  configure. Fix in M4: append `-g -G` after `enable_language(CUDA)` (or use
  `add_compile_options` with a CUDA generator expression).
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
