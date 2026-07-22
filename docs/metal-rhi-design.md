# Metal RHI: design and migration plan

Status: M0–M2 landed; M3 in progress (mesh scenes render on macOS via threaded-BVH
traversal behind the RayIntersector seam) · Owner: bdwhst · Last updated: 2026-07-22

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
| `Texture` | `cudaTextureObject_t` | `MTLTexture` + sampler, bindless heap index |
| `ComputePipeline` | registered launch thunk (named) | `MTLComputePipelineState` |
| `CommandStream` | `cudaStream_t` | `MTLCommandBuffer` + compute encoders |
| `RayIntersector` | CPU-built MTBVH upload | same (M3) or `MTLAccelerationStructure` (M5) |
| present | GL PBO interop (as today) | `CAMetalLayer` drawable (M5) |

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

**M3 — shared device code + full scenes on Metal** *(Mac, in progress)*:
- *Landed:* mesh scenes render — OBJ loading (`src/core/mesh_loader`), CPU-built
  six-direction threaded BVH (`src/core/bvh_builder`, median split; SAH later), and
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
- *Remaining:* math/RNG/texture shims for sharing device code with CUDA; port the
  remaining real BSDFs (rough dielectric, metallic workflow) and the spectral
  pipeline; make the real scene loader host-portable (migrating it into `src/core`) so
  `mini_scene` dies. This is the long pole.

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

**M5 — Metal-native features**: hardware RT path; interactive preview
(MetalKit + ImGui Metal backend) behind the `presentTarget` seam; OIDN-on-Metal
denoise.

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

## 9. Verification

- M1: visual check against `img/REFERENCE_cornell.5000samp.png` (framing, colors,
  shadows); CPU smoke test of pool machinery exists from M0.
- M2+: golden-image diffs on `scenes/` (cornell, bunny, sponza) between backends and
  against pre-RHI renders; per-stage buffer dumps when they diverge.
