# Portable device code: shim layer scope & design, Slang adoption scope

Companion to `metal-rhi-design.md` (M3 item "math/RNG shims for sharing device
code with CUDA"). Decision context: a custom shader language + transpiler was
considered and rejected — MSL and CUDA are both C++ dialects, so the delta is a
closed, mechanical list that a thin shim covers at a fraction of the cost of
maintaining a compiler. This doc scopes that shim, designs it, and scopes (but
does not green-light) the escalation path: adopting Slang.

## 1. Inventory — what shared code actually uses today

Measured on the current tree (grep counts, 2026-07-22):

| Construct | Where | MSL form | CUDA form |
|---|---|---|---|
| Address-space refs (`thread T&`) | bsdf_shared (4), raytrace (10), pathtrace.metal (64) | required | absent |
| `device`/`constant` pointers | all kernels + rt_* helpers | required | plain pointers |
| Vector math namespace | ~40 `metal::` calls in bsdf_shared alone | `metal::sqrt` … | glm / CUDA intrinsics |
| Vector types | everywhere | `metal::float3` (16 B), `packed_float3` (12 B) | `glm::vec3` (12 B) |
| Swizzles | pathtrace.metal (`.xyz` on float4) | native | glm needs `GLM_FORCE_SWIZZLE` (heavy) |
| π constant | bsdf_shared | `M_PI_F` | `math::pi` (mathUtils.h) |
| RNG | MiniRng (PCG hash) in pathtrace.metal | hand-rolled | `thrust::default_random_engine` |
| Device atomics | primitives.metal, wf kernels | `atomic_uint` + `*_explicit` | `cuda::atomic` / `atomicAdd` |
| Subgroup ops | primitives.metal (simd_prefix_exclusive_sum, simd_broadcast_first), threadgroup memory + barriers | simdgroup builtins | `__shfl_*` / CUB |
| Kernel entries + bindings | every kernel | `[[buffer(N)]]`, `[[thread_position_in_grid]]` | `__global__`, `threadIdx` |
| Pipeline specialization | wf_shade | function constants | template instantiation |
| Texture sampling | wf kernels via `tex_heap_sample` | bindless heap read | `tex2D` on `cudaTextureObject_t` |

Already-portable precedents in tree: `accel_shared.h`, `tonemap_shared.h`,
`envmap_shared.h`, `mini_shared.h`, `primitives_shared.h` each carry a private
`#ifdef __METAL_VERSION__` typedef block. The shim consolidates these.

## 2. Shim layer — scope

**In scope (tier A — the shim header):** scalar/vector math functions, vector
type aliases, address-space macros, constants, a shared PCG RNG, and the
`float4`→`float3` swizzle helper. This is sufficient for everything in
`src/core/*_shared.h`, the bodies of `rt_*` traversal helpers, and the BSDF /
scatter / camera logic — i.e. all *radiometry-affecting* code, which is exactly
the code that must be bit-comparable across backends in M4.

**Hard requirement: shaders are written once.** No kernel, BSDF, or primitive
is authored twice for the two platforms. The only per-backend device code
allowed is the shim layer itself, the ~5-line body of `tex_heap_sample`
(genuinely different APIs behind one name), and optional backend-specific fast
paths behind an existing seam (e.g. an M5 `intersection_query` traversal behind
`rt_closest_hit`, where the shared compute traversal remains the reference).

**In scope (tier B — kernel signature macros):** entry points are single-source
too. The RHI already fixes the binding convention (params at buffer(0),
resource i at buffer(i+1)), so the entry syntax is mechanically derivable from
a declaration macro:

```cpp
GPU_KERNEL(wf_shade)(GPU_KERNEL_PARAMS(WfCtl)
    GPU_BUFFER(gpu_atomic_uint,    counts,    1)
    GPU_BUFFER(const WfPath,       queue,     2)
    GPU_BUFFER(WfPath,             raysOut,   3)
    GPU_BUFFER(const MiniMaterial, materials, 4)
    GPU_TID_1D)
{
    uint tid = GPU_GLOBAL_ID_X;  // MSL: bound param; CUDA: blockIdx*blockDim+threadIdx
    ...
}
```

MSL expansion: `kernel void wf_shade(constant WfCtl& … [[buffer(0)]], device …
[[buffer(N)]], uint [[thread_position_in_grid]])`. CUDA expansion:
`__global__ void wf_shade(const WfCtl, …*)` (thread id computed in
`GPU_GLOBAL_ID_X`). Explicit slot numbers double as binding documentation.
Specialization: `GPU_SPEC_CONST(uint, kShadeMatType, 0)` lowers to
`[[function_constant(0)]]` on MSL and a template parameter on CUDA — exactly
the `rhi::SpecConstant` lowering `rhi.h` already promises.

**In scope (tier C — wave/threadgroup shims, makes the primitives
single-source):** both platforms execute 32-wide (Apple simdgroups, NVIDIA
warps — `primitives.metal` already assumes 32), and the primitives use a
five-entry op list, so they shim rather than fork: `gpu_wave_prefix_sum` ↔
`simd_prefix_exclusive_sum` / `__shfl_up` loop (or CUB warp scan),
`gpu_wave_broadcast_first` ↔ `simd_broadcast_first` / `__shfl_sync(…, 0)`,
`gpu_wave_sum` ↔ `simd_sum` / `__reduce_add_sync`, `GPU_SHARED` ↔
`threadgroup` / `__shared__`, `gpu_barrier()` ↔ `threadgroup_barrier` /
`__syncthreads()`. RhiTest then validates both backends against the same CPU
references. (Fallback if a wave shim proves subtly divergent in M4: that one
primitive may temporarily go CUB-backed per invariant I-2 — divergent
implementations behind one name are a measured exception, never the default.)

**Explicitly out of scope:**
- Texture sampling internals and pipeline-creation host code: already
  abstracted (`tex_heap_sample`, `rhi::SpecConstant`).
- Spectral types: the SampledSpectrum port is its own M3 item; it will be
  written *on top of* the shim from day one.
- Anything host-side: the shim is device-code-only.

**RNG decision folded in:** the shared header adopts MiniRng's PCG (it already
defines both backends' target behavior — mini images are the golden reference).
The CUDA renderer swaps `thrust::default_random_engine` for it in M4. This is
radiometry-affecting: CUDA golden images will change at that swap; plan the M4
comparison accordingly (compare against Metal renders, not pre-swap CUDA ones).

## 3. Shim layer — design

One header, owned by the backend-seam layer: **`src/rhi/gpu_portable.h`**.
Concatenated first for runtime MSL; `#include`-able everywhere else. Contents:

```cpp
// ---- personality selection ----
#if defined(__METAL_VERSION__)
  #define GPU_DEVICE   device      // address-space qualifiers
  #define GPU_THREAD   thread
  #define GPU_FN       inline
  typedef metal::float2 gpu_float2;   // 16B float3: matches MSL layout rules
  typedef metal::float3 gpu_float3;
  typedef metal::float4 gpu_float4;
  typedef metal::packed_float3 gpu_packed3;  // 12B, pairs with glm::vec3
  using namespace metal;   // unqualified sqrt/dot/… resolve to metal::
#elif defined(__CUDACC__)
  #define GPU_DEVICE
  #define GPU_THREAD
  // __device__ only, NOT __host__ __device__: shared code calls rsqrt and
  // float min/max, which nvcc provides as device-only builtins; a host pass
  // would fail. Host-side compilation is the third personality's job.
  #define GPU_FN       __device__ inline
  typedef glm::vec2 gpu_float2;  typedef glm::vec3 gpu_float3;  // glm is CUDA-safe,
  typedef glm::vec4 gpu_float4;  typedef glm::vec3 gpu_packed3; // already in tree
  // vector math via ADL on glm types; scalar via CUDA builtins
#else  // host C++ (tests, loaders) — CUDA spelling with GPU_FN = inline,
       // plus host definitions of the scalar gaps (rsqrt, float min/max)
#endif

#define GPU_PI 3.14159265358979323846f
GPU_FN gpu_float3 gpu_xyz(gpu_float4 v);          // the one swizzle we use
struct GpuRng { unsigned state; };                 // MiniRng's PCG, verbatim
GPU_FN float gpu_rand(GPU_THREAD GpuRng& r);
```

Shared code then reads:

```cpp
GPU_FN bool bsdf_sample_lambert(gpu_float3 rgb, gpu_float3 nF, float u1, float u2,
                                GPU_THREAD gpu_float3& rd,
                                GPU_THREAD gpu_float3& throughput);
```

Design rules (enforceable by eyeball / the host test):
1. Shared headers use only `gpu_*` types, unqualified math calls, `GPU_*`
   macros, and value/ref parameters. No raw `metal::`, no `thread`, no glm
   spelled out, no swizzles beyond `gpu_xyz`.
2. Structs shared with the host stay 16-byte-friendly (invariant I-3), using
   `gpu_packed3` where the 12-byte layout is load-bearing (WfPath).
3. Kernels are declared through the signature macros (§2 tier B) — raw
   `kernel void` / `__global__` in renderer code is a review flag.

**Verification (Mac-only, before the CUDA machine returns):** a `SharedHostTest`
target compiles every shared header as plain host C++ in the CUDA/glm
personality and asserts a table of known input→output values (BSDF samples, RNG
sequence, equirect UVs, ACES) matches values captured from the Metal path via
RhiTest. This proves both personalities parse *and agree numerically* — the
whole point of the layer. It also becomes the regression net for M4.

**Migration plan (each step lands green, mini images bitwise-unchanged):**
1. `gpu_portable.h` + `SharedHostTest` scaffold. *(small)*
2. Convert `bsdf_shared.h` (the heaviest `metal::` user) + value tests. *(medium)*
3. Convert `tonemap_shared.h`, `envmap_shared.h`, `accel_shared.h`, retiring
   their private typedef blocks (`rt_float3`, `tm_float3`, `env_float3`). *(small)*
4. Convert `raytrace.metal` helper bodies (traversal becomes shareable; whether
   CUDA *uses* it is the M4 BVH-unification decision, §8b of the design doc). *(medium)*
5. Extract MiniRng → `GpuRng`, camera-ray + scatter logic out of
   `pathtrace.metal` into a shared header. *(medium)*
6. Kernel signature macros + convert `pathtrace.metal`'s kernels; the file
   becomes fully single-source (rename away from `.metal`). *(medium)*
7. Wave/threadgroup shims + convert `primitives.metal`; RhiTest unchanged and
   green. *(medium; CUDA execution deferred to M4 like everything else)*

Total: roughly one-and-a-half M3-part-sized chunks (the textures chunk plus
half again — steps 6–7 are the increment the single-source requirement adds).
Risks: low — every step is mechanically verifiable against bitwise-identical
mini renders; the one semantic risk (MSL fast-math vs nvcc float contraction
producing different bits in M4) is a known M4 issue independent of this layer,
mitigated by compiling both with matching fast-math/fma flags when parity runs.

## 3b. Expressiveness — research-grade algorithms under the shim

The shim is a spelling convention over full C++, not a DSL: it does not shrink
what device code can express. The real constraint is the MSL∩CUDA intersection,
which any dual-platform approach shares: **no recursion** (iterative tree
walks — GPU practice anyway), **no device-side malloc** (pools + atomic
counters), **no virtual dispatch** (TaggedPointer/queue routing — already
doctrine). Worked example, SIGGRAPH-class path guiding (Müller SD-trees):
traversal = index-linked nodes over `GPU_BUFFER`s like the threaded BVH
(invariant I-1 forces the guiding structure GPU-portable from day one);
radiance splatting = one shim addition (`gpu_atomic_float`, exists on both);
sample warping + guide/BSDF one-sample MIS = shared math + `WfPath` fields;
per-iteration refinement = host-side or via the shared primitives. The host
personality doubles as a research tool: guiding logic debugs in a CPU unit
test with a real debugger, then runs on GPU unchanged.

Escape hatches are graded — pay only for the divergent piece, never contort
the algorithm: `#ifdef` island inside shared code → per-backend kernel behind
one name (invariant I-2, measured exception) → a new RHI seam (the
`RayIntersector` pattern) for genuinely divergent subsystems. Known case of
the last kind: neural components (NIS / neural radiance caches) — MLP
train/infer is tiny-cuda-nn on CUDA vs MPS/`simdgroup_matrix` on Metal, per
backend under any language choice, Slang included.

## 4. Slang adoption — scope only (not approved, not started)

[Slang](https://shader-slang.org) (Khronos-hosted, ex-NVIDIA) is the one
existing shading language with both **CUDA C++** and **Metal** backends. It
would replace the shim + concat with a real module system, and the wrapper
convention with single-source kernels. Scoped work if we ever pull the trigger:

1. **Toolchain**: obtain/pin `slangc`; CMake custom commands compiling shader
   modules per backend (Metal source or metallib on macOS, `.cu` into the nvcc
   build on Windows); prelude/runtime glue (`slang-cuda-prelude.h`). *(medium)*
2. **Binding model**: map our slot convention (params at buffer(0), resource i
   at buffer(i+1)) onto Slang parameter blocks, or adopt Slang reflection to
   drive `rhi` binding. Touches `rhi.h`, both backends, every dispatch site.
   *(medium, the riskiest interface change)*
3. **Port shared device code** to Slang syntax (HLSL-flavored: `float3`,
   `RWStructuredBuffer`, generics). Mechanical but total — everything §3
   migrates piecemeal, this rewrites at once. *(large)*
4. **Specialization**: `rhi::SpecConstant` → Slang link-time specialization /
   specialization constants on both targets. *(small)*
5. **Feature-risk spikes** (do these FIRST, they can kill the plan). Verified
   against Slang docs 2026-07: the portable bindless idiom
   (`DescriptorHandle<T>` heap indexing) is **explicitly unsupported on Metal
   and CUDA** — "on targets where that representation is unavailable (Metal,
   CUDA, CPU), using the syntax is diagnosed at compile time"; it degrades to
   plain `T`. That degradation matches CUDA naturally (flat memory model,
   textures are non-opaque values — `cudaTextureObject_t` arrays just work),
   but on Metal our MTLResourceID-in-a-buffer heap would depend on Slang
   lowering textures-in-device-memory through argument buffers, which is
   undocumented (docs only promise argument buffers for `ParameterBlock`, and
   note "Metal does not support arrays of buffers as of version 3.1"). Spike:
   prove a device-memory texture array round-trips on the Metal target, or the
   texture heap needs a per-backend escape hatch under Slang too. Also
   verified: `RaytracingAccelerationStructure` is **not supported on Slang's
   Metal target** — Slang would not deliver portable hardware RT for M5; that
   stays behind our `RayIntersector` seam regardless. Remaining spikes:
   wave/simdgroup intrinsics parity for the primitives; device-side atomics
   semantics. *(small each, gating)*
6. **Re-verification**: full golden-image pass on both backends; debugging
   story check (Metal shader debugger against slangc-generated source). *(small
   runs, real risk)*

Rough total: comparable to all of M2 plus half of M3 — an order of magnitude
more than the shim. Ongoing costs: slangc version churn, generated-code
debugging, one more toolchain on both platforms.

**Adoption triggers** (revisit at end of M4; adopt only if one fires):
- The shim's rule 1 stops holding — shared code genuinely needs divergent
  semantics (e.g. wave ops inside radiometry code) that macros can't paper over.
- The spectral port produces heavily templated shared code where dual-compiler
  error surfaces cost real time.
- A third backend appears (e.g. Vulkan/DX for a non-CUDA Windows preview) —
  at three targets the language pays for itself.

**Decision for now:** build the shim (§3). It is the M3 plan of record; every
piece of it (types, wrapper convention, value tests) remains necessary-or-useful
groundwork even under a later Slang migration.
