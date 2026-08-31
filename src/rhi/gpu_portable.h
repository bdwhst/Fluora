#ifndef RHI_GPU_PORTABLE_H
#define RHI_GPU_PORTABLE_H
// =============================================================================
// The portable device-code shim (docs/portable-device-code.md §3).
//
// Shaders in this codebase are written ONCE and compiled by three compilers:
//   MSL      (__METAL_VERSION__)  — runtime-concatenated, this header first;
//                                   #include lines below are inert under MSL.
//   CUDA     (__CUDACC__)         — regular #include; execution verified in M4.
//   host C++ (anything else)      — SharedHostTest value-parity + CPU debugging.
//
// Three type roles, chosen by layout needs (invariant I-3):
//   gpu_float3        value/math type; layout-free (glm::vec3 off-Metal).
//   gpu_storage3/4x4  16-byte-true storage for host<->device structs & buffers
//                     (MSL float3 / simd on Apple hosts / alignas(16) on CUDA).
//                     Load into value types with gpu_load3() before math.
//   gpu_packed3       12-byte packed storage (WfPath); glm::vec3 matches it.
//
// Math calls in shared code are UNQUALIFIED (sqrt, dot, normalize, fabs, ...):
// MSL resolves them via `using namespace metal`, glm types via ADL, scalars via
// <cmath>/CUDA builtins (host-only gaps defined below). Rules: fabs (never abs)
// for scalar floats; no swizzles except gpu_xyz(); no recursion; no device
// malloc; kernels declared only through the GPU_KERNEL macros.
// =============================================================================

#if defined(__METAL_VERSION__)
// ---------------------------------------------------------------------------
// MSL personality
// ---------------------------------------------------------------------------
#include <metal_stdlib>
using namespace metal;

// GPU_FN is the per-backend function QUALIFIER only (__device__ on CUDA,
// nothing here or on host) — like defines.h's GPU_FUNC. Definition sites
// spell `GPU_FN inline` explicitly: the inline is ODR linkage for functions
// defined in shared headers included by many host/CUDA TUs, and it stays
// visible rather than hiding in the macro.
#define GPU_FN
#define GPU_DEVICE device
#define GPU_THREAD thread
#define GPU_SHARED threadgroup        // variable declarations
#define GPU_SHARED_SPACE threadgroup  // pointer/reference qualifier
#define GPU_PARAMS_REF(T) constant T&

typedef float2 gpu_float2;
typedef float3 gpu_float3;
typedef float4 gpu_float4;
typedef uint2 gpu_uint2;
typedef uint4 gpu_uint4;
typedef packed_float3 gpu_packed3;
typedef float3 gpu_storage3;
typedef float4x4 gpu_storage4x4;
typedef atomic_uint gpu_atomic_uint;

GPU_FN inline gpu_float3 gpu_load3(gpu_storage3 v) { return v; }
GPU_FN inline gpu_float3 gpu_xyz(gpu_float4 v) { return v.xyz; }
GPU_FN inline bool gpu_all_finite(gpu_float3 v) { return all(isfinite(v)); }

typedef uchar4 gpu_uchar4;
GPU_FN inline gpu_uchar4 gpu_make_uchar4(uchar r, uchar g, uchar b, uchar a)
{
    return uchar4(r, g, b, a);
}

// Atomics: relaxed device-scope, the only ordering the renderer uses.
GPU_FN inline uint gpu_atomic_load(GPU_DEVICE gpu_atomic_uint* p)
{
    return atomic_load_explicit(p, memory_order_relaxed);
}
GPU_FN inline void gpu_atomic_store(GPU_DEVICE gpu_atomic_uint* p, uint v)
{
    atomic_store_explicit(p, v, memory_order_relaxed);
}
GPU_FN inline uint gpu_atomic_fetch_add(GPU_DEVICE gpu_atomic_uint* p, uint v)
{
    return atomic_fetch_add_explicit(p, v, memory_order_relaxed);
}
GPU_FN inline uint gpu_atomic_load(GPU_SHARED_SPACE gpu_atomic_uint* p)
{
    return atomic_load_explicit(p, memory_order_relaxed);
}
GPU_FN inline void gpu_atomic_store(GPU_SHARED_SPACE gpu_atomic_uint* p, uint v)
{
    atomic_store_explicit(p, v, memory_order_relaxed);
}
GPU_FN inline uint gpu_atomic_fetch_add(GPU_SHARED_SPACE gpu_atomic_uint* p, uint v)
{
    return atomic_fetch_add_explicit(p, v, memory_order_relaxed);
}

// Wave (32-wide on both platforms) + threadgroup ops. Capture the active
// lanes ONCE with gpu_wave_active() and pass the handle to every op in that
// scope: on CUDA a fresh coalesced group after a divergent branch has no
// reconvergence guarantee (Volta+ ITS) and can exclude the leader lane, so
// per-op group capture would corrupt e.g. the queue-alloc broadcast. On MSL
// the handle is an empty struct and the ops lower to plain simd_* builtins.
#define gpu_barrier() threadgroup_barrier(mem_flags::mem_threadgroup)
struct gpu_wave_t {};
GPU_FN inline gpu_wave_t gpu_wave_active() { return gpu_wave_t{}; }
GPU_FN inline uint gpu_wave_prefix_sum(gpu_wave_t, uint v) { return simd_prefix_exclusive_sum(v); }
GPU_FN inline uint gpu_wave_sum(gpu_wave_t, uint v) { return simd_sum(v); }
GPU_FN inline bool gpu_wave_is_first(gpu_wave_t) { return simd_is_first(); }
GPU_FN inline uint gpu_wave_broadcast_first(gpu_wave_t, uint v) { return simd_broadcast_first(v); }

// Kernel declaration macros. Slot numbers match the RHI binding convention
// (params at buffer(0), resource i at buffer(i+1)) and are explicit at the
// declaration so host dispatch lists can be checked against them by eye.
#define GPU_KERNEL(name) kernel void name
#define GPU_KERNEL_PARAMS(T, name) constant T& name [[buffer(0)]]
#define GPU_BUFFER(T, name, slot) , device T* name [[buffer(slot)]]
#define GPU_TID_1D , uint gpu_tid_ [[thread_position_in_grid]]
#define GPU_TID_2D , uint2 gpu_tid2_ [[thread_position_in_grid]]
#define GPU_TID_FULL                                                        \
    , uint gpu_tid_ [[thread_position_in_grid]]                             \
    , uint gpu_lid_ [[thread_index_in_threadgroup]]                         \
    , uint gpu_sg_ [[simdgroup_index_in_threadgroup]]                       \
    , uint gpu_nsg_ [[simdgroups_per_threadgroup]]                          \
    , uint gpu_group_ [[threadgroup_position_in_grid]]
#define GPU_GLOBAL_ID_X gpu_tid_
#define GPU_GLOBAL_ID_XY gpu_tid2_
#define GPU_LOCAL_ID gpu_lid_
#define GPU_WAVE_INDEX gpu_sg_
#define GPU_NUM_WAVES gpu_nsg_
#define GPU_GROUP_ID gpu_group_

// Pipeline specialization (rhi::SpecConstant). Specialized kernels and their
// GPU_SPEC_CONST declarations sit inside `#if GPU_HAS_SPEC_CONST` so backends
// without a lowering yet still compile every other kernel in the file; the
// missing kernel then fails loudly at pipeline creation, not file parse.
#define GPU_HAS_SPEC_CONST 1
#define GPU_SPEC_CONST(T, name, idx) constant T name [[function_constant(idx)]];

#else  // ------------------------------------------------------------------
// CUDA and host personalities share the glm value types; they differ in
// function qualifiers, intrinsics, and the wave/kernel machinery (CUDA-only).
// ---------------------------------------------------------------------------
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>

typedef unsigned int uint;    // duplicate-of-identical is legal where OS headers define it
typedef unsigned char uchar;

typedef glm::vec2 gpu_float2;
typedef glm::vec3 gpu_float3;
typedef glm::vec4 gpu_float4;
typedef glm::uvec2 gpu_uint2;
typedef glm::uvec4 gpu_uint4;
typedef glm::vec3 gpu_packed3;       // 12 bytes, matches MSL packed_float3
typedef unsigned int gpu_atomic_uint;

#define GPU_PARAMS_REF(T) const T&

#if defined(__CUDACC__)
// ---- CUDA personality (nvcc; kernels registered by rhi_cuda in M4) ----
#define GPU_FN __device__
struct alignas(16) gpu_storage3 { float x, y, z; };  // MSL float3 layout
typedef glm::mat4 gpu_storage4x4;    // column-major 64B, matches MSL float4x4
GPU_FN inline gpu_float3 gpu_load3(gpu_storage3 v) { return gpu_float3(v.x, v.y, v.z); }

typedef ::uchar4 gpu_uchar4;
GPU_FN inline gpu_uchar4 gpu_make_uchar4(uchar r, uchar g, uchar b, uchar a)
{
    return make_uchar4(r, g, b, a);
}

GPU_FN inline uint gpu_atomic_load(gpu_atomic_uint* p) { return *(volatile uint*)p; }
GPU_FN inline void gpu_atomic_store(gpu_atomic_uint* p, uint v) { *(volatile uint*)p = v; }
GPU_FN inline uint gpu_atomic_fetch_add(gpu_atomic_uint* p, uint v) { return atomicAdd(p, v); }

// Wave ops over the active lanes via cooperative groups (M4-verified). The
// gpu_wave_t handle captures the coalesced group once so every op — critically
// the post-branch broadcast in prim_queue_alloc — talks to the SAME lane set.
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#define gpu_barrier() __syncthreads()
struct gpu_wave_t {
    cooperative_groups::coalesced_group g;
};
GPU_FN inline gpu_wave_t gpu_wave_active() { return { cooperative_groups::coalesced_threads() }; }
GPU_FN inline uint gpu_wave_prefix_sum(gpu_wave_t w, uint v)
{
    return cooperative_groups::exclusive_scan(w.g, v);
}
GPU_FN inline uint gpu_wave_sum(gpu_wave_t w, uint v)
{
    return cooperative_groups::reduce(w.g, v, cooperative_groups::plus<uint>());
}
GPU_FN inline bool gpu_wave_is_first(gpu_wave_t w) { return w.g.thread_rank() == 0; }
GPU_FN inline uint gpu_wave_broadcast_first(gpu_wave_t w, uint v) { return w.g.shfl(v, 0); }

#define GPU_SHARED __shared__
#define GPU_SHARED_SPACE
#define GPU_KERNEL(name) __global__ void name
#define GPU_KERNEL_PARAMS(T, name) const T name
#define GPU_BUFFER(T, name, slot) , T* name
#define GPU_TID_1D
#define GPU_TID_2D
#define GPU_TID_FULL
#define GPU_GLOBAL_ID_X (blockIdx.x * blockDim.x + threadIdx.x)
#define GPU_GLOBAL_ID_XY \
    gpu_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)
#define GPU_LOCAL_ID threadIdx.x
#define GPU_WAVE_INDEX (threadIdx.x / 32u)
#define GPU_NUM_WAVES (blockDim.x / 32u)
#define GPU_GROUP_ID blockIdx.x
// No specialization lowering yet (GPU_HAS_SPEC_CONST 0 below): rhi_cuda (M4)
// picks one — template parameters or per-value -D compilation. Until then
// specialized kernels are compiled out by their #if guard and fail loudly at
// pipeline creation, so every other kernel in a file still builds.

#else
// ---- host personality (plain C++: SharedHostTest, CPU debugging) ----
#define GPU_FN
#if defined(__APPLE__)
// Storage aliases keep the types mac host code already uses (simd math ops on
// MiniObject transforms, 16-byte upload layouts) — this personality is both
// the value-parity test and the working host of FluoraMini.
#include <simd/simd.h>
typedef simd_float3 gpu_storage3;
typedef simd_float4x4 gpu_storage4x4;
GPU_FN inline gpu_float3 gpu_load3(gpu_storage3 v) { return gpu_float3(v.x, v.y, v.z); }
#else
struct alignas(16) gpu_storage3 { float x, y, z; };
typedef glm::mat4 gpu_storage4x4;
GPU_FN inline gpu_float3 gpu_load3(gpu_storage3 v) { return gpu_float3(v.x, v.y, v.z); }
#endif

struct gpu_uchar4 { uchar x, y, z, w; };
GPU_FN inline gpu_uchar4 gpu_make_uchar4(uchar r, uchar g, uchar b, uchar a)
{
    return gpu_uchar4{ r, g, b, a };
}

// Scalar intrinsics MSL/CUDA provide but plain C++ does not.
GPU_FN inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
GPU_FN inline float max(float a, float b) { return a > b ? a : b; }
GPU_FN inline float min(float a, float b) { return a < b ? a : b; }
#endif  // __CUDACC__ / host

#define GPU_DEVICE
#define GPU_THREAD
#define GPU_HAS_SPEC_CONST 0

// glm gaps shared by CUDA + host.
GPU_FN inline float length_squared(gpu_float3 v) { return glm::dot(v, v); }
GPU_FN inline gpu_float3 gpu_xyz(gpu_float4 v) { return gpu_float3(v.x, v.y, v.z); }
GPU_FN inline bool gpu_all_finite(gpu_float3 v)
{
#if defined(__CUDACC__)
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
#else
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
#endif
}

#endif  // personalities

#define GPU_PI 3.14159265358979323846f

// ---------------------------------------------------------------------------
// Shared RNG: PCG output permutation on an LCG state (identical draw sequence
// on every backend — the golden images depend on it). Seed via .state.
// ---------------------------------------------------------------------------
struct GpuRng {
    uint state;
};

GPU_FN inline float gpu_rand(GPU_THREAD GpuRng& r)
{
    r.state = r.state * 747796405u + 2891336453u;
    uint w = ((r.state >> ((r.state >> 28u) + 4u)) ^ r.state) * 277803737u;
    w = (w >> 22u) ^ w;
    return (float)w * (1.0f / 4294967296.0f);
}

#endif  // RHI_GPU_PORTABLE_H
