#ifndef RHI_GPU_PORTABLE_H
#define RHI_GPU_PORTABLE_H
// =============================================================================
// The portable device-code shim (docs/portable-device-code.md §3).
//
// Shaders in this codebase are written ONCE and compiled by three compilers:
//   MSL      (__METAL_VERSION__)  — runtime-concatenated, this header first;
//                                   #include lines below are inert under MSL.
//   CUDA     (__CUDACC__)         — regular #include; kernels registered per
//                                   rhi_cuda.h (M4: RhiTest/SharedHostTest and
//                                   every FluoraMini scene verified on CUDA).
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
GPU_FN inline bool gpu_isinf(float x) { return isinf(x); }

// Scalar f-suffix intrinsics MSL spells without the suffix; shared code uses
// the suffixed names so host/CUDA get the float-precision versions.
GPU_FN inline float atanhf(float x) { return atanh(x); }
GPU_FN inline float coshf(float x) { return cosh(x); }
GPU_FN inline float copysignf(float x, float y) { return copysign(x, y); }
GPU_FN inline float fmaf(float a, float b, float c) { return fma(a, b, c); }

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

// Kernel declaration macros. GPU_KERNEL names the kernel and the thread ids it
// reads (GPU_TID_1D / GPU_TID_2D / GPU_TID_FULL / GPU_TID_NONE); the parameter
// list that follows is an ordinary one — the parameter block, then one
// GPU_BUFFER per resource, commas written by the author, buffer slots implied
// by position:
//   GPU_KERNEL(k, GPU_TID_1D)(GPU_KERNEL_PARAMS(P, p),
//       GPU_BUFFER(const T, in),     // buffer(1)
//       GPU_BUFFER(T, out))          // buffer(2)
// GPU_KERNEL expands to `kernel void k GPU_SIG_GPU_TID_1D`, a function-like
// macro name that the preprocessor's rescan applies to the parenthesized list
// in the source (standard C/C++ rescanning, the BOOST_PP idiom). Under MSL
// that wrapper prepends the [[attribute]] id parameters and numbers the
// buffers; under CUDA it is the identity (ids come from blockIdx/threadIdx,
// buffers bind positionally), so both backends see the same parameter list.
// Positional buffer slots. GPU_SIG_* receives the whole parameter list; the
// GPU_PP_MAP machinery below numbers it: argument 0 (the parameter block)
// becomes buffer(0), argument i becomes buffer(i) -- the same order the host
// binds resources in (rhi.h: resource i at buffer(i+1)), so slot numbers
// cannot drift from the dispatch list. Standard count-and-map preprocessor
// idiom; at most GPU_PP_MAX_ARGS parameters per kernel (a longer list fails
// to compile instead of misbinding).
#define GPU_PP_MAX_ARGS 32
#define GPU_PP_CAT(a, b) GPU_PP_CAT_(a, b)
#define GPU_PP_CAT_(a, b) a##b
#define GPU_PP_NARGS(...) GPU_PP_NARGS_(__VA_ARGS__, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define GPU_PP_NARGS_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, N, ...) N
#define GPU_SLOT0(x) constant x [[buffer(0)]]
#define GPU_SLOTN(i, x) device x [[buffer(i)]]
#define GPU_PP_MAP(...) GPU_PP_CAT(GPU_PP_MAP_, GPU_PP_NARGS(__VA_ARGS__))(__VA_ARGS__)
#define GPU_PP_MAP_1(a0) GPU_SLOT0(a0)
#define GPU_PP_MAP_2(a0, a1) GPU_SLOT0(a0), GPU_SLOTN(1, a1)
#define GPU_PP_MAP_3(a0, a1, a2) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2)
#define GPU_PP_MAP_4(a0, a1, a2, a3) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3)
#define GPU_PP_MAP_5(a0, a1, a2, a3, a4) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4)
#define GPU_PP_MAP_6(a0, a1, a2, a3, a4, a5) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5)
#define GPU_PP_MAP_7(a0, a1, a2, a3, a4, a5, a6) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6)
#define GPU_PP_MAP_8(a0, a1, a2, a3, a4, a5, a6, a7) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7)
#define GPU_PP_MAP_9(a0, a1, a2, a3, a4, a5, a6, a7, a8) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8)
#define GPU_PP_MAP_10(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9)
#define GPU_PP_MAP_11(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10)
#define GPU_PP_MAP_12(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11)
#define GPU_PP_MAP_13(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12)
#define GPU_PP_MAP_14(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13)
#define GPU_PP_MAP_15(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14)
#define GPU_PP_MAP_16(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15)
#define GPU_PP_MAP_17(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16)
#define GPU_PP_MAP_18(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17)
#define GPU_PP_MAP_19(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18)
#define GPU_PP_MAP_20(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18), GPU_SLOTN(19, a19)
#define GPU_PP_MAP_21(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18), GPU_SLOTN(19, a19), GPU_SLOTN(20, a20)
#define GPU_PP_MAP_22(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18), GPU_SLOTN(19, a19), GPU_SLOTN(20, a20), GPU_SLOTN(21, a21)
#define GPU_PP_MAP_23(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18), GPU_SLOTN(19, a19), GPU_SLOTN(20, a20), GPU_SLOTN(21, a21), GPU_SLOTN(22, a22)
#define GPU_PP_MAP_24(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18), GPU_SLOTN(19, a19), GPU_SLOTN(20, a20), GPU_SLOTN(21, a21), GPU_SLOTN(22, a22), GPU_SLOTN(23, a23)
#define GPU_PP_MAP_25(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18), GPU_SLOTN(19, a19), GPU_SLOTN(20, a20), GPU_SLOTN(21, a21), GPU_SLOTN(22, a22), GPU_SLOTN(23, a23), GPU_SLOTN(24, a24)
#define GPU_PP_MAP_26(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18), GPU_SLOTN(19, a19), GPU_SLOTN(20, a20), GPU_SLOTN(21, a21), GPU_SLOTN(22, a22), GPU_SLOTN(23, a23), GPU_SLOTN(24, a24), GPU_SLOTN(25, a25)
#define GPU_PP_MAP_27(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18), GPU_SLOTN(19, a19), GPU_SLOTN(20, a20), GPU_SLOTN(21, a21), GPU_SLOTN(22, a22), GPU_SLOTN(23, a23), GPU_SLOTN(24, a24), GPU_SLOTN(25, a25), GPU_SLOTN(26, a26)
#define GPU_PP_MAP_28(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18), GPU_SLOTN(19, a19), GPU_SLOTN(20, a20), GPU_SLOTN(21, a21), GPU_SLOTN(22, a22), GPU_SLOTN(23, a23), GPU_SLOTN(24, a24), GPU_SLOTN(25, a25), GPU_SLOTN(26, a26), GPU_SLOTN(27, a27)
#define GPU_PP_MAP_29(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18), GPU_SLOTN(19, a19), GPU_SLOTN(20, a20), GPU_SLOTN(21, a21), GPU_SLOTN(22, a22), GPU_SLOTN(23, a23), GPU_SLOTN(24, a24), GPU_SLOTN(25, a25), GPU_SLOTN(26, a26), GPU_SLOTN(27, a27), GPU_SLOTN(28, a28)
#define GPU_PP_MAP_30(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18), GPU_SLOTN(19, a19), GPU_SLOTN(20, a20), GPU_SLOTN(21, a21), GPU_SLOTN(22, a22), GPU_SLOTN(23, a23), GPU_SLOTN(24, a24), GPU_SLOTN(25, a25), GPU_SLOTN(26, a26), GPU_SLOTN(27, a27), GPU_SLOTN(28, a28), GPU_SLOTN(29, a29)
#define GPU_PP_MAP_31(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18), GPU_SLOTN(19, a19), GPU_SLOTN(20, a20), GPU_SLOTN(21, a21), GPU_SLOTN(22, a22), GPU_SLOTN(23, a23), GPU_SLOTN(24, a24), GPU_SLOTN(25, a25), GPU_SLOTN(26, a26), GPU_SLOTN(27, a27), GPU_SLOTN(28, a28), GPU_SLOTN(29, a29), GPU_SLOTN(30, a30)
#define GPU_PP_MAP_32(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31) GPU_SLOT0(a0), GPU_SLOTN(1, a1), GPU_SLOTN(2, a2), GPU_SLOTN(3, a3), GPU_SLOTN(4, a4), GPU_SLOTN(5, a5), GPU_SLOTN(6, a6), GPU_SLOTN(7, a7), GPU_SLOTN(8, a8), GPU_SLOTN(9, a9), GPU_SLOTN(10, a10), GPU_SLOTN(11, a11), GPU_SLOTN(12, a12), GPU_SLOTN(13, a13), GPU_SLOTN(14, a14), GPU_SLOTN(15, a15), GPU_SLOTN(16, a16), GPU_SLOTN(17, a17), GPU_SLOTN(18, a18), GPU_SLOTN(19, a19), GPU_SLOTN(20, a20), GPU_SLOTN(21, a21), GPU_SLOTN(22, a22), GPU_SLOTN(23, a23), GPU_SLOTN(24, a24), GPU_SLOTN(25, a25), GPU_SLOTN(26, a26), GPU_SLOTN(27, a27), GPU_SLOTN(28, a28), GPU_SLOTN(29, a29), GPU_SLOTN(30, a30), GPU_SLOTN(31, a31)
#define GPU_KERNEL(name, tid) kernel void name GPU_SIG_##tid
#define GPU_SIG_GPU_TID_NONE(...) (GPU_PP_MAP(__VA_ARGS__))
#define GPU_SIG_GPU_TID_1D(...) (uint gpu_tid_ [[thread_position_in_grid]], GPU_PP_MAP(__VA_ARGS__))
#define GPU_SIG_GPU_TID_2D(...) (uint2 gpu_tid2_ [[thread_position_in_grid]], GPU_PP_MAP(__VA_ARGS__))
#define GPU_SIG_GPU_TID_FULL(...)                                           \
    (uint gpu_tid_ [[thread_position_in_grid]],                             \
     uint gpu_lid_ [[thread_index_in_threadgroup]],                         \
     uint gpu_sg_ [[simdgroup_index_in_threadgroup]],                       \
     uint gpu_nsg_ [[simdgroups_per_threadgroup]],                          \
     uint gpu_group_ [[threadgroup_position_in_grid]], GPU_PP_MAP(__VA_ARGS__))
// Bare declarators; GPU_PP_MAP adds the address space and [[buffer(i)]].
#define GPU_KERNEL_PARAMS(T, name) T& name
#define GPU_BUFFER(T, name) T* name
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
// Annotation (no code on any backend): the specialization values a kernel is
// built with, e.g. GPU_SPEC_INSTANCES(wf_shade, 0, MINI_MAT_DIFFUSE, ...).
// cmake/GenerateCudaKernels.cmake reads it to emit the CUDA registrations;
// Metal specializes at pipeline creation and ignores it.
#define GPU_SPEC_INSTANCES(...)

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
// ---- CUDA personality (nvcc; kernels registered through rhi_cuda.h) ----
#define GPU_FN __device__
struct alignas(16) gpu_storage3 { float x, y, z; };  // MSL float3 layout
typedef glm::mat4 gpu_storage4x4;    // column-major 64B, matches MSL float4x4
// One 16-byte vector load (alignas(16) makes the reinterpret valid) instead
// of three scalar loads — the BVH inner loop reads bmin/bmax/positions this way.
GPU_FN inline gpu_float3 gpu_load3(const gpu_storage3& v)
{
    float4 v4 = *reinterpret_cast<const float4*>(&v);
    return gpu_float3(v4.x, v4.y, v4.z);
}

typedef ::uchar4 gpu_uchar4;
GPU_FN inline gpu_uchar4 gpu_make_uchar4(uchar r, uchar g, uchar b, uchar a)
{
    return make_uchar4(r, g, b, a);
}

// Relaxed device-scope atomics. A naturally aligned 32-bit volatile access is
// a single untorn load/store on NVIDIA GPUs, which is exactly the relaxed
// atomic load/store the Metal shim provides (no ordering, just atomicity), and
// unlike atomicAdd(p, 0) it does not serialize the million threads that read a
// queue count at the top of every wavefront kernel.
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
// Kernels keep external linkage (nvcc fails to register template kernels
// whose template arguments name internal-linkage functions, which the
// indirect-dispatch launcher in rhi_cuda.h relies on), so each _gpu.h file's
// kernels must be compiled in exactly one .cu TU — the one that registers
// them with RHI_CUDA_REGISTER_* (rhi_cuda.h). primitives_gpu.h, whose helpers
// renderer kernels also need, guards its kernels with
// GPU_PRIMITIVES_HELPERS_ONLY for that reason.
// Thread ids come from blockIdx/threadIdx (GPU_GLOBAL_ID_X etc.), so every
// GPU_SIG_* wrapper is the identity: the parameter list is used as written.
#define GPU_KERNEL(name, tid) __global__ void name GPU_SIG_##tid
#define GPU_SIG_GPU_TID_NONE(...) (__VA_ARGS__)
#define GPU_SIG_GPU_TID_1D(...) (__VA_ARGS__)
#define GPU_SIG_GPU_TID_2D(...) (__VA_ARGS__)
#define GPU_SIG_GPU_TID_FULL(...) (__VA_ARGS__)
#define GPU_KERNEL_PARAMS(T, name) const T name
#define GPU_BUFFER(T, name) T* name
#define GPU_GLOBAL_ID_X (blockIdx.x * blockDim.x + threadIdx.x)
#define GPU_GLOBAL_ID_XY \
    gpu_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y)
#define GPU_LOCAL_ID threadIdx.x
#define GPU_WAVE_INDEX (threadIdx.x / 32u)
#define GPU_NUM_WAVES (blockDim.x / 32u)
#define GPU_GROUP_ID blockIdx.x
// Pipeline specialization lowers to a template parameter: GPU_SPEC_CONST must
// immediately precede its GPU_KERNEL, which becomes `template <T name>
// static __global__ void kernel(...)`. One constant per kernel (the Metal
// analog allows several; extend to a pack when a kernel needs it). Host side,
// RHI_CUDA_REGISTER_SPEC(kernel, index, value) registers `kernel<value>` under
// the same {entryPoint, constants} key rhi::ComputePipelineDesc carries.
#define GPU_HAS_SPEC_CONST 1
#define GPU_SPEC_CONST(T, name, idx) template <T name>
#define GPU_SPEC_INSTANCES(...)   // annotation read by cmake/GenerateCudaKernels.cmake

// NaN-consistent vector min/max: MSL's are componentwise fmin/fmax
// (NaN-suppressing); glm's are `a < b ? a : b` (NaN-propagating), a parity
// break in the slab tests when a ray origin lies on a node plane with a zero
// direction component (0*inf). These non-template overloads beat glm's
// templates for exact gpu_* argument types.
GPU_FN inline gpu_float2 min(gpu_float2 a, gpu_float2 b) { return gpu_float2(fminf(a.x, b.x), fminf(a.y, b.y)); }
GPU_FN inline gpu_float2 max(gpu_float2 a, gpu_float2 b) { return gpu_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y)); }
GPU_FN inline gpu_float3 min(gpu_float3 a, gpu_float3 b) { return gpu_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
GPU_FN inline gpu_float3 max(gpu_float3 a, gpu_float3 b) { return gpu_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
GPU_FN inline gpu_float4 min(gpu_float4 a, gpu_float4 b) { return gpu_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w)); }
GPU_FN inline gpu_float4 max(gpu_float4 a, gpu_float4 b) { return gpu_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w)); }

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

// Scalar intrinsics MSL/CUDA provide but plain C++ does not. min/max are
// fmin/fmax so NaN handling matches MSL and CUDA's device overloads.
GPU_FN inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
GPU_FN inline float max(float a, float b) { return std::fmax(a, b); }
GPU_FN inline float min(float a, float b) { return std::fmin(a, b); }
GPU_FN inline gpu_float2 min(gpu_float2 a, gpu_float2 b) { return gpu_float2(std::fmin(a.x, b.x), std::fmin(a.y, b.y)); }
GPU_FN inline gpu_float2 max(gpu_float2 a, gpu_float2 b) { return gpu_float2(std::fmax(a.x, b.x), std::fmax(a.y, b.y)); }
GPU_FN inline gpu_float3 min(gpu_float3 a, gpu_float3 b) { return gpu_float3(std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z)); }
GPU_FN inline gpu_float3 max(gpu_float3 a, gpu_float3 b) { return gpu_float3(std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z)); }
GPU_FN inline gpu_float4 min(gpu_float4 a, gpu_float4 b) { return gpu_float4(std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z), std::fmin(a.w, b.w)); }
GPU_FN inline gpu_float4 max(gpu_float4 a, gpu_float4 b) { return gpu_float4(std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z), std::fmax(a.w, b.w)); }
// No kernel/spec-constant lowering on hosts (kernels are not compiled here).
#define GPU_HAS_SPEC_CONST 0
#define GPU_SPEC_INSTANCES(...)
#endif  // __CUDACC__ / host

#define GPU_DEVICE
#define GPU_THREAD

// glm gaps shared by CUDA + host.
GPU_FN inline float length_squared(gpu_float3 v) { return glm::dot(v, v); }
GPU_FN inline gpu_float3 gpu_xyz(gpu_float4 v) { return gpu_float3(v.x, v.y, v.z); }
GPU_FN inline bool gpu_isinf(float x)
{
#if defined(__CUDACC__)
    return isinf(x);
#else
    return std::isinf(x);
#endif
}
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
