#ifndef CORE_ACCEL_SHARED_H
#define CORE_ACCEL_SHARED_H
// Acceleration-structure types shared between portable host code (src/core),
// the device traversal (src/rhi/raytrace.metal), and renderer kernels.
// Compiled by host clang (simd types) and the Metal compiler (invariant I-3);
// self-contained because MSL sources are concatenated, not #included.

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef metal::float3 rt_float3;
#else
#include <simd/simd.h>
typedef simd_float3 rt_float3;
#endif

// Stackless threaded BVH over world-space triangles. Six DFS orderings (per
// dominant ray axis/sign, near child first) share one tree and one reordered
// triangle array; node arrays are concatenated per direction with stride
// numNodes. A link value == numNodes terminates traversal.
//
// Triangles are uint4 {i0, i1, i2, userData} into a float3 position array;
// userData (the renderer stores a material id there) is passed through to hits.
struct RtBvhNode {
    rt_float3 bmin;
    rt_float3 bmax;
    unsigned int hitLink;
    unsigned int missLink;
    unsigned int triStart;   // range into the reordered triangle array
    unsigned int triCount;   // 0 for interior nodes
};

#endif
