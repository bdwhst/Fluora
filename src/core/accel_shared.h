#ifndef CORE_ACCEL_SHARED_H
#define CORE_ACCEL_SHARED_H
// Acceleration-structure types shared between portable host code (src/core),
// the device traversal (src/rhi/raytrace_gpu.h), and renderer kernels.
// Single-source via the gpu_portable shim; gpu_storage3 keeps the 16-byte
// float3 layout identical across MSL, hosts, and CUDA (invariant I-3).

#ifndef __METAL_VERSION__
#include "../rhi/gpu_portable.h"
#endif

// Stackless threaded BVH over world-space triangles. Six DFS orderings (per
// dominant ray axis/sign, near child first) share one tree and one reordered
// triangle array; node arrays are concatenated per direction with stride
// numNodes. A link value == numNodes terminates traversal.
//
// Triangles are uint4 {i0, i1, i2, userData} into a float3 position array;
// userData (the renderer stores a material id there) is passed through to hits.
struct RtBvhNode {
    gpu_storage3 bmin;
    gpu_storage3 bmax;
    unsigned int hitLink;
    unsigned int missLink;
    unsigned int triStart;   // range into the reordered triangle array
    unsigned int triCount;   // 0 for interior nodes
};

#endif
