#ifndef RHI_RAYTRACE_GPU_H
#define RHI_RAYTRACE_GPU_H
// Device-side ray tracing seam (design doc §5): renderer kernels call
// rt_closest_hit() and stay ignorant of the traversal technique. This
// implementation walks the six-direction threaded BVH built by
// src/core/bvh_builder.cpp (mini analog of the CUDA MTBVH in
// intersections.cu); a Metal hardware-RT implementation with the same
// signature is the M5 fast path.
//
// Single-source via the gpu_portable shim (docs/portable-device-code.md);
// compiled after gpu_portable.h and core/accel_shared.h (concatenated under
// MSL, #included elsewhere).

#ifndef __METAL_VERSION__
#include "gpu_portable.h"
#include "../core/accel_shared.h"
#endif

struct RtHit {
    float t;           // in: max distance; out: hit distance
    gpu_float3 n;      // geometric normal (unoriented)
    gpu_float2 bary;   // barycentrics (weights of v1, v2)
    uint triIdx;       // index into the (reordered) triangle array
    uint userData;     // triangle uint4.w (renderer stores a material id)
    bool hit;
};

GPU_FN inline bool rt_intersect_triangle(gpu_float3 ro, gpu_float3 rd,
                                  gpu_float3 v0, gpu_float3 v1, gpu_float3 v2,
                                  GPU_THREAD float& tOut, GPU_THREAD gpu_float2& baryOut)
{
    // Moller-Trumbore, no backface culling (dielectrics need both sides)
    gpu_float3 e1 = v1 - v0;
    gpu_float3 e2 = v2 - v0;
    gpu_float3 pv = cross(rd, e2);
    float det = dot(e1, pv);
    if (fabs(det) < 1e-9f)
        return false;
    float invDet = 1.0f / det;
    gpu_float3 tv = ro - v0;
    float u = dot(tv, pv) * invDet;
    if (u < 0.0f || u > 1.0f)
        return false;
    gpu_float3 qv = cross(tv, e1);
    float v = dot(rd, qv) * invDet;
    if (v < 0.0f || u + v > 1.0f)
        return false;
    float t = dot(e2, qv) * invDet;
    if (t < 1e-5f)
        return false;
    tOut = t;
    baryOut = gpu_float2(u, v);
    return true;
}

GPU_FN inline bool rt_aabb_hit(gpu_float3 ro, gpu_float3 invD,
                        gpu_float3 bmin, gpu_float3 bmax, float tMax)
{
    gpu_float3 t0 = (bmin - ro) * invD;
    gpu_float3 t1 = (bmax - ro) * invD;
    gpu_float3 tS = min(t0, t1);
    gpu_float3 tB = max(t0, t1);
    float tmin = max(max(tS.x, tS.y), tS.z);
    float tmax = min(min(tB.x, tB.y), tB.z);
    return tmax >= max(tmin, 0.0f) && tmin < tMax;
}

// Stackless traversal: pick the direction-ordered node array by the ray's
// dominant axis/sign, then follow hit/miss links — no stack, front-to-back-ish.
// hit.t must be initialized to the current closest distance (or INFINITY).
GPU_FN inline void rt_closest_hit(gpu_float3 ro, gpu_float3 rd,
                           GPU_DEVICE const RtBvhNode* nodes, uint numNodes,
                           GPU_DEVICE const gpu_uint4* tris,
                           GPU_DEVICE const gpu_storage3* positions,
                           GPU_THREAD RtHit& hit)
{
    hit.hit = false;
    if (numNodes == 0)
        return;
    gpu_float3 a = abs(rd);
    uint axis = a.x > a.y ? (a.x > a.z ? 0u : 2u) : (a.y > a.z ? 1u : 2u);
    uint dirIdx = axis * 2u + (rd[axis] < 0.0f ? 1u : 0u);
    GPU_DEVICE const RtBvhNode* base = nodes + dirIdx * numNodes;
    gpu_float3 invD = 1.0f / rd;
    int bestTri = -1;
    gpu_float2 bestBary = gpu_float2(0.0f);
    uint curr = 0;
    while (curr < numNodes) {
        GPU_DEVICE const RtBvhNode& node = base[curr];
        if (rt_aabb_hit(ro, invD, gpu_load3(node.bmin), gpu_load3(node.bmax), hit.t)) {
            for (uint i = 0; i < node.triCount; i++) {
                gpu_uint4 tri = tris[node.triStart + i];
                float t;
                gpu_float2 bary;
                if (rt_intersect_triangle(ro, rd, gpu_load3(positions[tri.x]),
                                          gpu_load3(positions[tri.y]),
                                          gpu_load3(positions[tri.z]), t, bary)
                    && t < hit.t) {
                    hit.t = t;
                    bestTri = (int)(node.triStart + i);
                    bestBary = bary;
                }
            }
            curr = node.hitLink;
        } else {
            curr = node.missLink;
        }
    }
    if (bestTri >= 0) {
        gpu_uint4 tri = tris[bestTri];
        hit.n = normalize(cross(gpu_load3(positions[tri.y]) - gpu_load3(positions[tri.x]),
                                gpu_load3(positions[tri.z]) - gpu_load3(positions[tri.x])));
        hit.bary = bestBary;
        hit.triIdx = (uint)bestTri;
        hit.userData = tri.w;
        hit.hit = true;
    }
}

// Vertex-attribute interpolation for a hit; the attribute arrays share the
// triangle's vertex indices (unified vertices, core/mesh_loader).

// Zero-length vertex normals mean "not provided" — fall back to geometric.
GPU_FN inline gpu_float3 rt_shading_normal(GPU_DEVICE const gpu_uint4* tris,
                                    GPU_DEVICE const gpu_storage3* normals,
                                    GPU_THREAD const RtHit& hit)
{
    gpu_uint4 tri = tris[hit.triIdx];
    gpu_float3 n = gpu_load3(normals[tri.x]) * (1.0f - hit.bary.x - hit.bary.y)
                 + gpu_load3(normals[tri.y]) * hit.bary.x
                 + gpu_load3(normals[tri.z]) * hit.bary.y;
    return dot(n, n) > 1e-12f ? normalize(n) : hit.n;
}

GPU_FN inline gpu_float2 rt_interp_uv(GPU_DEVICE const gpu_uint4* tris,
                               GPU_DEVICE const gpu_float2* uvs,
                               GPU_THREAD const RtHit& hit)
{
    gpu_uint4 tri = tris[hit.triIdx];
    return uvs[tri.x] * (1.0f - hit.bary.x - hit.bary.y)
         + uvs[tri.y] * hit.bary.x
         + uvs[tri.z] * hit.bary.y;
}

#endif // RHI_RAYTRACE_GPU_H
