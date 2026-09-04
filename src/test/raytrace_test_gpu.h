#ifndef TEST_RAYTRACE_TEST_GPU_H
#define TEST_RAYTRACE_TEST_GPU_H
// RaytraceTest's GPU kernel, single-source (concatenated for MSL after
// raytrace_gpu.h, #included by the generated CUDA registration TU): traces
// the test rays through the uploaded BVH via the same rt_* seam the renderer
// kernels use, so the GPU personality compares 1:1 against the host-C++
// traversal in raytrace_test.cpp.
//
// Ray layout: 2 float4 per ray — (origin.xyz, occlusion tMax), (dir.xyz, 0).
// Out layout: RT_TEST_OUT_STRIDE float4 per ray:
//   [0] (t, triIdx or -1, bary.u, bary.v)   t stays INFINITY on miss
//   [1] (hit ? 1 : 0, occluded-within-tMax ? 1 : 0, userData, 0)
//   [2] shading normal.xyz, 0               zero on miss
//   [3] interpolated uv.xy, 0, 0            zero on miss

#ifndef __METAL_VERSION__
#include "../rhi/gpu_portable.h"
#include "../rhi/primitives_shared.h"
#include "../core/accel_shared.h"
#include "../rhi/raytrace_gpu.h"
#endif

#define RT_TEST_OUT_STRIDE 4u

struct RtTestParams {
    unsigned int n;               // ray count
    unsigned int numNodesPerDir;  // stride of the six direction-ordered node tables
};

#if defined(__METAL_VERSION__) || defined(__CUDACC__)
GPU_KERNEL(rt_test_trace, GPU_TID_1D)(GPU_KERNEL_PARAMS(RtTestParams, P),
    GPU_BUFFER(const gpu_float4, rays),
    GPU_BUFFER(const RtBvhNode, nodes),
    GPU_BUFFER(const gpu_uint4, tris),
    GPU_BUFFER(const gpu_storage3, positions),
    GPU_BUFFER(const gpu_storage3, normals),
    GPU_BUFFER(const gpu_float2, uvs),
    GPU_BUFFER(gpu_float4, outv))
{
    uint tid = GPU_GLOBAL_ID_X;
    if (tid >= P.n)
        return;
    gpu_float3 ro = gpu_xyz(rays[tid * 2u]);
    float tMax = rays[tid * 2u].w;
    gpu_float3 rd = gpu_xyz(rays[tid * 2u + 1u]);
    RtHit hit;
    hit.t = INFINITY;
    rt_closest_hit(ro, rd, nodes, P.numNodesPerDir, tris, positions, hit);
    bool occ = rt_occluded(ro, rd, nodes, P.numNodesPerDir, tris, positions, tMax);
    gpu_float3 sn = hit.hit ? rt_shading_normal(tris, normals, hit) : gpu_float3(0.0f);
    gpu_float2 uv = hit.hit ? rt_interp_uv(tris, uvs, hit) : gpu_float2(0.0f);
    uint o = tid * RT_TEST_OUT_STRIDE;
    gpu_float2 bary = hit.hit ? hit.bary : gpu_float2(0.0f);  // bary is undefined on miss
    outv[o] = gpu_float4(hit.t, hit.hit ? (float)hit.triIdx : -1.0f,
                         bary.x, bary.y);
    outv[o + 1u] = gpu_float4(hit.hit ? 1.0f : 0.0f, occ ? 1.0f : 0.0f,
                              hit.hit ? (float)hit.userData : 0.0f, 0.0f);
    outv[o + 2u] = gpu_float4(sn, 0.0f);
    outv[o + 3u] = gpu_float4(uv.x, uv.y, 0.0f, 0.0f);
}
#endif

#endif  // TEST_RAYTRACE_TEST_GPU_H
