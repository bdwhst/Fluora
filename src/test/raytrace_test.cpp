// Unit + equivalence tests for the intersection/BVH seam (the M4 review's
// coverage gap): rt_intersect_triangle / rt_aabb_hit against hand-computed
// cases, then host-personality rt_closest_hit / rt_occluded over a real
// bvh_builder tree vs brute force across all triangles, then the identical
// rays traced on the GPU (rt_test_trace) vs the host traversal.
//
// Host-vs-brute comparisons are EXACT (both paths run the same host float
// code on the same inputs, so the traversal may only differ by missing or
// inventing a hit). Host-vs-GPU compares hit/miss, triangle identity,
// userData and occlusion exactly, and t/bary/normal/uv with a tolerance
// (Metal compiles with fast math). Exits nonzero on failure.
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "../rhi/gpu_portable.h"
#include "../rhi/primitives_shared.h"
#include "../core/accel_shared.h"
#include "../rhi/raytrace_gpu.h"
#include "raytrace_test_gpu.h"

#include "../core/bvh_builder.h"
#include "../rhi/rhi.h"

namespace {

std::string readTextFile(const std::string& path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int failures = 0;

void check(bool ok, const char* name)
{
    std::cout << (ok ? "PASS " : "FAIL ") << name << "\n";
    if (!ok)
        failures++;
}

gpu_storage3 store3(float x, float y, float z)
{
    gpu_storage3 s;
    s.x = x; s.y = y; s.z = z;
    return s;
}

bool near1(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

// Brute-force reference: closest hit over every triangle with the same
// rt_intersect_triangle the traversal uses.
struct BruteHit {
    bool hit = false;
    float t = INFINITY;
    uint32_t triIdx = 0;
    gpu_float2 bary = gpu_float2(0.0f);
};

BruteHit bruteClosest(gpu_float3 ro, gpu_float3 rd,
                      const std::vector<gpu_uint4>& tris,
                      const std::vector<gpu_storage3>& pos)
{
    BruteHit best;
    for (uint32_t i = 0; i < tris.size(); i++) {
        float t;
        gpu_float2 bary;
        if (rt_intersect_triangle(ro, rd, gpu_load3(pos[tris[i].x]), gpu_load3(pos[tris[i].y]),
                                  gpu_load3(pos[tris[i].z]), t, bary)
            && t < best.t) {
            best.hit = true;
            best.t = t;
            best.triIdx = i;
            best.bary = bary;
        }
    }
    return best;
}

} // namespace

int main()
{
    // ---- rt_intersect_triangle: hand-computed cases on the canonical tri ----
    {
        gpu_float3 v0(0, 0, 0), v1(1, 0, 0), v2(0, 1, 0);
        float t;
        gpu_float2 b;
        bool front = rt_intersect_triangle(gpu_float3(0.25f, 0.25f, 1.0f),
                                           gpu_float3(0, 0, -1), v0, v1, v2, t, b);
        check(front && near1(t, 1.0f, 1e-6f) && near1(b.x, 0.25f, 1e-6f)
                  && near1(b.y, 0.25f, 1e-6f), "triFrontHit");
        bool back = rt_intersect_triangle(gpu_float3(0.25f, 0.25f, -1.0f),
                                          gpu_float3(0, 0, 1), v0, v1, v2, t, b);
        check(back && near1(t, 1.0f, 1e-6f), "triBackfaceHit");  // no culling
        check(!rt_intersect_triangle(gpu_float3(0.7f, 0.7f, 1.0f), gpu_float3(0, 0, -1),
                                     v0, v1, v2, t, b), "triMissOutside");
        check(!rt_intersect_triangle(gpu_float3(0.25f, 0.25f, 1.0f), gpu_float3(1, 0, 0),
                                     v0, v1, v2, t, b), "triMissParallel");
        check(!rt_intersect_triangle(gpu_float3(0.25f, 0.25f, 1.0f), gpu_float3(0, 0, 1),
                                     v0, v1, v2, t, b), "triMissBehind");
        check(!rt_intersect_triangle(gpu_float3(0.25f, 0.25f, 0.0f), gpu_float3(0, 0, -1),
                                     v0, v1, v2, t, b), "triEpsilonReject");  // t < 1e-5
    }

    // ---- rt_aabb_hit: unit cases on the [-1,1]^3 box ----
    {
        gpu_float3 bmin(-1, -1, -1), bmax(1, 1, 1);
        gpu_float3 d = gpu_float3(0, 0, -1);
        check(rt_aabb_hit(gpu_float3(0, 0, 5), 1.0f / d, bmin, bmax, INFINITY), "aabbHit");
        check(!rt_aabb_hit(gpu_float3(3, 0, 5), 1.0f / d, bmin, bmax, INFINITY), "aabbMiss");
        check(!rt_aabb_hit(gpu_float3(0, 0, 5), 1.0f / d, bmin, bmax, 3.0f), "aabbTMaxCutoff");
        check(rt_aabb_hit(gpu_float3(0, 0, 0), 1.0f / d, bmin, bmax, INFINITY), "aabbInside");
        check(!rt_aabb_hit(gpu_float3(0, 0, -5), 1.0f / d, bmin, bmax, INFINITY), "aabbBehind");
    }

    // ---- single-triangle scene: attribute interpolation + hit metadata ----
    {
        std::vector<gpu_storage3> pos = { store3(0, 0, 0), store3(1, 0, 0), store3(0, 1, 0) };
        std::vector<gpu_uint4> tris = { gpu_uint4(0, 1, 2, 7) };
        BvhBuildResult bvh = buildThreadedBvh6(pos, tris);
        RtHit hit;
        hit.t = INFINITY;
        rt_closest_hit(gpu_float3(0.25f, 0.25f, 1.0f), gpu_float3(0, 0, -1),
                       bvh.nodes.data(), bvh.numNodesPerDir, tris.data(), pos.data(), hit);
        check(hit.hit && hit.userData == 7u && near1(hit.t, 1.0f, 1e-6f), "singleTriHit");

        std::vector<gpu_float2> uvs = { gpu_float2(0, 0), gpu_float2(1, 0), gpu_float2(0, 1) };
        gpu_float2 uv = rt_interp_uv(tris.data(), uvs.data(), hit);
        check(near1(uv.x, 0.25f, 1e-6f) && near1(uv.y, 0.25f, 1e-6f), "interpUv");

        std::vector<gpu_storage3> nrm = { store3(0, 0, 1), store3(1, 0, 0), store3(0, 1, 0) };
        gpu_float3 expect = normalize(gpu_float3(0.25f, 0.25f, 0.5f));  // bary (0.25, 0.25)
        gpu_float3 sn = rt_shading_normal(tris.data(), nrm.data(), hit);
        check(near1(sn.x, expect.x, 1e-6f) && near1(sn.y, expect.y, 1e-6f)
                  && near1(sn.z, expect.z, 1e-6f), "shadingNormalInterp");

        std::vector<gpu_storage3> zeroNrm = { store3(0, 0, 0), store3(0, 0, 0), store3(0, 0, 0) };
        gpu_float3 fb = rt_shading_normal(tris.data(), zeroNrm.data(), hit);
        check(near1(fb.x, hit.n.x, 0.0f) && near1(fb.y, hit.n.y, 0.0f)
                  && near1(fb.z, hit.n.z, 0.0f), "shadingNormalZeroFallback");
    }

    // ---- random scene: BVH traversal vs brute force, host personality ----
    std::mt19937 rng(20260903);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    auto rnd = [&](float lo, float hi) { return lo + (hi - lo) * uni(rng); };

    const uint32_t numTris = 300;
    std::vector<gpu_storage3> pos;
    std::vector<gpu_uint4> tris;
    for (uint32_t i = 0; i < numTris; i++) {
        // Reject slivers: a tiny Moller-Trumbore det amplifies rounding-mode
        // differences between personalities into visible t error, which would
        // test conditioning, not traversal.
        gpu_float3 a, b, c;
        do {
            a = gpu_float3(rnd(-2, 2), rnd(-2, 2), rnd(-2, 2));
            float s = rnd(0.05f, 0.7f);
            b = a + gpu_float3(rnd(-1, 1), rnd(-1, 1), rnd(-1, 1)) * s;
            c = a + gpu_float3(rnd(-1, 1), rnd(-1, 1), rnd(-1, 1)) * s;
        } while (length(cross(b - a, c - a)) < 5e-3f);
        uint32_t i0 = (uint32_t)pos.size();
        pos.push_back(store3(a.x, a.y, a.z));
        pos.push_back(store3(b.x, b.y, b.z));
        pos.push_back(store3(c.x, c.y, c.z));
        tris.push_back(gpu_uint4(i0, i0 + 1, i0 + 2, i));
    }
    // Vertex attributes for the GPU pass: any nonzero normals + distinct uvs.
    std::vector<gpu_storage3> nrm;
    std::vector<gpu_float2> uvs;
    for (const gpu_storage3& p : pos) {
        gpu_float3 v = gpu_load3(p) + gpu_float3(0.1f, 0.2f, 3.0f);  // never zero-length
        gpu_float3 n = normalize(v);
        nrm.push_back(store3(n.x, n.y, n.z));
        uvs.push_back(gpu_float2(p.x, p.y));
    }
    BvhBuildResult bvh = buildThreadedBvh6(pos, tris);  // reorders `tris`

    const uint32_t numRays = 20000;
    std::vector<gpu_float3> rayO(numRays), rayD(numRays);
    std::vector<float> rayTMax(numRays);
    std::vector<BruteHit> brute(numRays);
    std::vector<RtHit> host(numRays);
    std::vector<char> hostOcc(numRays);
    uint32_t hits = 0, traversalMismatch = 0, occMismatch = 0;
    for (uint32_t r = 0; r < numRays; r++) {
        gpu_float3 ro(rnd(-3, 3), rnd(-3, 3), rnd(-3, 3));
        gpu_float3 rd;
        if (r % 2 == 0) {
            // Aimed: at a random interior point of a random triangle.
            const gpu_uint4& tri = tris[(uint32_t)(rnd(0, 1) * 0.999f * numTris)];
            float u = rnd(0.05f, 0.6f), v = rnd(0.05f, 0.6f) * (1.0f - u);
            gpu_float3 p = gpu_load3(pos[tri.x]) * (1.0f - u - v)
                         + gpu_load3(pos[tri.y]) * u + gpu_load3(pos[tri.z]) * v;
            rd = normalize(p - ro);
        } else {
            gpu_float3 d(rnd(-1, 1), rnd(-1, 1), rnd(-1, 1));
            if (dot(d, d) < 1e-4f)
                d = gpu_float3(1, 0, 0);
            rd = normalize(d);
        }
        brute[r] = bruteClosest(ro, rd, tris, pos);
        host[r].t = INFINITY;
        rt_closest_hit(ro, rd, bvh.nodes.data(), bvh.numNodesPerDir,
                       tris.data(), pos.data(), host[r]);
        // Exact agreement: same code evaluates every candidate triangle.
        bool same = host[r].hit == brute[r].hit
                 && (!host[r].hit
                     || (host[r].triIdx == brute[r].triIdx && host[r].t == brute[r].t
                         && host[r].bary.x == brute[r].bary.x
                         && host[r].bary.y == brute[r].bary.y
                         && host[r].userData == tris[brute[r].triIdx].w));
        if (!same)
            traversalMismatch++;
        if (host[r].hit)
            hits++;

        // Occlusion: margins around the known closest hit exercise both
        // outcomes; a fixed horizon compares against brute force directly.
        float tMax = brute[r].hit ? brute[r].t * (r % 4 < 2 ? 0.9f : 1.1f) : 6.0f;
        bool occ = rt_occluded(ro, rd, bvh.nodes.data(), bvh.numNodesPerDir,
                               tris.data(), pos.data(), tMax);
        bool expect = brute[r].hit ? brute[r].t < tMax : false;
        if (occ != expect)
            occMismatch++;
        rayO[r] = ro;
        rayD[r] = rd;
        rayTMax[r] = tMax;
        hostOcc[r] = occ ? 1 : 0;
    }
    std::cout << "  " << hits << "/" << numRays << " rays hit\n";
    check(hits > numRays / 3, "rayCoverage");  // aimed rays guarantee plenty
    check(traversalMismatch == 0, "bvhClosestHitVsBruteForce");
    check(occMismatch == 0, "bvhOccludedVsBruteForce");

    // ---- GPU pass: identical rays through rt_test_trace ----
    try {
        rhi::DeviceDesc desc;
        // Safe math: the comparison is about traversal correctness; under
        // fast math the compiler's per-kernel contraction adds value noise
        // (grazing hits especially). Also exercises DeviceDesc::safeMath.
        desc.safeMath = true;
        if (rhi::kNativeBackend == rhi::BackendKind::Metal) {
            desc.shaderSource = readTextFile(std::string(RHI_SHADER_DIR) + "/gpu_portable.h")
                              + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/primitives_shared.h")
                              + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/accel_shared.h")
                              + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/raytrace_gpu.h")
                              + "\n" + readTextFile(std::string(TEST_SHADER_DIR) + "/raytrace_test_gpu.h");
        }
        auto device = rhi::createDevice(rhi::kNativeBackend, desc);
        std::cout << "backend: " << rhi::backendName(rhi::kNativeBackend) << "\n";
        auto stream = device->createStream();
        auto pipeline = device->createPipeline({ "rt_test_trace" });

        auto makeShared = [&](const void* data, size_t bytes, const char* name) {
            auto buf = device->createBuffer({ bytes, rhi::MemoryLocation::Shared, name });
            if (data)
                std::memcpy(buf->hostPtr(), data, bytes);
            else
                std::memset(buf->hostPtr(), 0, bytes);
            return buf;
        };
        std::vector<gpu_float4> rays(numRays * 2);
        for (uint32_t r = 0; r < numRays; r++) {
            rays[r * 2] = gpu_float4(rayO[r], rayTMax[r]);
            rays[r * 2 + 1] = gpu_float4(rayD[r], 0.0f);
        }
        auto rayBuf = makeShared(rays.data(), rays.size() * sizeof(gpu_float4), "rays");
        auto nodeBuf = makeShared(bvh.nodes.data(), bvh.nodes.size() * sizeof(RtBvhNode), "nodes");
        auto triBuf = makeShared(tris.data(), tris.size() * sizeof(gpu_uint4), "tris");
        auto posBuf = makeShared(pos.data(), pos.size() * sizeof(gpu_storage3), "positions");
        auto nrmBuf = makeShared(nrm.data(), nrm.size() * sizeof(gpu_storage3), "normals");
        auto uvBuf = makeShared(uvs.data(), uvs.size() * sizeof(gpu_float2), "uvs");
        auto outBuf = makeShared(nullptr, numRays * RT_TEST_OUT_STRIDE * sizeof(gpu_float4), "out");

        RtTestParams p = {};
        p.n = numRays;
        p.numNodesPerDir = bvh.numNodesPerDir;
        stream->dispatch(*pipeline, { (numRays + PRIM_TILE - 1) / PRIM_TILE, 1, 1 },
                         { PRIM_TILE, 1, 1 }, &p, sizeof(p),
                         { rayBuf.get(), nodeBuf.get(), triBuf.get(), posBuf.get(),
                           nrmBuf.get(), uvBuf.get(), outBuf.get() });
        stream->waitIdle();

        const gpu_float4* out = (const gpu_float4*)outBuf->hostPtr();
        uint32_t gpuMismatch = 0;
        for (uint32_t r = 0; r < numRays; r++) {
            const gpu_float4& o0 = out[r * RT_TEST_OUT_STRIDE];
            const gpu_float4& o1 = out[r * RT_TEST_OUT_STRIDE + 1];
            const gpu_float4& o2 = out[r * RT_TEST_OUT_STRIDE + 2];
            const gpu_float4& o3 = out[r * RT_TEST_OUT_STRIDE + 3];
            bool gHit = o1.x != 0.0f;
            bool gOcc = o1.y != 0.0f;
            bool ok = gHit == host[r].hit && gOcc == (hostOcc[r] != 0);
            if (ok && host[r].hit) {
                ok = (uint32_t)o0.y == host[r].triIdx && (uint32_t)o1.z == host[r].userData;
                // Value comparison only at non-grazing incidence: near the
                // triangle plane Moller-Trumbore's det -> 0 and last-ulp
                // host/GPU differences amplify without bound — that is
                // conditioning, not traversal. Structural checks above stay
                // strict for every ray.
                if (ok && std::fabs(dot(rayD[r], host[r].n)) > 0.05f) {
                    gpu_float3 sn = rt_shading_normal(tris.data(), nrm.data(), host[r]);
                    gpu_float2 uv = rt_interp_uv(tris.data(), uvs.data(), host[r]);
                    float tol = 1e-4f * (1.0f + host[r].t);
                    ok = near1(o0.x, host[r].t, tol)
                      && near1(o0.z, host[r].bary.x, 1e-4f) && near1(o0.w, host[r].bary.y, 1e-4f)
                      && near1(o2.x, sn.x, 1e-3f) && near1(o2.y, sn.y, 1e-3f)
                      && near1(o2.z, sn.z, 1e-3f)
                      && near1(o3.x, uv.x, 1e-3f) && near1(o3.y, uv.y, 1e-3f);
                }
            }
            if (!ok && gpuMismatch++ < 5)
                std::cerr << "  ray " << r << ": host hit=" << host[r].hit
                          << " t=" << host[r].t << " tri=" << host[r].triIdx
                          << " | gpu hit=" << gHit << " t=" << o0.x << " tri=" << o0.y << "\n";
        }
        check(gpuMismatch == 0, "gpuTraversalMatchesHost");
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
