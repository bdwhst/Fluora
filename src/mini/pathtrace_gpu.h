#ifndef MINI_PATHTRACE_GPU_H
#define MINI_PATHTRACE_GPU_H
// FluoraMini path tracing kernels (docs/metal-rhi-design.md M1+M2, NEE/MIS and
// media in M4 part 2). Two execution modes share every helper below, so their
// images are bitwise identical:
//  - megakernel (M1): one dispatch per sample, full path per thread
//  - wavefront (M2): raygen -> per-bounce [intersect -> shade -> shadow]
//    stages with atomic work queues and indirect dispatch; terminated paths
//    simply are not re-enqueued (no compaction pass)
//
// Direct lighting (the CUDA renderer's misIntegrator port): at every
// non-specular vertex one light is sampled uniformly (next-event estimation,
// power-heuristic MIS against the BSDF pdf); emissive/env hits reached by BSDF
// sampling carry the complementary weight. Every contribution is converted to
// film RGB and added to the accumulator on its own, in the same order in both
// modes (per bounce: the hit's emission, else its NEE term), which is what
// keeps mega == wavefront bitwise now that a path adds more than once.
//
// Participating media (the volume integrator port, core/medium_shared.h):
// every ray segment is traced by miniTrace, which walks surfaces and media
// together — a homogeneous medium is delta-tracked at the hero wavelength
// (the collision either absorbs the path or becomes a phase-function vertex,
// shaded like a surface vertex with HG in place of a BSDF), and surfaceless
// MINI_MAT_INTERFACE hits switch the current medium and continue without
// counting a bounce. Spectral MIS over the four wavelengths follows PBRT-v4:
// the path carries r, the per-wavelength ratio of its sampling pdf to the
// hero's, and every contribution is divided by the average of r (the
// balance heuristic over wavelengths). With no media r stays exactly 1, so
// media-free scenes render bit-for-bit as before. Shadow rays inside media
// use the analytic Beer-Lambert transmittance (no per-wavelength pdf, hence
// no r_l term) and pass through interfaces like camera paths do.
//
// Single-source via the gpu_portable shim (docs/portable-device-code.md).
// Under MSL this file is concatenated last; elsewhere the #includes below
// resolve. texture_gpu.h supplies tex_heap_sample/RhiTex per backend.

#ifndef __METAL_VERSION__
#include "../rhi/gpu_portable.h"
#define GPU_PRIMITIVES_HELPERS_ONLY  // prim_queue_alloc; the kernels live in rhi_cuda.cu
#include "../rhi/primitives_gpu.h"
#include "../rhi/raytrace_gpu.h"
#include "../rhi/texture_gpu.h"
#include "../core/spectrum_shared.h"
#include "../core/bsdf_shared.h"
#include "../core/medium_shared.h"
#include "../core/envmap_shared.h"
#include "../core/light_shared.h"
#include "../core/tonemap_shared.h"
#include "mini_shared.h"
#endif

#define MINI_SCATTER_OFFSET 1e-4f   // origin push-off along the facing normal
#define MINI_SHADOW_TMAX_EPS 1e-3f  // shadow rays stop this short of the light (world units)
#define MINI_MAX_CROSSINGS 32u      // interface pass-throughs per segment before the path is dropped

// Slab test against the unit cube [-0.5, 0.5]^3 in object space. Ray direction
// is unnormalized (object-space transform of a world ray), so t is measured in
// world units and comparable across objects.
GPU_FN inline bool intersectUnitCube(gpu_float3 ro, gpu_float3 rd,
                              GPU_THREAD float& tOut, GPU_THREAD gpu_float3& nOut)
{
    gpu_float3 invD = 1.0f / rd;
    gpu_float3 tA = (gpu_float3(-0.5f) - ro) * invD;
    gpu_float3 tB = (gpu_float3(0.5f) - ro) * invD;
    gpu_float3 tSmall = min(tA, tB);
    gpu_float3 tBig = max(tA, tB);
    float tmin = max(max(tSmall.x, tSmall.y), tSmall.z);
    float tmax = min(min(tBig.x, tBig.y), tBig.z);
    if (tmax < tmin || tmax < 1e-6f)
        return false;
    float t = tmin > 1e-6f ? tmin : tmax;
    gpu_float3 p = ro + rd * t;
    gpu_float3 ap = abs(p);
    float m = max(max(ap.x, ap.y), ap.z);
    gpu_float3 n = gpu_float3(0.0f);
    if (m == ap.x)      n.x = p.x > 0.0f ? 1.0f : -1.0f;
    else if (m == ap.y) n.y = p.y > 0.0f ? 1.0f : -1.0f;
    else                n.z = p.z > 0.0f ? 1.0f : -1.0f;
    tOut = t;
    nOut = n;
    return true;
}

GPU_FN inline bool intersectUnitSphere(gpu_float3 ro, gpu_float3 rd,
                                GPU_THREAD float& tOut, GPU_THREAD gpu_float3& nOut)
{
    float a = dot(rd, rd);
    float b = 2.0f * dot(ro, rd);
    float c = dot(ro, ro) - 0.25f;
    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f)
        return false;
    float sq = sqrt(disc);
    float t0 = (-b - sq) / (2.0f * a);
    float t1 = (-b + sq) / (2.0f * a);
    float t = t0 > 1e-6f ? t0 : (t1 > 1e-6f ? t1 : -1.0f);
    if (t < 0.0f)
        return false;
    tOut = t;
    nOut = normalize(ro + rd * t);
    return true;
}

struct MiniHit {
    float t;
    gpu_float3 n;   // world-space shading normal (interpolated when provided)
    gpu_float3 ng;  // geometric normal: outward for analytic objects, winding for meshes
    gpu_float2 uv;  // mesh uv, (-1,-1) for analytic objects / no texcoords
    int objIdx;     // >=0 analytic object, -2 mesh triangle, -1 miss
    uint matId;
    uint triIdx;    // mesh hits: index into the reordered triangle array
};

// Analytic scaffolding objects tested brute-force; mesh geometry goes through
// the rt_closest_hit seam (src/rhi/raytrace_gpu.h).
GPU_FN inline bool sceneIntersect(gpu_float3 ro, gpu_float3 rd,
                           GPU_DEVICE const MiniObject* objects, uint numObjects,
                           GPU_DEVICE const RtBvhNode* nodes, uint numNodes,
                           GPU_DEVICE const gpu_uint4* tris,
                           GPU_DEVICE const gpu_storage3* positions,
                           GPU_DEVICE const gpu_storage3* normals,
                           GPU_DEVICE const gpu_float2* uvs,
                           GPU_THREAD MiniHit& hit)
{
    hit.t = INFINITY;
    hit.n = gpu_float3(0.0f);
    hit.ng = gpu_float3(0.0f);
    hit.uv = gpu_float2(-1.0f);
    hit.objIdx = -1;
    hit.matId = 0;
    hit.triIdx = 0;
    for (uint i = 0; i < numObjects; i++) {
        GPU_DEVICE const MiniObject& obj = objects[i];
        gpu_float3 roLocal = gpu_xyz(obj.invTransform * gpu_float4(ro, 1.0f));
        gpu_float3 rdLocal = gpu_xyz(obj.invTransform * gpu_float4(rd, 0.0f));
        float t;
        gpu_float3 nLocal;
        bool ok = (obj.geomType == MINI_GEOM_CUBE)
            ? intersectUnitCube(roLocal, rdLocal, t, nLocal)
            : intersectUnitSphere(roLocal, rdLocal, t, nLocal);
        if (ok && t < hit.t) {
            hit.t = t;
            hit.n = normalize(gpu_xyz(obj.invTranspose * gpu_float4(nLocal, 0.0f)));
            hit.ng = hit.n;
            hit.objIdx = (int)i;
            hit.matId = (uint)obj.materialId;
        }
    }
    RtHit rhit;
    rhit.t = hit.t;
    rt_closest_hit(ro, rd, nodes, numNodes, tris, positions, rhit);
    if (rhit.hit) {
        hit.t = rhit.t;
        hit.n = rt_shading_normal(tris, normals, rhit);
        hit.ng = rhit.n;
        hit.uv = rt_interp_uv(tris, uvs, rhit);
        hit.matId = rhit.userData;
        hit.objIdx = -2;
        hit.triIdx = rhit.triIdx;
    }
    return hit.objIdx != -1;
}

// Shadow-ray visibility: any analytic object or triangle within (0, tMax).
GPU_FN inline bool sceneOccluded(gpu_float3 ro, gpu_float3 rd, float tMax,
                                 GPU_DEVICE const MiniObject* objects, uint numObjects,
                                 GPU_DEVICE const RtBvhNode* nodes, uint numNodes,
                                 GPU_DEVICE const gpu_uint4* tris,
                                 GPU_DEVICE const gpu_storage3* positions)
{
    for (uint i = 0; i < numObjects; i++) {
        GPU_DEVICE const MiniObject& obj = objects[i];
        gpu_float3 roLocal = gpu_xyz(obj.invTransform * gpu_float4(ro, 1.0f));
        gpu_float3 rdLocal = gpu_xyz(obj.invTransform * gpu_float4(rd, 0.0f));
        float t;
        gpu_float3 nLocal;
        bool ok = (obj.geomType == MINI_GEOM_CUBE)
            ? intersectUnitCube(roLocal, rdLocal, t, nLocal)
            : intersectUnitSphere(roLocal, rdLocal, t, nLocal);
        if (ok && t < tMax)
            return true;
    }
    return rt_occluded(ro, rd, nodes, numNodes, tris, positions, tMax);
}

// Same jittered pinhole formula as generateRayFromCamera in pathtrace.cu.
GPU_FN inline void generateCameraRay(GPU_PARAMS_REF(MiniParams) P, gpu_uint2 gid,
                              GPU_THREAD GpuRng& rng,
                              GPU_THREAD gpu_float3& ro, GPU_THREAD gpu_float3& rd)
{
    float jx = gpu_rand(rng) - 0.5f;
    float jy = gpu_rand(rng) - 0.5f;
    rd = normalize(gpu_load3(P.camView)
        - gpu_load3(P.camRight) * P.pixelLenX * ((float)gid.x - (float)P.width * 0.5f + jx)
        - gpu_load3(P.camUp)    * P.pixelLenY * ((float)gid.y - (float)P.height * 0.5f + jy));
    ro = gpu_load3(P.camPos);
}

// Environment radiance for an escaped ray: equirect texel RGB, clamped like
// scene.cpp's max-luminance guard, spectralized as an RGB illuminant (the
// ImageInfiniteLight::L port).
GPU_FN inline GpuSpectrum miniEnvRadiance(GPU_DEVICE const RhiTex* texHeap, uint envMapIdx,
                                   gpu_float3 rd, GpuWavelengths swl,
                                   GPU_DEVICE const float* spd,
                                   GPU_DEVICE const float* r2s)
{
    gpu_float3 sky = gpu_xyz(tex_heap_sample(texHeap, envMapIdx, env_equirect_uv(rd)));
    sky = min(sky, gpu_float3(MINI_ENV_MAX_RADIANCE));
    return spd_rgb_illuminant_sample(r2s, spd, sky, swl);
}

// EmissiveMaterial::Le — rgb*emittance as an RGB illuminant.
GPU_FN inline GpuSpectrum miniEmission(GPU_DEVICE const MiniMaterial& mat, GpuWavelengths swl,
                                       GPU_DEVICE const float* spd, GPU_DEVICE const float* r2s)
{
    return spd_rgb_illuminant_sample(r2s, spd, gpu_load3(mat.rgb) * mat.emittance, swl);
}

// Spectral radiance -> film RGB: pixel sensor XYZ (spectrum_shared.h), then
// the host-derived output matrix rows.
GPU_FN inline gpu_float3 miniFilmRgb(GpuSpectrum L, GpuWavelengths swl,
                              GPU_DEVICE const float* spd,
                              gpu_float3 r0, gpu_float3 r1, gpu_float3 r2)
{
    gpu_float3 xyz = spd_to_sensor_xyz(L, swl, spd);
    return gpu_float3(dot(r0, xyz), dot(r1, xyz), dot(r2, xyz));
}

// Everything the shading and tracing helpers need to reach: scene arrays +
// per-dispatch scalars. Built by each kernel from its bound buffers and
// parameter block so the helpers have one signature in both modes; a kernel
// that never traces leaves the geometry-attribute pointers null.
struct MiniShadeCtx {
    GPU_DEVICE const MiniMaterial* materials;
    GPU_DEVICE const MiniObject* objects;
    GPU_DEVICE const RtBvhNode* nodes;
    GPU_DEVICE const gpu_uint4* tris;
    GPU_DEVICE const gpu_storage3* positions;
    GPU_DEVICE const gpu_storage3* normals;
    GPU_DEVICE const gpu_float2* uvs;
    GPU_DEVICE const RtLight* lights;
    GPU_DEVICE const float* envDist;
    GPU_DEVICE const RhiTex* texHeap;
    GPU_DEVICE const float* spd;
    GPU_DEVICE const float* r2s;
    GPU_DEVICE const MediumGpu* media;
    uint numObjects;
    uint numNodes;
    uint numLights;
    uint numMedia;
    uint envMapIdx;
    uint envW, envH;
    gpu_float3 filmR0, filmR1, filmR2;
};

// Uniform light sampler: pmf 1/numLights for every light.
GPU_FN inline float miniLightPmf(GPU_THREAD const MiniShadeCtx& c)
{
    return c.numLights > 0 ? 1.0f / (float)c.numLights : 0.0f;
}

// Solid-angle pdf (times the light pmf) of the light sampler producing the
// emissive point `hit` (at pLight) from the vertex `roVertex` -- the MIS
// weight input for BSDF-sampled emissive hits. roVertex is the scattering
// vertex, not the origin of the last segment: interface crossings restart the
// segment but the light sample would have been drawn at the vertex. Uses the
// primitive's geometric normal like light sampling.
GPU_FN inline float miniHitLightPdf(GPU_THREAD const MiniShadeCtx& c, GPU_THREAD const MiniHit& hit,
                                    gpu_float3 pLight, gpu_float3 roVertex)
{
    float pdfArea;
    gpu_float3 nLight;
    if (hit.objIdx >= 0) {
        GPU_DEVICE const MiniObject& obj = c.objects[hit.objIdx];
        pdfArea = obj.geomType == MINI_GEOM_SPHERE ? light_pdf_area_sphere(obj.transform)
                                                   : light_pdf_area_cube(obj.transform);
        nLight = hit.n;   // analytic normals are the outward geometric normal
    } else {
        gpu_uint4 tri = c.tris[hit.triIdx];
        gpu_float3 v0 = gpu_load3(c.positions[tri.x]);
        gpu_float3 v1 = gpu_load3(c.positions[tri.y]);
        gpu_float3 v2 = gpu_load3(c.positions[tri.z]);
        pdfArea = light_pdf_area_tri(v0, v1, v2);
        nLight = normalize(cross(v1 - v0, v2 - v0));
    }
    return miniLightPmf(c) * light_pdf_area(pdfArea, pLight, nLight, roVertex);
}

// Draws this material's spectral parameters at a hit (albedo with texture,
// conductor eta/k or reflectance-mode k).
GPU_FN inline void miniMaterialSpectra(uint type, GPU_DEVICE const MiniMaterial& mat,
                                       gpu_float2 uv, GPU_THREAD const MiniShadeCtx& c,
                                       GpuWavelengths swl,
                                       GPU_THREAD GpuSpectrum& reflectance,
                                       GPU_THREAD GpuSpectrum& eta, GPU_THREAD GpuSpectrum& k)
{
    if (type == MINI_MAT_DIFFUSE) {
        gpu_float3 albedo = gpu_load3(mat.rgb);
        if (mat.texIdx != MINI_TEX_NONE)
            albedo *= gpu_xyz(tex_heap_sample(c.texHeap, mat.texIdx, uv));
        reflectance = spd_rgb_albedo_sample(c.r2s, albedo, swl);
    } else if (type == MINI_MAT_CONDUCTOR) {
        if (mat.etaSpd != SPD_NONE && mat.kSpd != SPD_NONE) {
            eta = spd_dense_sample(c.spd, mat.etaSpd, swl);
            k = spd_dense_sample(c.spd, mat.kSpd, swl);
        } else {
            // PBRT reflectance mode for RGB "microfacet" materials (the CUDA
            // renderer rejects these scenes): eta = 1, k = 2 sqrt(r)/sqrt(1-r).
            GpuSpectrum r = spd_rgb_albedo_sample(c.r2s, gpu_load3(mat.rgb), swl);
            eta = GpuSpectrum(1.0f);
            for (int i = 0; i < SPD_N_SAMPLES; i++)
                k[i] = 2.0f * sqrt(r[i]) / sqrt(max(1.0f - r[i], 1e-4f));
        }
    }
}

// Delta BSDFs get no next-event estimation and no MIS on their next hit.
GPU_FN inline bool miniMaterialSpecular(uint type, GPU_DEVICE const MiniMaterial& mat)
{
    return type == MINI_MAT_GLASS || (type == MINI_MAT_CONDUCTOR && mat.roughness < 1e-3f);
}

// ==========================================================================
// Media
// ==========================================================================

// The medium a ray is in after crossing a surface with the direction `dir`,
// given the surface's medium interface (PBRT: leaving along the normal means
// the outside). Surfaces without an interface leave the medium alone.
GPU_FN inline int miniCrossMedium(GPU_DEVICE const MiniMaterial& mat, gpu_float3 dir,
                                  gpu_float3 n, int medium)
{
    if (mat.mediumIn == mat.mediumOut)
        return medium;
    return dot(dir, n) > 0.0f ? mat.mediumOut : mat.mediumIn;
}

GPU_FN inline void miniMediumSigma(GPU_THREAD const MiniShadeCtx& c, int medium, GpuWavelengths swl,
                                   GPU_THREAD GpuSpectrum& sigmaA, GPU_THREAD GpuSpectrum& sigmaS)
{
    GPU_DEVICE const MediumGpu& m = c.media[medium];
    sigmaA = spd_dense_sample(c.spd, m.sigmaASpd, swl);
    sigmaS = spd_dense_sample(c.spd, m.sigmaSSpd, swl);
}

// What a traced segment ended with.
#define MINI_TRACE_MISS     0  // escaped the scene (environment)
#define MINI_TRACE_SURFACE  1  // hit a shadeable surface: `hit` is valid
#define MINI_TRACE_SCATTER  2  // real scatter inside `medium` at hit.t along the ray
#define MINI_TRACE_ABSORBED 3  // absorbed in a medium (or dropped): path ends

// Traces one path segment from (ro, rd) through surfaces and media. Advances
// ro across MINI_MAT_INTERFACE hits (switching `medium`), so on return the
// event lies at ro + rd * hit.t. In a medium the segment is delta-tracked at
// the hero wavelength: the sampled collision is a real absorption/scatter
// event (homogeneous majorant == sigma_t, so there are no null collisions);
// throughput and r pick up the per-wavelength ratios of the transmittance
// and scattering coefficients to the hero's. Consumes RNG draws only inside
// media, so media-free scenes keep their RNG streams.
GPU_FN inline int miniTrace(GPU_THREAD const MiniShadeCtx& c,
                            GPU_THREAD gpu_float3& ro, gpu_float3 rd, GPU_THREAD int& medium,
                            GPU_THREAD GpuSpectrum& throughput, GPU_THREAD GpuSpectrum& r,
                            GpuWavelengths swl, GPU_THREAD GpuRng& rng,
                            GPU_THREAD MiniHit& hit)
{
    for (uint crossing = 0; crossing < MINI_MAX_CROSSINGS; crossing++) {
        bool found = sceneIntersect(ro, rd, c.objects, c.numObjects, c.nodes, c.numNodes,
                                    c.tris, c.positions, c.normals, c.uvs, hit);
        if (medium >= 0) {
            GpuSpectrum sigmaA, sigmaS;
            miniMediumSigma(c, medium, swl, sigmaA, sigmaS);
            GpuSpectrum sigmaT = sigmaA + sigmaS;
            float tMax = found ? hit.t : INFINITY;
            // Exponential distance at the hero wavelength (sample_exponential);
            // sigma_t == 0 never collides.
            float u = gpu_rand(rng);
            float t = sigmaT[0] > 0.0f ? -log(1.0f - u) / sigmaT[0] : INFINITY;
            if (t < tMax) {
                GpuSpectrum Tmaj = medium_transmittance(sigmaT, t);
                float uMode = gpu_rand(rng);
                if (uMode < sigmaA[0] / sigmaT[0])
                    return MINI_TRACE_ABSORBED;
                // Real scatter: f = Tmaj * sigma_s, sampled with pdf Tmaj[0] * sigma_s[0].
                float pdf = Tmaj[0] * sigmaS[0];
                GpuSpectrum ratio = Tmaj * sigmaS / pdf;
                throughput *= ratio;
                r *= ratio;
                hit.t = t;
                hit.objIdx = -3;
                hit.matId = (uint)medium;
                return MINI_TRACE_SCATTER;
            }
            // No collision before the surface (or ever): pdf Tmaj[0] of
            // reaching it, transmittance Tmaj for every wavelength.
            GpuSpectrum Tmaj = medium_transmittance(sigmaT, tMax);
            GpuSpectrum ratio = Tmaj / Tmaj[0];
            throughput *= ratio;
            r *= ratio;
        }
        if (!found)
            return MINI_TRACE_MISS;
        GPU_DEVICE const MiniMaterial& mat = c.materials[hit.matId];
        if (mat.type != MINI_MAT_INTERFACE)
            return MINI_TRACE_SURFACE;
        // Surfaceless boundary: switch medium, restart just past it.
        medium = miniCrossMedium(mat, rd, hit.ng, medium);
        ro = ro + rd * hit.t + (dot(rd, hit.ng) > 0.0f ? hit.ng : -hit.ng) * MINI_SCATTER_OFFSET;
    }
    return MINI_TRACE_ABSORBED;
}

// Spectral transmittance from `ro` toward the light at distance tMax (INFINITY
// for the environment): 0 when a shadeable surface blocks the way, else the
// Beer-Lambert product over the media crossed. Media-free scenes take the
// any-hit path, which is what they always did.
GPU_FN inline GpuSpectrum miniShadowTransmittance(GPU_THREAD const MiniShadeCtx& c,
                                                  gpu_float3 ro, gpu_float3 rd, float tMax,
                                                  int medium, GpuWavelengths swl)
{
    if (c.numMedia == 0) {
        bool blocked = sceneOccluded(ro, rd, tMax, c.objects, c.numObjects, c.nodes, c.numNodes,
                                     c.tris, c.positions);
        return GpuSpectrum(blocked ? 0.0f : 1.0f);
    }
    GpuSpectrum T = GpuSpectrum(1.0f);
    for (uint crossing = 0; crossing < MINI_MAX_CROSSINGS; crossing++) {
        MiniHit hit;
        bool found = sceneIntersect(ro, rd, c.objects, c.numObjects, c.nodes, c.numNodes,
                                    c.tris, c.positions, c.normals, c.uvs, hit);
        bool reached = !found || hit.t >= tMax;
        if (medium >= 0) {
            GpuSpectrum sigmaA, sigmaS;
            miniMediumSigma(c, medium, swl, sigmaA, sigmaS);
            T *= medium_transmittance(sigmaA + sigmaS, reached ? tMax : hit.t);
        }
        if (reached)
            return T;
        GPU_DEVICE const MiniMaterial& mat = c.materials[hit.matId];
        if (mat.type != MINI_MAT_INTERFACE)
            return GpuSpectrum(0.0f);
        medium = miniCrossMedium(mat, rd, hit.ng, medium);
        ro = ro + rd * hit.t + (dot(rd, hit.ng) > 0.0f ? hit.ng : -hit.ng) * MINI_SCATTER_OFFSET;
        tMax -= hit.t;
    }
    return GpuSpectrum(0.0f);
}

// ==========================================================================
// Vertex shading (surface BSDFs and medium phase function)
// ==========================================================================

// A next-event-estimation candidate: the shadow ray plus the spectral
// contribution it adds, scaled by the transmittance to the light (already
// MIS-weighted, divided by the light pdf and by the spectral-MIS average).
struct MiniShadowRay {
    gpu_float3 origin;
    gpu_float3 dir;
    float tMax;
    GpuSpectrum L;
    int medium;    // medium the ray starts in
};

// Samples one light for the vertex p (surface: facing normal nF; medium
// vertex: type MINI_MAT_MEDIUM, nF unused) and evaluates the BSDF or phase
// function toward it (sample_Ld port). Consumes exactly four RNG draws.
// Returns false when nothing can be contributed (no lights, back-facing,
// zero BSDF/pdf).
GPU_FN inline bool miniSampleDirect(uint type, GPU_DEVICE const MiniMaterial& mat,
                                    gpu_float3 p, gpu_float3 nF, gpu_float3 rd, gpu_float2 uv,
                                    int medium, GPU_THREAD const MiniShadeCtx& c,
                                    GpuSpectrum throughput, GpuSpectrum r, GpuWavelengths swl,
                                    GPU_THREAD GpuRng& rng, GPU_THREAD MiniShadowRay& sr)
{
    float uL = gpu_rand(rng);
    float u1 = gpu_rand(rng);
    float u2 = gpu_rand(rng);
    float u3 = gpu_rand(rng);
    if (c.numLights == 0)
        return false;
    int li = (int)(uL * (float)c.numLights);
    li = li < 0 ? 0 : (li >= (int)c.numLights ? (int)c.numLights - 1 : li);
    RtLight lt = c.lights[li];

    gpu_float3 wi;
    float pdfL;
    GpuSpectrum Le;
    float tMax;
    if (lt.type == RT_LIGHT_ENV) {
        if (!env_sample_dir(c.envDist, c.envW, c.envH, gpu_float2(u1, u2), wi, pdfL))
            return false;
        Le = miniEnvRadiance(c.texHeap, c.envMapIdx, wi, swl, c.spd, c.r2s);
        tMax = INFINITY;
    } else {
        LightAreaSample s;
        uint lmat;
        if (lt.type == RT_LIGHT_TRI) {
            gpu_uint4 tri = c.tris[lt.index];
            s = light_sample_tri(gpu_load3(c.positions[tri.x]), gpu_load3(c.positions[tri.y]),
                                 gpu_load3(c.positions[tri.z]), gpu_float2(u1, u2));
            lmat = tri.w;
        } else {
            GPU_DEVICE const MiniObject& obj = c.objects[lt.index];
            s = lt.type == RT_LIGHT_SPHERE
                ? light_sample_sphere(obj.transform, p, gpu_float2(u1, u2))
                : light_sample_cube(obj.transform, gpu_float3(u1, u2, u3));
            lmat = (uint)obj.materialId;
        }
        float dist;
        pdfL = light_area_to_solid_angle(s, p, wi, dist);
        if (pdfL <= 0.0f)
            return false;
        Le = miniEmission(c.materials[lmat], swl, c.spd, c.r2s);
        tMax = dist - MINI_SHADOW_TMAX_EPS;
        if (tMax <= 0.0f)
            return false;
    }
    pdfL *= miniLightPmf(c);

    GpuSpectrum f;
    float pdfB;
    bool ok;
    if (type == MINI_MAT_MEDIUM) {
        // Phase function: scalar, no hemisphere restriction. wo = -rd.
        float ph = hg_phase(dot(-rd, wi), c.media[medium].g);
        f = GpuSpectrum(ph);
        pdfB = ph;
        ok = ph > 0.0f;
    } else {
        if (dot(wi, nF) <= 0.0f)   // reflection-only BSDFs see nothing from below
            return false;
        GpuSpectrum reflectance = GpuSpectrum(0.0f), eta = GpuSpectrum(1.0f), k = GpuSpectrum(0.0f);
        miniMaterialSpectra(type, mat, uv, c, swl, reflectance, eta, k);
        if (type == MINI_MAT_DIFFUSE)
            ok = bsdf_eval_lambert(reflectance, nF, wi, f, pdfB);
        else if (type == MINI_MAT_CONDUCTOR)
            ok = bsdf_eval_conductor(eta, k, mat.roughness, nF, rd, wi, f, pdfB);
        else
            ok = false;   // glass is specular: never reaches here
    }
    if (!ok)
        return false;
    float w = mis_power2(pdfL, pdfB);
    GpuSpectrum Ls = Le * f * throughput * (w / pdfL);
    Ls = Ls / spd_average(r);
    sr.origin = type == MINI_MAT_MEDIUM ? p : p + nF * MINI_SCATTER_OFFSET;
    sr.dir = wi;
    sr.tMax = tMax;
    sr.L = Ls;
    sr.medium = medium;
    return true;
}

// Samples the BSDF (or the phase function at a medium vertex) and updates the
// path (get_bxdf + sample_f port). Shared by the megakernel switch and the
// specialized shade kernel, so both modes consume identical RNG streams and
// math. `type` is mat.type in the megakernel (dynamic branch) and a function
// constant in wf_shade (branch folds at pipeline creation). Surfaces with a
// medium interface move the path into the medium on the side it leaves
// toward. Returns false when the sample is absorbed.
GPU_FN inline bool miniScatter(uint type, GPU_DEVICE const MiniMaterial& mat,
                        gpu_float3 p, gpu_float3 n, gpu_float2 uv,
                        GPU_THREAD const MiniShadeCtx& c,
                        GPU_THREAD gpu_float3& ro, GPU_THREAD gpu_float3& rd,
                        GPU_THREAD int& medium,
                        GPU_THREAD GpuSpectrum& throughput,
                        GPU_THREAD GpuWavelengths& swl, GPU_THREAD GpuRng& rng,
                        GPU_THREAD float& pdf)
{
    if (type == MINI_MAT_MEDIUM) {
        // HG sampling: throughput *= phase/pdf == 1 (the sampled pdf is the
        // phase value), so only the direction changes.
        float u1 = gpu_rand(rng);
        float u2 = gpu_rand(rng);
        gpu_float3 wi;
        pdf = hg_sample(-rd, c.media[medium].g, u1, u2, wi);
        if (!(pdf > 0.0f))
            return false;
        ro = p;
        rd = wi;
        return true;
    }
    gpu_float3 nF = dot(n, rd) < 0.0f ? n : -n;
    bool alive;
    pdf = 0.0f;
    if (type == MINI_MAT_DIFFUSE) {
        GpuSpectrum reflectance, eta, k;
        miniMaterialSpectra(type, mat, uv, c, swl, reflectance, eta, k);
        float u1 = gpu_rand(rng);
        float u2 = gpu_rand(rng);
        alive = bsdf_sample_lambert(reflectance, nF, u1, u2, rd, throughput, pdf);
    } else if (type == MINI_MAT_CONDUCTOR) {
        GpuSpectrum reflectance, eta, k;
        miniMaterialSpectra(type, mat, uv, c, swl, reflectance, eta, k);
        float u1 = gpu_rand(rng);
        float u2 = gpu_rand(rng);
        alive = bsdf_sample_conductor(eta, k, mat.roughness, nF, u1, u2, rd, throughput, pdf);
    } else {  // MINI_MAT_GLASS
        float etaVal = mat.ior;
        if (mat.etaSpd != SPD_NONE) {
            // Dispersive eta: evaluate at the hero wavelength and collapse the
            // secondary wavelengths (DielectricMaterial::get_bxdf).
            int o = (int)(swl.lambda.x + 0.5f) - (int)SPD_LAMBDA_MIN;
            o = spd_clampi(o, 0, (int)SPD_TABLE_SIZE - 1);
            etaVal = c.spd[mat.etaSpd + (uint)o];
            spd_terminate_secondary(swl);
        }
        float u = gpu_rand(rng);
        alive = bsdf_sample_dielectric(etaVal, n, u, rd, throughput, pdf);
    }
    if (!alive)
        return false;
    ro = p + nF * (dot(rd, nF) >= 0.0f ? MINI_SCATTER_OFFSET : -MINI_SCATTER_OFFSET);
    // The refracted/reflected direction is defined against the same normal,
    // so its side of n says which medium the path continues in.
    medium = miniCrossMedium(mat, rd, n, medium);
    return true;
}

// One non-emissive path vertex: next-event estimation (unless the BSDF is a
// delta) then BSDF/phase sampling. The order of RNG draws is fixed here for
// both modes. `haveShadow` reports a shadow ray to trace; the return value
// says whether the path continues.
GPU_FN inline bool miniShadeVertex(uint type, GPU_DEVICE const MiniMaterial& mat,
                                   gpu_float3 p, gpu_float3 n, gpu_float2 uv,
                                   GPU_THREAD const MiniShadeCtx& c,
                                   GPU_THREAD gpu_float3& ro, GPU_THREAD gpu_float3& rd,
                                   GPU_THREAD int& medium,
                                   GPU_THREAD GpuSpectrum& throughput, GpuSpectrum r,
                                   GPU_THREAD GpuWavelengths& swl, GPU_THREAD GpuRng& rng,
                                   GPU_THREAD float& pdf, GPU_THREAD bool& specular,
                                   GPU_THREAD bool& haveShadow, GPU_THREAD MiniShadowRay& sr)
{
    specular = type == MINI_MAT_MEDIUM ? false : miniMaterialSpecular(type, mat);
    haveShadow = false;
    if (!specular) {
        gpu_float3 nF = dot(n, rd) < 0.0f ? n : -n;
        haveShadow = miniSampleDirect(type, mat, p, nF, rd, uv, medium, c, throughput, r, swl,
                                      rng, sr);
    }
    return miniScatter(type, mat, p, n, uv, c, ro, rd, medium, throughput, swl, rng, pdf);
}

// Adds a path contribution to the film after the spectral-MIS division.
GPU_FN inline void miniAddRadiance(GPU_DEVICE gpu_float4* accum, uint pixel, GpuSpectrum Ls,
                                   GpuSpectrum r, GpuWavelengths swl,
                                   GPU_THREAD const MiniShadeCtx& c)
{
    Ls = Ls / spd_average(r);
    gpu_float3 rgb = miniFilmRgb(Ls, swl, c.spd, c.filmR0, c.filmR1, c.filmR2);
    if (gpu_all_finite(rgb))
        accum[pixel] += gpu_float4(rgb, 0.0f);
}

// Traces a shadow ray and lands its contribution (both modes).
GPU_FN inline void miniResolveShadow(GPU_DEVICE gpu_float4* accum, uint pixel,
                                     GPU_THREAD const MiniShadowRay& sr, GpuWavelengths swl,
                                     GPU_THREAD const MiniShadeCtx& c)
{
    GpuSpectrum T = miniShadowTransmittance(c, sr.origin, sr.dir, sr.tMax, sr.medium, swl);
    gpu_float3 rgb = miniFilmRgb(sr.L * T, swl, c.spd, c.filmR0, c.filmR1, c.filmR2);
    if (gpu_all_finite(rgb))
        accum[pixel] += gpu_float4(rgb, 0.0f);
}

// ==========================================================================
// Megakernel mode (M1)
// ==========================================================================

GPU_KERNEL(pathtraceKernel, GPU_TID_2D)(GPU_KERNEL_PARAMS(MiniParams, P),
    GPU_BUFFER(gpu_float4, accum),
    GPU_BUFFER(const MiniMaterial, materials),
    GPU_BUFFER(const MiniObject, objects),
    GPU_BUFFER(const RtBvhNode, bvhNodes),
    GPU_BUFFER(const gpu_uint4, tris),
    GPU_BUFFER(const gpu_storage3, positions),
    GPU_BUFFER(const RhiTex, texHeap),
    GPU_BUFFER(const gpu_storage3, normals),
    GPU_BUFFER(const gpu_float2, uvs),
    GPU_BUFFER(const float, spd),
    GPU_BUFFER(const float, r2s),
    GPU_BUFFER(const RtLight, lights),
    GPU_BUFFER(const float, envDist),
    GPU_BUFFER(const MediumGpu, media))
{
    gpu_uint2 gid = GPU_GLOBAL_ID_XY;
    if (gid.x >= P.width || gid.y >= P.height)
        return;
    uint idx = gid.y * P.width + gid.x;

    MiniShadeCtx c;
    c.materials = materials;
    c.objects = objects;
    c.nodes = bvhNodes;
    c.tris = tris;
    c.positions = positions;
    c.normals = normals;
    c.uvs = uvs;
    c.lights = lights;
    c.envDist = envDist;
    c.texHeap = texHeap;
    c.spd = spd;
    c.r2s = r2s;
    c.media = media;
    c.numObjects = P.numObjects;
    c.numNodes = P.bvhNumNodes;
    c.numLights = P.numLights;
    c.numMedia = P.numMedia;
    c.envMapIdx = P.envMapIdx;
    c.envW = P.envW;
    c.envH = P.envH;
    c.filmR0 = gpu_load3(P.filmR0);
    c.filmR1 = gpu_load3(P.filmR1);
    c.filmR2 = gpu_load3(P.filmR2);

    GpuRng rng;
    rng.state = idx * 9781u + P.iter * 26699u + 1u;
    gpu_rand(rng);  // decorrelate the low-entropy seed
    gpu_rand(rng);

    gpu_float3 ro, rd;
    generateCameraRay(P, gid, rng, ro, rd);
    GpuWavelengths swl = spd_sample_visible(gpu_rand(rng));

    GpuSpectrum throughput = GpuSpectrum(1.0f);
    GpuSpectrum r = GpuSpectrum(1.0f);
    int medium = P.cameraMedium;
    float lastPdf = 0.0f;
    bool prevSpecular = false;

    for (uint depth = 0; depth < P.maxDepth; depth++) {
        gpu_float3 roVertex = ro;
        MiniHit hit;
        int ev = miniTrace(c, ro, rd, medium, throughput, r, swl, rng, hit);
        if (ev == MINI_TRACE_ABSORBED)
            break;
        if (ev == MINI_TRACE_MISS) {
            // Escaped: environment radiance if the scene has a SKYBOX, else black.
            if (P.envMapIdx != MINI_ENV_NONE) {
                GpuSpectrum Ls = throughput * miniEnvRadiance(texHeap, P.envMapIdx, rd, swl, spd, r2s);
                if (!(depth == 0 || prevSpecular))
                    Ls *= mis_power2(lastPdf, miniLightPmf(c) * env_pdf_dir(envDist, P.envW, P.envH, rd));
                miniAddRadiance(accum, idx, Ls, r, swl, c);
            }
            break;
        }
        gpu_float3 p = ro + rd * hit.t;
        uint type;
        uint matId = 0;
        if (ev == MINI_TRACE_SCATTER) {
            type = MINI_MAT_MEDIUM;
        } else {
            matId = hit.matId;
            type = (uint)materials[matId].type;
            if (type == MINI_MAT_EMITTING) {
                GpuSpectrum Ls = throughput * miniEmission(materials[matId], swl, spd, r2s);
                if (!(depth == 0 || prevSpecular))
                    Ls *= mis_power2(lastPdf, miniHitLightPdf(c, hit, p, roVertex));
                miniAddRadiance(accum, idx, Ls, r, swl, c);
                break;
            }
        }
        GPU_DEVICE const MiniMaterial& mat = materials[matId];
        GpuWavelengths swlVertex = swl;
        float pdf;
        bool specular, haveShadow;
        MiniShadowRay sr;
        bool alive = miniShadeVertex(type, mat, p, hit.n, hit.uv, c, ro, rd, medium, throughput,
                                     r, swl, rng, pdf, specular, haveShadow, sr);
        if (haveShadow)
            miniResolveShadow(accum, idx, sr, swlVertex, c);
        if (!alive)
            break;
        lastPdf = pdf;
        prevSpecular = specular;
    }
}

// ==========================================================================
// Wavefront mode (M2)
// ==========================================================================

struct WfPath {
    gpu_packed3 origin;     float t;
    gpu_packed3 dir;        uint pixel;
    GpuSpectrum throughput;                    // one float4 spectrum
    gpu_packed3 normal;     uint depth;
    uint matId; float u; float v; uint rng;    // uv of the pending hit; matId is the
                                               // medium index for a scatter vertex
    float lambdaU;          uint wlFlags;      // wavelengths recomputed per stage
    float lastPdf;          int medium;        // BSDF pdf of the last scatter (MIS); current medium
    GpuSpectrum r;                             // spectral-MIS pdf ratio (media)
};
static_assert(sizeof(WfPath) == WF_PATHSTATE_SIZE, "host allocates queues with this stride");

struct WfShadowRay {
    gpu_packed3 origin;     float tMax;
    gpu_packed3 dir;        uint pixel;
    GpuSpectrum L;
    float lambdaU;          uint wlFlags;      int medium;     uint pad0;
};
static_assert(sizeof(WfShadowRay) == WF_SHADOWRAY_SIZE, "host allocates the shadow queue with this stride");

// Rebuild wavelengths from the raygen draw + dispersion flag — deterministic,
// so carrying 8 bytes beats carrying the 32-byte struct.
GPU_FN inline GpuWavelengths wfWavelengths(float lambdaU, uint wlFlags)
{
    GpuWavelengths swl = spd_sample_visible(lambdaU);
    if ((wlFlags & WF_FLAG_SECONDARY_TERMINATED) != 0u)
        spd_terminate_secondary(swl);
    return swl;
}

GPU_FN inline MiniShadeCtx wfCtx(GPU_PARAMS_REF(WfCtl) C,
                                 GPU_DEVICE const MiniMaterial* materials,
                                 GPU_DEVICE const MiniObject* objects,
                                 GPU_DEVICE const RtBvhNode* nodes,
                                 GPU_DEVICE const gpu_uint4* tris,
                                 GPU_DEVICE const gpu_storage3* positions,
                                 GPU_DEVICE const gpu_storage3* normals,
                                 GPU_DEVICE const gpu_float2* uvs,
                                 GPU_DEVICE const RtLight* lights,
                                 GPU_DEVICE const float* envDist,
                                 GPU_DEVICE const RhiTex* texHeap,
                                 GPU_DEVICE const float* spd,
                                 GPU_DEVICE const float* r2s,
                                 GPU_DEVICE const MediumGpu* media)
{
    MiniShadeCtx c;
    c.materials = materials;
    c.objects = objects;
    c.nodes = nodes;
    c.tris = tris;
    c.positions = positions;
    c.normals = normals;
    c.uvs = uvs;
    c.lights = lights;
    c.envDist = envDist;
    c.texHeap = texHeap;
    c.spd = spd;
    c.r2s = r2s;
    c.media = media;
    c.numObjects = C.numObjects;
    c.numNodes = C.bvhNumNodes;
    c.numLights = C.numLights;
    c.numMedia = C.numMedia;
    c.envMapIdx = C.envMapIdx;
    c.envW = C.envW;
    c.envH = C.envH;
    c.filmR0 = gpu_load3(C.filmR0);
    c.filmR1 = gpu_load3(C.filmR1);
    c.filmR2 = gpu_load3(C.filmR2);
    return c;
}

GPU_KERNEL(wf_raygen, GPU_TID_2D)(GPU_KERNEL_PARAMS(MiniParams, P),
    GPU_BUFFER(WfPath, rays),
    GPU_BUFFER(gpu_atomic_uint, counts))
{
    gpu_uint2 gid = GPU_GLOBAL_ID_XY;
    if (gid.x >= P.width || gid.y >= P.height)
        return;
    uint idx = gid.y * P.width + gid.x;

    GpuRng rng;
    rng.state = idx * 9781u + P.iter * 26699u + 1u;
    gpu_rand(rng);
    gpu_rand(rng);

    gpu_float3 ro, rd;
    generateCameraRay(P, gid, rng, ro, rd);
    float lambdaU = gpu_rand(rng);

    WfPath path;
    path.origin = ro;
    path.dir = rd;
    path.throughput = GpuSpectrum(1.0f);
    path.normal = gpu_float3(0.0f);
    path.t = 0.0f;
    path.pixel = idx;
    path.rng = rng.state;
    path.depth = 0;
    path.matId = 0;
    path.u = 0.0f;
    path.v = 0.0f;
    path.lambdaU = lambdaU;
    path.wlFlags = 0;
    path.lastPdf = 0.0f;
    path.medium = P.cameraMedium;
    path.r = GpuSpectrum(1.0f);
    rays[idx] = path;

    if (idx == 0) {
        // Shade/shadow counters are zeroed by wf_prep_intersect each bounce.
        gpu_atomic_store(&counts[WF_COUNT_RAY_A], P.width * P.height);
        gpu_atomic_store(&counts[WF_COUNT_RAY_B], 0u);
    }
}

// Single-thread dispatches turning GPU-written queue counts into indirect
// threadgroup args, keeping the bounce loop free of CPU readbacks.
GPU_KERNEL(wf_prep_intersect, GPU_TID_NONE)(GPU_KERNEL_PARAMS(WfCtl, C),
    GPU_BUFFER(gpu_atomic_uint, counts),
    GPU_BUFFER(uint, args))
{
    uint c = gpu_atomic_load(&counts[C.srcCounter]);
    args[WF_ARG_INTERSECT * 4 + 0] = (c + PRIM_TILE - 1u) / PRIM_TILE;
    args[WF_ARG_INTERSECT * 4 + 1] = 1u;
    args[WF_ARG_INTERSECT * 4 + 2] = 1u;
    gpu_atomic_store(&counts[WF_COUNT_SHADE_DIFFUSE], 0u);
    gpu_atomic_store(&counts[WF_COUNT_SHADE_CONDUCTOR], 0u);
    gpu_atomic_store(&counts[WF_COUNT_SHADE_GLASS], 0u);
    gpu_atomic_store(&counts[WF_COUNT_SHADE_MEDIUM], 0u);
    gpu_atomic_store(&counts[WF_COUNT_SHADOW], 0u);
}

GPU_KERNEL(wf_prep_shade, GPU_TID_NONE)(GPU_KERNEL_PARAMS(WfCtl, C),
    GPU_BUFFER(gpu_atomic_uint, counts),
    GPU_BUFFER(uint, args))
{
    constexpr uint queues[4] = { WF_COUNT_SHADE_DIFFUSE, WF_COUNT_SHADE_CONDUCTOR,
                                 WF_COUNT_SHADE_GLASS, WF_COUNT_SHADE_MEDIUM };
    constexpr uint slots[4] = { WF_ARG_DIFFUSE, WF_ARG_CONDUCTOR, WF_ARG_GLASS, WF_ARG_MEDIUM };
    for (uint i = 0; i < 4; i++) {
        uint c = gpu_atomic_load(&counts[queues[i]]);
        args[slots[i] * 4 + 0] = (c + PRIM_TILE - 1u) / PRIM_TILE;
        args[slots[i] * 4 + 1] = 1u;
        args[slots[i] * 4 + 2] = 1u;
    }
    gpu_atomic_store(&counts[C.zeroCounter], 0u);
}

// After the shade kernels: size the shadow-ray dispatch.
GPU_KERNEL(wf_prep_shadow, GPU_TID_NONE)(GPU_KERNEL_PARAMS(WfCtl, C),
    GPU_BUFFER(gpu_atomic_uint, counts),
    GPU_BUFFER(uint, args))
{
    uint c = gpu_atomic_load(&counts[WF_COUNT_SHADOW]);
    args[WF_ARG_SHADOW * 4 + 0] = (c + PRIM_TILE - 1u) / PRIM_TILE;
    args[WF_ARG_SHADOW * 4 + 1] = 1u;
    args[WF_ARG_SHADOW * 4 + 2] = 1u;
}

// Intersect routes surviving paths into their material type's shade queue
// (tier-1 material dispatch: the queue decides which BSDF code runs, not a
// per-thread branch) — real scatters inside a medium go to the medium queue.
// Emissive and environment hits are resolved here with their MIS weight
// against the light sampler.
GPU_KERNEL(wf_intersect, GPU_TID_1D)(GPU_KERNEL_PARAMS(WfCtl, C),
    GPU_BUFFER(gpu_atomic_uint, counts),
    GPU_BUFFER(const WfPath, raysIn),
    GPU_BUFFER(const MiniObject, objects),
    GPU_BUFFER(const RtBvhNode, bvhNodes),
    GPU_BUFFER(const gpu_uint4, tris),
    GPU_BUFFER(const gpu_storage3, positions),
    GPU_BUFFER(const MiniMaterial, materials),
    GPU_BUFFER(gpu_float4, accum),
    GPU_BUFFER(WfPath, qDiffuse),
    GPU_BUFFER(WfPath, qConductor),
    GPU_BUFFER(WfPath, qGlass),
    GPU_BUFFER(const RhiTex, texHeap),
    GPU_BUFFER(const gpu_storage3, normals),
    GPU_BUFFER(const gpu_float2, uvs),
    GPU_BUFFER(const float, spd),
    GPU_BUFFER(const float, r2s),
    GPU_BUFFER(const float, envDist),
    GPU_BUFFER(const MediumGpu, media),
    GPU_BUFFER(WfPath, qMedium))
{
    uint tid = GPU_GLOBAL_ID_X;
    if (tid >= gpu_atomic_load(&counts[C.srcCounter]))
        return;
    WfPath path = raysIn[tid];
    // Lights are not needed here (pmf is 1/numLights); pass null.
    MiniShadeCtx c = wfCtx(C, materials, objects, bvhNodes, tris, positions, normals, uvs,
                           (GPU_DEVICE const RtLight*)0, envDist, texHeap, spd, r2s, media);
    gpu_float3 roVertex = gpu_float3(path.origin);
    gpu_float3 ro = roVertex;
    gpu_float3 rd = gpu_float3(path.dir);
    GpuSpectrum throughput = path.throughput;
    GpuSpectrum r = path.r;
    int medium = path.medium;
    GpuRng rng;
    rng.state = path.rng;
    GpuWavelengths swl = wfWavelengths(path.lambdaU, path.wlFlags);
    bool weighted = !(path.depth == 0 || (path.wlFlags & WF_FLAG_PREV_SPECULAR) != 0u);
    MiniHit hit;
    int ev = miniTrace(c, ro, rd, medium, throughput, r, swl, rng, hit);
    if (ev == MINI_TRACE_ABSORBED)
        return;
    if (ev == MINI_TRACE_MISS) {
        // Escaped: environment radiance (resolved inline like emissive hits),
        // then simply not re-enqueued.
        if (C.envMapIdx != MINI_ENV_NONE) {
            GpuSpectrum Ls = throughput * miniEnvRadiance(texHeap, C.envMapIdx, rd, swl, spd, r2s);
            if (weighted)
                Ls *= mis_power2(path.lastPdf,
                                 miniLightPmf(c) * env_pdf_dir(envDist, C.envW, C.envH, rd));
            miniAddRadiance(accum, path.pixel, Ls, r, swl, c);
        }
        return;
    }

    // State the trace advanced (origin past interfaces, medium, RNG, spectra).
    path.origin = ro;
    path.t = hit.t;
    path.throughput = throughput;
    path.r = r;
    path.medium = medium;
    path.rng = rng.state;
    if (ev == MINI_TRACE_SCATTER) {
        path.matId = (uint)medium;
        qMedium[prim_queue_alloc(&counts[WF_COUNT_SHADE_MEDIUM])] = path;
        return;
    }

    GPU_DEVICE const MiniMaterial& mat = materials[hit.matId];
    if (mat.type == MINI_MAT_EMITTING) {
        // One path per pixel per sample, so this write does not race.
        GpuSpectrum Ls = throughput * miniEmission(mat, swl, spd, r2s);
        if (weighted)
            Ls *= mis_power2(path.lastPdf, miniHitLightPdf(c, hit, ro + rd * hit.t, roVertex));
        miniAddRadiance(accum, path.pixel, Ls, r, swl, c);
        return;
    }

    path.normal = hit.n;
    path.matId = hit.matId;
    path.u = hit.uv.x;
    path.v = hit.uv.y;
    if (mat.type == MINI_MAT_DIFFUSE)
        qDiffuse[prim_queue_alloc(&counts[WF_COUNT_SHADE_DIFFUSE])] = path;
    else if (mat.type == MINI_MAT_CONDUCTOR)
        qConductor[prim_queue_alloc(&counts[WF_COUNT_SHADE_CONDUCTOR])] = path;
    else
        qGlass[prim_queue_alloc(&counts[WF_COUNT_SHADE_GLASS])] = path;
}

// One shade kernel, specialized per vertex type at pipeline creation
// (rhi::SpecConstant -> function constants / template instantiation): the
// material branches fold away, and the queue guarantees the specialization
// matches every path in it — divergence-free shading with a single source of
// truth. Next-event estimation is sampled here and traced by wf_shadow.
// Guarded so backends without a spec-const lowering still compile the rest of
// this file (gpu_portable.h GPU_HAS_SPEC_CONST).
#if GPU_HAS_SPEC_CONST
GPU_SPEC_CONST(uint, kShadeMatType, 0)

GPU_KERNEL(wf_shade, GPU_TID_1D)(GPU_KERNEL_PARAMS(WfCtl, C),
    GPU_BUFFER(gpu_atomic_uint, counts),
    GPU_BUFFER(const WfPath, queue),
    GPU_BUFFER(WfPath, raysOut),
    GPU_BUFFER(const MiniMaterial, materials),
    GPU_BUFFER(const RhiTex, texHeap),
    GPU_BUFFER(const float, spd),
    GPU_BUFFER(const float, r2s),
    GPU_BUFFER(const RtLight, lights),
    GPU_BUFFER(const MiniObject, objects),
    GPU_BUFFER(const gpu_uint4, tris),
    GPU_BUFFER(const gpu_storage3, positions),
    GPU_BUFFER(const float, envDist),
    GPU_BUFFER(WfShadowRay, shadowQueue),
    GPU_BUFFER(const MediumGpu, media))
{
    uint tid = GPU_GLOBAL_ID_X;
    if (tid >= gpu_atomic_load(&counts[C.srcCounter]))
        return;
    WfPath path = queue[tid];
    MiniShadeCtx c = wfCtx(C, materials, objects, (GPU_DEVICE const RtBvhNode*)0, tris, positions,
                           (GPU_DEVICE const gpu_storage3*)0, (GPU_DEVICE const gpu_float2*)0,
                           lights, envDist, texHeap, spd, r2s, media);
    GpuRng rng;
    rng.state = path.rng;
    gpu_float3 ro = gpu_float3(path.origin);
    gpu_float3 rd = gpu_float3(path.dir);
    GpuSpectrum throughput = path.throughput;
    int medium = path.medium;
    GpuWavelengths swl = wfWavelengths(path.lambdaU, path.wlFlags);
    // A medium vertex carries its medium index in matId, not a material.
    uint matId = kShadeMatType == MINI_MAT_MEDIUM ? 0u : path.matId;
    float pdf;
    bool specular, haveShadow;
    MiniShadowRay sr;
    bool alive = miniShadeVertex(kShadeMatType, materials[matId], ro + rd * path.t,
                                 gpu_float3(path.normal), gpu_float2(path.u, path.v), c,
                                 ro, rd, medium, throughput, path.r, swl, rng, pdf, specular,
                                 haveShadow, sr);
    if (haveShadow) {
        WfShadowRay s;
        s.origin = sr.origin;
        s.tMax = sr.tMax;
        s.dir = sr.dir;
        s.pixel = path.pixel;
        s.L = sr.L;
        s.lambdaU = path.lambdaU;
        s.wlFlags = path.wlFlags;   // the vertex's wavelengths (pre-scatter)
        s.medium = sr.medium;
        s.pad0 = 0;
        shadowQueue[prim_queue_alloc(&counts[WF_COUNT_SHADOW])] = s;
    }
    if (!alive || path.depth + 1 >= C.maxDepth)
        return;
    path.origin = ro;
    path.dir = rd;
    path.throughput = throughput;
    path.medium = medium;
    path.rng = rng.state;
    path.lastPdf = pdf;
    path.wlFlags &= ~WF_FLAG_PREV_SPECULAR;
    if (specular)
        path.wlFlags |= WF_FLAG_PREV_SPECULAR;
    if (spd_secondary_terminated(swl))
        path.wlFlags |= WF_FLAG_SECONDARY_TERMINATED;
    path.depth++;
    raysOut[prim_queue_alloc(&counts[C.dstCounter])] = path;
}
GPU_SPEC_INSTANCES(wf_shade, 0, MINI_MAT_DIFFUSE, MINI_MAT_CONDUCTOR, MINI_MAT_GLASS, MINI_MAT_MEDIUM)
#endif  // GPU_HAS_SPEC_CONST

// Shadow rays: transmittance to the light (visibility in media-free scenes),
// then the precomputed contribution lands.
GPU_KERNEL(wf_shadow, GPU_TID_1D)(GPU_KERNEL_PARAMS(WfCtl, C),
    GPU_BUFFER(gpu_atomic_uint, counts),
    GPU_BUFFER(const WfShadowRay, shadowQueue),
    GPU_BUFFER(const MiniObject, objects),
    GPU_BUFFER(const RtBvhNode, bvhNodes),
    GPU_BUFFER(const gpu_uint4, tris),
    GPU_BUFFER(const gpu_storage3, positions),
    GPU_BUFFER(gpu_float4, accum),
    GPU_BUFFER(const MiniMaterial, materials),
    GPU_BUFFER(const gpu_storage3, normals),
    GPU_BUFFER(const gpu_float2, uvs),
    GPU_BUFFER(const float, spd),
    GPU_BUFFER(const MediumGpu, media))
{
    uint tid = GPU_GLOBAL_ID_X;
    if (tid >= gpu_atomic_load(&counts[WF_COUNT_SHADOW]))
        return;
    WfShadowRay s = shadowQueue[tid];
    MiniShadeCtx c = wfCtx(C, materials, objects, bvhNodes, tris, positions, normals, uvs,
                           (GPU_DEVICE const RtLight*)0, (GPU_DEVICE const float*)0,
                           (GPU_DEVICE const RhiTex*)0, spd, (GPU_DEVICE const float*)0, media);
    MiniShadowRay sr;
    sr.origin = gpu_float3(s.origin);
    sr.dir = gpu_float3(s.dir);
    sr.tMax = s.tMax;
    sr.L = s.L;
    sr.medium = s.medium;
    miniResolveShadow(accum, s.pixel, sr, wfWavelengths(s.lambdaU, s.wlFlags), c);
}

// ==========================================================================
// Preview
// ==========================================================================

// Tonemaps the accumulator into the RHI present target (RGBA8) each iteration.
// P.iter carries the number of completed samples. Mirrors x like saveImage()
// (the quirk all saved renders share), so the window shows exactly what the
// PNG will contain.
GPU_KERNEL(present_tonemap, GPU_TID_2D)(GPU_KERNEL_PARAMS(MiniParams, P),
    GPU_BUFFER(const gpu_float4, accum),
    GPU_BUFFER(gpu_uchar4, out))
{
    gpu_uint2 gid = GPU_GLOBAL_ID_XY;
    if (gid.x >= P.width || gid.y >= P.height)
        return;
    gpu_float3 c = tonemap_aces(gpu_xyz(accum[gid.y * P.width + gid.x]) / (float)P.iter);
    out[gid.y * P.width + (P.width - 1u - gid.x)] =
        gpu_make_uchar4((uchar)(c.x * 255.0f + 0.5f), (uchar)(c.y * 255.0f + 0.5f),
                        (uchar)(c.z * 255.0f + 0.5f), 255);
}

#endif // MINI_PATHTRACE_GPU_H
