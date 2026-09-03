#ifndef MINI_PATHTRACE_GPU_H
#define MINI_PATHTRACE_GPU_H
// FluoraMini path tracing kernels (docs/metal-rhi-design.md M1+M2, NEE/MIS in
// M4 part 2). Two execution modes share every helper below, so their images
// are bitwise identical:
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
#include "../core/envmap_shared.h"
#include "../core/light_shared.h"
#include "../core/tonemap_shared.h"
#include "mini_shared.h"
#endif

#define MINI_SCATTER_OFFSET 1e-4f   // origin push-off along the facing normal
#define MINI_SHADOW_TMAX_EPS 1e-3f  // shadow rays stop this short of the light (world units)

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

// Everything the shading helpers need to reach: scene arrays + per-dispatch
// scalars. Built by each kernel from its bound buffers and parameter block so
// the helpers have one signature in both modes.
struct MiniShadeCtx {
    GPU_DEVICE const MiniMaterial* materials;
    GPU_DEVICE const MiniObject* objects;
    GPU_DEVICE const gpu_uint4* tris;
    GPU_DEVICE const gpu_storage3* positions;
    GPU_DEVICE const RtLight* lights;
    GPU_DEVICE const float* envDist;
    GPU_DEVICE const RhiTex* texHeap;
    GPU_DEVICE const float* spd;
    GPU_DEVICE const float* r2s;
    uint numLights;
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
// point `hit` reached from `ro` -- the MIS weight input for BSDF-sampled
// emissive hits. Uses the primitive's geometric normal like light sampling.
GPU_FN inline float miniHitLightPdf(GPU_THREAD const MiniShadeCtx& c, GPU_THREAD const MiniHit& hit,
                                    gpu_float3 ro, gpu_float3 rd)
{
    gpu_float3 pLight = ro + rd * hit.t;
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
    return miniLightPmf(c) * light_pdf_area(pdfArea, pLight, nLight, ro);
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

// A next-event-estimation candidate: the shadow ray plus the film RGB it adds
// if unoccluded (already MIS-weighted and divided by the light pdf).
struct MiniShadowRay {
    gpu_float3 origin;
    gpu_float3 dir;
    float tMax;
    gpu_float3 rgb;
};

// Samples one light for the vertex (p, nF) and evaluates the BSDF toward it
// (sample_Ld port). Consumes exactly four RNG draws. Returns false when
// nothing can be contributed (no lights, back-facing, zero BSDF/pdf).
GPU_FN inline bool miniSampleDirect(uint type, GPU_DEVICE const MiniMaterial& mat,
                                    gpu_float3 p, gpu_float3 nF, gpu_float3 rd, gpu_float2 uv,
                                    GPU_THREAD const MiniShadeCtx& c,
                                    GpuSpectrum throughput, GpuWavelengths swl,
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
    if (dot(wi, nF) <= 0.0f)   // reflection-only BSDFs see nothing from below
        return false;

    GpuSpectrum reflectance = GpuSpectrum(0.0f), eta = GpuSpectrum(1.0f), k = GpuSpectrum(0.0f);
    miniMaterialSpectra(type, mat, uv, c, swl, reflectance, eta, k);
    GpuSpectrum f;
    float pdfB;
    bool ok;
    if (type == MINI_MAT_DIFFUSE)
        ok = bsdf_eval_lambert(reflectance, nF, wi, f, pdfB);
    else if (type == MINI_MAT_CONDUCTOR)
        ok = bsdf_eval_conductor(eta, k, mat.roughness, nF, rd, wi, f, pdfB);
    else
        ok = false;   // glass is specular: never reaches here
    if (!ok)
        return false;
    float w = mis_power2(pdfL, pdfB);
    GpuSpectrum Ls = Le * f * throughput * (w / pdfL);
    gpu_float3 rgb = miniFilmRgb(Ls, swl, c.spd, c.filmR0, c.filmR1, c.filmR2);
    if (!gpu_all_finite(rgb))
        return false;
    sr.origin = p + nF * MINI_SCATTER_OFFSET;
    sr.dir = wi;
    sr.tMax = tMax;
    sr.rgb = rgb;
    return true;
}

// Samples the BSDF and updates the path (get_bxdf + sample_f port). Shared by
// the megakernel switch and the specialized shade kernel, so both modes
// consume identical RNG streams and math. `type` is mat.type in the
// megakernel (dynamic branch) and a function constant in wf_shade (branch
// folds at pipeline creation). Returns false when the sample is absorbed.
GPU_FN inline bool miniScatter(uint type, GPU_DEVICE const MiniMaterial& mat,
                        gpu_float3 p, gpu_float3 n, gpu_float2 uv,
                        GPU_THREAD const MiniShadeCtx& c,
                        GPU_THREAD gpu_float3& ro, GPU_THREAD gpu_float3& rd,
                        GPU_THREAD GpuSpectrum& throughput,
                        GPU_THREAD GpuWavelengths& swl, GPU_THREAD GpuRng& rng,
                        GPU_THREAD float& pdf)
{
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
    return true;
}

// One non-emissive path vertex: next-event estimation (unless the BSDF is a
// delta) then BSDF sampling. The order of RNG draws is fixed here for both
// modes. `haveShadow` reports a shadow ray to trace; the return value says
// whether the path continues.
GPU_FN inline bool miniShadeVertex(uint type, GPU_DEVICE const MiniMaterial& mat,
                                   gpu_float3 p, gpu_float3 n, gpu_float2 uv,
                                   GPU_THREAD const MiniShadeCtx& c,
                                   GPU_THREAD gpu_float3& ro, GPU_THREAD gpu_float3& rd,
                                   GPU_THREAD GpuSpectrum& throughput,
                                   GPU_THREAD GpuWavelengths& swl, GPU_THREAD GpuRng& rng,
                                   GPU_THREAD float& pdf, GPU_THREAD bool& specular,
                                   GPU_THREAD bool& haveShadow, GPU_THREAD MiniShadowRay& sr)
{
    specular = miniMaterialSpecular(type, mat);
    haveShadow = false;
    if (!specular) {
        gpu_float3 nF = dot(n, rd) < 0.0f ? n : -n;
        haveShadow = miniSampleDirect(type, mat, p, nF, rd, uv, c, throughput, swl, rng, sr);
    }
    return miniScatter(type, mat, p, n, uv, c, ro, rd, throughput, swl, rng, pdf);
}

// ==========================================================================
// Megakernel mode (M1)
// ==========================================================================

GPU_KERNEL(pathtraceKernel)(GPU_KERNEL_PARAMS(MiniParams, P)
    GPU_BUFFER(gpu_float4, accum, 1)
    GPU_BUFFER(const MiniMaterial, materials, 2)
    GPU_BUFFER(const MiniObject, objects, 3)
    GPU_BUFFER(const RtBvhNode, bvhNodes, 4)
    GPU_BUFFER(const gpu_uint4, tris, 5)
    GPU_BUFFER(const gpu_storage3, positions, 6)
    GPU_BUFFER(const RhiTex, texHeap, 7)
    GPU_BUFFER(const gpu_storage3, normals, 8)
    GPU_BUFFER(const gpu_float2, uvs, 9)
    GPU_BUFFER(const float, spd, 10)
    GPU_BUFFER(const float, r2s, 11)
    GPU_BUFFER(const RtLight, lights, 12)
    GPU_BUFFER(const float, envDist, 13)
    GPU_TID_2D)
{
    gpu_uint2 gid = GPU_GLOBAL_ID_XY;
    if (gid.x >= P.width || gid.y >= P.height)
        return;
    uint idx = gid.y * P.width + gid.x;

    MiniShadeCtx c;
    c.materials = materials;
    c.objects = objects;
    c.tris = tris;
    c.positions = positions;
    c.lights = lights;
    c.envDist = envDist;
    c.texHeap = texHeap;
    c.spd = spd;
    c.r2s = r2s;
    c.numLights = P.numLights;
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
    float lastPdf = 0.0f;
    bool prevSpecular = false;

    for (uint depth = 0; depth < P.maxDepth; depth++) {
        MiniHit hit;
        if (!sceneIntersect(ro, rd, objects, P.numObjects, bvhNodes, P.bvhNumNodes,
                            tris, positions, normals, uvs, hit)) {
            // Escaped: environment radiance if the scene has a SKYBOX, else black.
            if (P.envMapIdx != MINI_ENV_NONE) {
                GpuSpectrum Ls = throughput * miniEnvRadiance(texHeap, P.envMapIdx, rd, swl, spd, r2s);
                if (!(depth == 0 || prevSpecular))
                    Ls *= mis_power2(lastPdf, miniLightPmf(c) * env_pdf_dir(envDist, P.envW, P.envH, rd));
                gpu_float3 rgb = miniFilmRgb(Ls, swl, spd, c.filmR0, c.filmR1, c.filmR2);
                if (gpu_all_finite(rgb))
                    accum[idx] += gpu_float4(rgb, 0.0f);
            }
            break;
        }
        GPU_DEVICE const MiniMaterial& mat = materials[hit.matId];
        if (mat.type == MINI_MAT_EMITTING) {
            GpuSpectrum Ls = throughput * miniEmission(mat, swl, spd, r2s);
            if (!(depth == 0 || prevSpecular))
                Ls *= mis_power2(lastPdf, miniHitLightPdf(c, hit, ro, rd));
            gpu_float3 rgb = miniFilmRgb(Ls, swl, spd, c.filmR0, c.filmR1, c.filmR2);
            if (gpu_all_finite(rgb))
                accum[idx] += gpu_float4(rgb, 0.0f);
            break;
        }
        float pdf;
        bool specular, haveShadow;
        MiniShadowRay sr;
        bool alive = miniShadeVertex(mat.type, mat, ro + rd * hit.t, hit.n, hit.uv, c,
                                     ro, rd, throughput, swl, rng, pdf, specular, haveShadow, sr);
        if (haveShadow
            && !sceneOccluded(sr.origin, sr.dir, sr.tMax, objects, P.numObjects,
                              bvhNodes, P.bvhNumNodes, tris, positions))
            accum[idx] += gpu_float4(sr.rgb, 0.0f);
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
    uint matId; float u; float v; uint rng;    // uv of the pending hit
    float lambdaU;          uint wlFlags;      // wavelengths recomputed per stage
    float lastPdf;          uint pad1;         // BSDF pdf of the last scatter (MIS)
};
static_assert(sizeof(WfPath) == WF_PATHSTATE_SIZE, "host allocates queues with this stride");

struct WfShadowRay {
    gpu_packed3 origin;     float tMax;
    gpu_packed3 dir;        uint pixel;
    gpu_packed3 rgb;        uint pad0;
};
static_assert(sizeof(WfShadowRay) == WF_SHADOWRAY_SIZE, "host allocates the shadow queue with this stride");

// Rebuild this path's wavelengths from its raygen draw + dispersion flag —
// deterministic, so carrying 8 bytes beats carrying the 32-byte struct.
GPU_FN inline GpuWavelengths wfWavelengths(WfPath path)
{
    GpuWavelengths swl = spd_sample_visible(path.lambdaU);
    if ((path.wlFlags & WF_FLAG_SECONDARY_TERMINATED) != 0u)
        spd_terminate_secondary(swl);
    return swl;
}

GPU_FN inline MiniShadeCtx wfCtx(GPU_PARAMS_REF(WfCtl) C,
                                 GPU_DEVICE const MiniMaterial* materials,
                                 GPU_DEVICE const MiniObject* objects,
                                 GPU_DEVICE const gpu_uint4* tris,
                                 GPU_DEVICE const gpu_storage3* positions,
                                 GPU_DEVICE const RtLight* lights,
                                 GPU_DEVICE const float* envDist,
                                 GPU_DEVICE const RhiTex* texHeap,
                                 GPU_DEVICE const float* spd,
                                 GPU_DEVICE const float* r2s)
{
    MiniShadeCtx c;
    c.materials = materials;
    c.objects = objects;
    c.tris = tris;
    c.positions = positions;
    c.lights = lights;
    c.envDist = envDist;
    c.texHeap = texHeap;
    c.spd = spd;
    c.r2s = r2s;
    c.numLights = C.numLights;
    c.envMapIdx = C.envMapIdx;
    c.envW = C.envW;
    c.envH = C.envH;
    c.filmR0 = gpu_load3(C.filmR0);
    c.filmR1 = gpu_load3(C.filmR1);
    c.filmR2 = gpu_load3(C.filmR2);
    return c;
}

GPU_KERNEL(wf_raygen)(GPU_KERNEL_PARAMS(MiniParams, P)
    GPU_BUFFER(WfPath, rays, 1)
    GPU_BUFFER(gpu_atomic_uint, counts, 2)
    GPU_TID_2D)
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
    path.pad1 = 0;
    rays[idx] = path;

    if (idx == 0) {
        // Shade/shadow counters are zeroed by wf_prep_intersect each bounce.
        gpu_atomic_store(&counts[WF_COUNT_RAY_A], P.width * P.height);
        gpu_atomic_store(&counts[WF_COUNT_RAY_B], 0u);
    }
}

// Single-thread dispatches turning GPU-written queue counts into indirect
// threadgroup args, keeping the bounce loop free of CPU readbacks.
GPU_KERNEL(wf_prep_intersect)(GPU_KERNEL_PARAMS(WfCtl, C)
    GPU_BUFFER(gpu_atomic_uint, counts, 1)
    GPU_BUFFER(uint, args, 2))
{
    uint c = gpu_atomic_load(&counts[C.srcCounter]);
    args[WF_ARG_INTERSECT * 4 + 0] = (c + PRIM_TILE - 1u) / PRIM_TILE;
    args[WF_ARG_INTERSECT * 4 + 1] = 1u;
    args[WF_ARG_INTERSECT * 4 + 2] = 1u;
    gpu_atomic_store(&counts[WF_COUNT_SHADE_DIFFUSE], 0u);
    gpu_atomic_store(&counts[WF_COUNT_SHADE_CONDUCTOR], 0u);
    gpu_atomic_store(&counts[WF_COUNT_SHADE_GLASS], 0u);
    gpu_atomic_store(&counts[WF_COUNT_SHADOW], 0u);
}

GPU_KERNEL(wf_prep_shade)(GPU_KERNEL_PARAMS(WfCtl, C)
    GPU_BUFFER(gpu_atomic_uint, counts, 1)
    GPU_BUFFER(uint, args, 2))
{
    constexpr uint queues[3] = { WF_COUNT_SHADE_DIFFUSE, WF_COUNT_SHADE_CONDUCTOR,
                                 WF_COUNT_SHADE_GLASS };
    constexpr uint slots[3] = { WF_ARG_DIFFUSE, WF_ARG_CONDUCTOR, WF_ARG_GLASS };
    for (uint i = 0; i < 3; i++) {
        uint c = gpu_atomic_load(&counts[queues[i]]);
        args[slots[i] * 4 + 0] = (c + PRIM_TILE - 1u) / PRIM_TILE;
        args[slots[i] * 4 + 1] = 1u;
        args[slots[i] * 4 + 2] = 1u;
    }
    gpu_atomic_store(&counts[C.zeroCounter], 0u);
}

// After the shade kernels: size the shadow-ray dispatch.
GPU_KERNEL(wf_prep_shadow)(GPU_KERNEL_PARAMS(WfCtl, C)
    GPU_BUFFER(gpu_atomic_uint, counts, 1)
    GPU_BUFFER(uint, args, 2))
{
    uint c = gpu_atomic_load(&counts[WF_COUNT_SHADOW]);
    args[WF_ARG_SHADOW * 4 + 0] = (c + PRIM_TILE - 1u) / PRIM_TILE;
    args[WF_ARG_SHADOW * 4 + 1] = 1u;
    args[WF_ARG_SHADOW * 4 + 2] = 1u;
}

// Intersect routes surviving paths into their material type's shade queue
// (tier-1 material dispatch: the queue decides which BSDF code runs, not a
// per-thread branch). Emissive and environment hits are resolved here with
// their MIS weight against the light sampler.
GPU_KERNEL(wf_intersect)(GPU_KERNEL_PARAMS(WfCtl, C)
    GPU_BUFFER(gpu_atomic_uint, counts, 1)
    GPU_BUFFER(const WfPath, raysIn, 2)
    GPU_BUFFER(const MiniObject, objects, 3)
    GPU_BUFFER(const RtBvhNode, bvhNodes, 4)
    GPU_BUFFER(const gpu_uint4, tris, 5)
    GPU_BUFFER(const gpu_storage3, positions, 6)
    GPU_BUFFER(const MiniMaterial, materials, 7)
    GPU_BUFFER(gpu_float4, accum, 8)
    GPU_BUFFER(WfPath, qDiffuse, 9)
    GPU_BUFFER(WfPath, qConductor, 10)
    GPU_BUFFER(WfPath, qGlass, 11)
    GPU_BUFFER(const RhiTex, texHeap, 12)
    GPU_BUFFER(const gpu_storage3, normals, 13)
    GPU_BUFFER(const gpu_float2, uvs, 14)
    GPU_BUFFER(const float, spd, 15)
    GPU_BUFFER(const float, r2s, 16)
    GPU_BUFFER(const float, envDist, 17)
    GPU_TID_1D)
{
    uint tid = GPU_GLOBAL_ID_X;
    if (tid >= gpu_atomic_load(&counts[C.srcCounter]))
        return;
    WfPath path = raysIn[tid];
    // Lights are not needed here (pmf is 1/numLights); pass null.
    MiniShadeCtx c = wfCtx(C, materials, objects, tris, positions, (GPU_DEVICE const RtLight*)0,
                           envDist, texHeap, spd, r2s);
    gpu_float3 ro = gpu_float3(path.origin);
    gpu_float3 rd = gpu_float3(path.dir);
    bool weighted = !(path.depth == 0 || (path.wlFlags & WF_FLAG_PREV_SPECULAR) != 0u);
    MiniHit hit;
    if (!sceneIntersect(ro, rd, objects, C.numObjects, bvhNodes, C.bvhNumNodes, tris,
                        positions, normals, uvs, hit)) {
        // Escaped: environment radiance (resolved inline like emissive hits),
        // then simply not re-enqueued.
        if (C.envMapIdx != MINI_ENV_NONE) {
            GpuWavelengths swl = wfWavelengths(path);
            GpuSpectrum Ls = path.throughput
                           * miniEnvRadiance(texHeap, C.envMapIdx, rd, swl, spd, r2s);
            if (weighted)
                Ls *= mis_power2(path.lastPdf,
                                 miniLightPmf(c) * env_pdf_dir(envDist, C.envW, C.envH, rd));
            gpu_float3 rgb = miniFilmRgb(Ls, swl, spd, c.filmR0, c.filmR1, c.filmR2);
            if (gpu_all_finite(rgb))
                accum[path.pixel] += gpu_float4(rgb, 0.0f);
        }
        return;
    }

    GPU_DEVICE const MiniMaterial& mat = materials[hit.matId];
    if (mat.type == MINI_MAT_EMITTING) {
        // One path per pixel per sample, so this write does not race.
        GpuWavelengths swl = wfWavelengths(path);
        GpuSpectrum Ls = path.throughput * miniEmission(mat, swl, spd, r2s);
        if (weighted)
            Ls *= mis_power2(path.lastPdf, miniHitLightPdf(c, hit, ro, rd));
        gpu_float3 rgb = miniFilmRgb(Ls, swl, spd, c.filmR0, c.filmR1, c.filmR2);
        if (gpu_all_finite(rgb))
            accum[path.pixel] += gpu_float4(rgb, 0.0f);
        return;
    }

    path.t = hit.t;
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

// One shade kernel, specialized per material type at pipeline creation
// (rhi::SpecConstant -> function constants / template instantiation): the
// material branches fold away, and the queue guarantees the specialization
// matches every path in it — divergence-free shading with a single source of
// truth. Next-event estimation is sampled here and traced by wf_shadow.
// Guarded so backends without a spec-const lowering still compile the rest of
// this file (gpu_portable.h GPU_HAS_SPEC_CONST).
#if GPU_HAS_SPEC_CONST
GPU_SPEC_CONST(uint, kShadeMatType, 0)

GPU_KERNEL(wf_shade)(GPU_KERNEL_PARAMS(WfCtl, C)
    GPU_BUFFER(gpu_atomic_uint, counts, 1)
    GPU_BUFFER(const WfPath, queue, 2)
    GPU_BUFFER(WfPath, raysOut, 3)
    GPU_BUFFER(const MiniMaterial, materials, 4)
    GPU_BUFFER(const RhiTex, texHeap, 5)
    GPU_BUFFER(const float, spd, 6)
    GPU_BUFFER(const float, r2s, 7)
    GPU_BUFFER(const RtLight, lights, 8)
    GPU_BUFFER(const MiniObject, objects, 9)
    GPU_BUFFER(const gpu_uint4, tris, 10)
    GPU_BUFFER(const gpu_storage3, positions, 11)
    GPU_BUFFER(const float, envDist, 12)
    GPU_BUFFER(WfShadowRay, shadowQueue, 13)
    GPU_TID_1D)
{
    uint tid = GPU_GLOBAL_ID_X;
    if (tid >= gpu_atomic_load(&counts[C.srcCounter]))
        return;
    WfPath path = queue[tid];
    MiniShadeCtx c = wfCtx(C, materials, objects, tris, positions, lights, envDist,
                           texHeap, spd, r2s);
    GpuRng rng;
    rng.state = path.rng;
    gpu_float3 ro = gpu_float3(path.origin);
    gpu_float3 rd = gpu_float3(path.dir);
    GpuSpectrum throughput = path.throughput;
    GpuWavelengths swl = wfWavelengths(path);
    float pdf;
    bool specular, haveShadow;
    MiniShadowRay sr;
    bool alive = miniShadeVertex(kShadeMatType, materials[path.matId], ro + rd * path.t,
                                 gpu_float3(path.normal), gpu_float2(path.u, path.v), c,
                                 ro, rd, throughput, swl, rng, pdf, specular, haveShadow, sr);
    if (haveShadow) {
        WfShadowRay s;
        s.origin = sr.origin;
        s.tMax = sr.tMax;
        s.dir = sr.dir;
        s.pixel = path.pixel;
        s.rgb = sr.rgb;
        s.pad0 = 0;
        shadowQueue[prim_queue_alloc(&counts[WF_COUNT_SHADOW])] = s;
    }
    if (!alive || path.depth + 1 >= C.maxDepth)
        return;
    path.origin = ro;
    path.dir = rd;
    path.throughput = throughput;
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
GPU_SPEC_INSTANCES(wf_shade, 0, MINI_MAT_DIFFUSE, MINI_MAT_CONDUCTOR, MINI_MAT_GLASS)
#endif  // GPU_HAS_SPEC_CONST

// Shadow rays: visibility test, then the precomputed contribution lands.
GPU_KERNEL(wf_shadow)(GPU_KERNEL_PARAMS(WfCtl, C)
    GPU_BUFFER(gpu_atomic_uint, counts, 1)
    GPU_BUFFER(const WfShadowRay, shadowQueue, 2)
    GPU_BUFFER(const MiniObject, objects, 3)
    GPU_BUFFER(const RtBvhNode, bvhNodes, 4)
    GPU_BUFFER(const gpu_uint4, tris, 5)
    GPU_BUFFER(const gpu_storage3, positions, 6)
    GPU_BUFFER(gpu_float4, accum, 7)
    GPU_TID_1D)
{
    uint tid = GPU_GLOBAL_ID_X;
    if (tid >= gpu_atomic_load(&counts[WF_COUNT_SHADOW]))
        return;
    WfShadowRay s = shadowQueue[tid];
    if (!sceneOccluded(gpu_float3(s.origin), gpu_float3(s.dir), s.tMax, objects, C.numObjects,
                       bvhNodes, C.bvhNumNodes, tris, positions))
        accum[s.pixel] += gpu_float4(gpu_float3(s.rgb), 0.0f);
}

// ==========================================================================
// Preview
// ==========================================================================

// Tonemaps the accumulator into the RHI present target (RGBA8) each iteration.
// P.iter carries the number of completed samples. Mirrors x like saveImage()
// (the quirk all saved renders share), so the window shows exactly what the
// PNG will contain.
GPU_KERNEL(present_tonemap)(GPU_KERNEL_PARAMS(MiniParams, P)
    GPU_BUFFER(const gpu_float4, accum, 1)
    GPU_BUFFER(gpu_uchar4, out, 2)
    GPU_TID_2D)
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
