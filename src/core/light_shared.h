#ifndef CORE_LIGHT_SHARED_H
#define CORE_LIGHT_SHARED_H
// Portable light sampling for next-event estimation (design doc M4 part 2):
// ports of the CUDA renderer's DiffuseAreaLight / ImageInfiniteLight /
// UniformLightSampler / Distribution2D onto flat buffers (invariant I-1 —
// indices and offsets, no pointers). Single-source via the gpu_portable shim.
//
// A light is one RtLight record: an emissive analytic object (cube/sphere,
// index into the object array), an emissive triangle (index into the
// reordered triangle array), or the environment map. Lights are picked
// uniformly (pmf = 1/numLights). Emission itself is evaluated by the caller
// (it knows the material/env tables); this header only produces positions,
// directions and pdfs.
//
// Area lights are one-sided: sampling returns pdf 0 from the back, and
// light_pdf_area() returns 0 for back-side hits, so back-side emission only
// arrives through BSDF sampling with full weight (unbiased, matches the CUDA
// renderer). The env-map pdf uses the equirect Jacobian p(uv)/(2 pi^2 cos(lat)).

#ifndef __METAL_VERSION__
#include "../rhi/gpu_portable.h"
#include "envmap_shared.h"
#endif

#define RT_LIGHT_CUBE   0u
#define RT_LIGHT_SPHERE 1u
#define RT_LIGHT_TRI    2u
#define RT_LIGHT_ENV    3u

struct RtLight {
    unsigned int type;    // RT_LIGHT_*
    unsigned int index;   // object index (cube/sphere) or triangle index (tri); unused for env
    unsigned int pad0, pad1;
};

// Env-map 2D distribution (core/lights.cpp buildEnvDistribution) as one float
// buffer, nu x nv texels:
//   [0, nu*nv)                        conditional func rows (v-major)
//   then nv*(nu+1)                    conditional cdf rows
//   then nv                           conditional funcInt per row
//   then nv                           marginal func
//   then nv+1                         marginal cdf
//   then 1                            marginal funcInt
#define ENVDIST_OFF_COND_FUNC(nu, nv)  0u
#define ENVDIST_OFF_COND_CDF(nu, nv)   ((nu) * (nv))
#define ENVDIST_OFF_COND_INT(nu, nv)   ((nu) * (nv) + (nv) * ((nu) + 1u))
#define ENVDIST_OFF_MARG_FUNC(nu, nv)  ((nu) * (nv) + (nv) * ((nu) + 1u) + (nv))
#define ENVDIST_OFF_MARG_CDF(nu, nv)   ((nu) * (nv) + (nv) * ((nu) + 1u) + 2u * (nv))
#define ENVDIST_OFF_MARG_INT(nu, nv)   ((nu) * (nv) + (nv) * ((nu) + 1u) + 3u * (nv) + 1u)
#define ENVDIST_FLOATS(nu, nv)         ((nu) * (nv) + (nv) * ((nu) + 1u) + 3u * (nv) + 2u)

// Largest i in [0, n-1] with cdf[i] <= u (cdf has n+1 entries; PBRT FindInterval).
GPU_FN inline uint dist1d_find_interval(GPU_DEVICE const float* cdf, uint n, float u)
{
    int size = (int)n - 1, first = 1;
    while (size > 0) {
        int hlf = size >> 1, mid = first + hlf;
        bool pred = cdf[mid] <= u;
        first = pred ? mid + 1 : first;
        size = pred ? size - hlf - 1 : hlf;
    }
    int r = first - 1;
    return (uint)(r < 0 ? 0 : (r > (int)n - 1 ? (int)n - 1 : r));
}

// Distribution1D::sample_continuous over func[n], cdf[n+1], funcInt.
GPU_FN inline float dist1d_sample(GPU_DEVICE const float* func, GPU_DEVICE const float* cdf,
                                  float funcInt, uint n, float u,
                                  GPU_THREAD float& pdf, GPU_THREAD uint& offset)
{
    uint off = dist1d_find_interval(cdf, n, u);
    offset = off;
    float du = u - cdf[off];
    float w = cdf[off + 1] - cdf[off];
    if (w > 0.0f)
        du /= w;
    pdf = funcInt > 0.0f ? func[off] / funcInt : 0.0f;
    return ((float)off + du) / (float)n;
}

// Distribution2D::sample_continuous: uv in [0,1)^2 with pdf w.r.t. uv area.
GPU_FN inline gpu_float2 envdist_sample(GPU_DEVICE const float* d, uint nu, uint nv,
                                        gpu_float2 u, GPU_THREAD float& pdf)
{
    float pdfV, pdfU;
    uint v, dummy;
    float dv = dist1d_sample(d + ENVDIST_OFF_MARG_FUNC(nu, nv), d + ENVDIST_OFF_MARG_CDF(nu, nv),
                             d[ENVDIST_OFF_MARG_INT(nu, nv)], nv, u.y, pdfV, v);
    float du = dist1d_sample(d + ENVDIST_OFF_COND_FUNC(nu, nv) + v * nu,
                             d + ENVDIST_OFF_COND_CDF(nu, nv) + v * (nu + 1u),
                             d[ENVDIST_OFF_COND_INT(nu, nv) + v], nu, u.x, pdfU, dummy);
    pdf = pdfU * pdfV;
    return gpu_float2(du, dv);
}

GPU_FN inline float envdist_pdf(GPU_DEVICE const float* d, uint nu, uint nv, gpu_float2 uv)
{
    int iu = (int)(uv.x * (float)nu), iv = (int)(uv.y * (float)nv);
    iu = iu < 0 ? 0 : (iu > (int)nu - 1 ? (int)nu - 1 : iu);
    iv = iv < 0 ? 0 : (iv > (int)nv - 1 ? (int)nv - 1 : iv);
    float margInt = d[ENVDIST_OFF_MARG_INT(nu, nv)];
    return margInt > 0.0f ? d[ENVDIST_OFF_COND_FUNC(nu, nv) + (uint)iv * nu + (uint)iu] / margInt
                          : 0.0f;
}

// Inverse of env_equirect_uv (same approximate constants, so the pair
// round-trips): u -> longitude about +Y from +X toward +Z, v -> latitude.
GPU_FN inline gpu_float3 env_equirect_dir(gpu_float2 uv)
{
    float theta = (uv.x - 0.5f) / 0.1591f;
    float lat = (uv.y - 0.5f) / 0.3183f;
    float cl = cos(lat);
    return gpu_float3(cl * cos(theta), sin(lat), cl * sin(theta));
}

// Solid-angle pdf of an env direction: p(uv) / (2 pi^2 cos(latitude)).
GPU_FN inline float env_pdf_dir(GPU_DEVICE const float* d, uint nu, uint nv, gpu_float3 dir)
{
    float cl = sqrt(max(1.0f - dir.y * dir.y, 0.0f));
    if (cl < 1e-6f)
        return 0.0f;
    return envdist_pdf(d, nu, nv, env_equirect_uv(dir)) / (2.0f * GPU_PI * GPU_PI * cl);
}

// Uniform sampling of an env direction from the luminance distribution.
GPU_FN inline bool env_sample_dir(GPU_DEVICE const float* d, uint nu, uint nv, gpu_float2 u,
                                  GPU_THREAD gpu_float3& wi, GPU_THREAD float& pdf)
{
    float mapPdf;
    gpu_float2 uv = envdist_sample(d, nu, nv, u, mapPdf);
    if (mapPdf == 0.0f)
        return false;
    wi = env_equirect_dir(uv);
    float cl = sqrt(max(1.0f - wi.y * wi.y, 0.0f));
    if (cl < 1e-6f)
        return false;
    pdf = mapPdf / (2.0f * GPU_PI * GPU_PI * cl);
    return true;
}

// ---------------------------------------------------------------------------
// Area lights over the analytic unit primitives (transform = object-to-world,
// column-major, same convention as MiniObject) and world-space triangles.
// ---------------------------------------------------------------------------

struct LightAreaSample {
    gpu_float3 p;     // point on the light
    gpu_float3 n;     // unit outward normal at p
    float pdfArea;    // 1 / area
};

GPU_FN inline gpu_float3 light_xform_point(gpu_storage4x4 m, gpu_float3 p)
{
    return gpu_xyz(m * gpu_float4(p, 1.0f));
}
GPU_FN inline gpu_float3 light_xform_vec(gpu_storage4x4 m, gpu_float3 v)
{
    return gpu_xyz(m * gpu_float4(v, 0.0f));
}

GPU_FN inline float light_sphere_radius(gpu_storage4x4 m)
{
    return length(light_xform_vec(m, gpu_float3(0.5f, 0.0f, 0.0f)));  // assumes uniform scale
}

GPU_FN inline float light_cube_area(gpu_storage4x4 m, GPU_THREAD gpu_float3& vx,
                                    GPU_THREAD gpu_float3& vy, GPU_THREAD gpu_float3& vz,
                                    GPU_THREAD float& Axy, GPU_THREAD float& Axz,
                                    GPU_THREAD float& Ayz)
{
    vx = light_xform_vec(m, gpu_float3(1.0f, 0.0f, 0.0f));
    vy = light_xform_vec(m, gpu_float3(0.0f, 1.0f, 0.0f));
    vz = light_xform_vec(m, gpu_float3(0.0f, 0.0f, 1.0f));
    Axy = fabs(length(cross(vx, vy)));
    Axz = fabs(length(cross(vx, vz)));
    Ayz = fabs(length(cross(vy, vz)));
    return 2.0f * (Axy + Axz + Ayz);
}

// Sphere: uniform over the hemisphere facing the shading point pi.
GPU_FN inline LightAreaSample light_sample_sphere(gpu_storage4x4 m, gpu_float3 pi, gpu_float2 u)
{
    LightAreaSample s;
    gpu_float3 c = light_xform_point(m, gpu_float3(0.0f, 0.0f, 0.0f));
    float R = light_sphere_radius(m);
    float z = u.x;
    float sq = sqrt(max(1.0f - z * z, 0.0f));
    float phi = 2.0f * GPU_PI * u.y;
    gpu_float3 local = gpu_float3(cos(phi) * sq, sin(phi) * sq, z);
    gpu_float3 N = normalize(pi - c);
    gpu_float3 t = fabs(N.x) > 0.9f ? gpu_float3(0, 1, 0) : gpu_float3(1, 0, 0);
    gpu_float3 b1 = normalize(cross(N, t));
    gpu_float3 b2 = cross(N, b1);
    s.n = normalize(b1 * local.x + b2 * local.y + N * local.z);
    s.p = c + s.n * R;
    s.pdfArea = 1.0f / (2.0f * GPU_PI * R * R);
    return s;
}

// Cube: face chosen by area, then uniform on the face (CUDA renderer's scheme).
GPU_FN inline LightAreaSample light_sample_cube(gpu_storage4x4 m, gpu_float3 u)
{
    LightAreaSample s;
    gpu_float3 vx, vy, vz;
    float Axy, Axz, Ayz;
    float area = light_cube_area(m, vx, vy, vz, Axy, Axz, Ayz);
    gpu_float3 v0 = light_xform_point(m, gpu_float3(-0.5f, -0.5f, -0.5f));
    s.pdfArea = 1.0f / area;
    float sel = u.x * area;
    float i = u.y, j = u.z;
    float limit = Axy;
    if (sel < limit) {
        s.p = v0 + vx * i + vy * j;
        s.n = normalize(-vz);
    } else if (sel < (limit += Axy)) {
        s.p = v0 + vz + vx * i + vy * j;
        s.n = normalize(vz);
    } else if (sel < (limit += Axz)) {
        s.p = v0 + vx * i + vz * j;
        s.n = normalize(-vy);
    } else if (sel < (limit += Axz)) {
        s.p = v0 + vy + vx * i + vz * j;
        s.n = normalize(vy);
    } else if (sel < (limit += Ayz)) {
        s.p = v0 + vy * i + vz * j;
        s.n = normalize(-vx);
    } else {
        s.p = v0 + vx + vy * i + vz * j;
        s.n = normalize(vx);
    }
    return s;
}

// Triangle (world-space vertices): uniform by area.
GPU_FN inline LightAreaSample light_sample_tri(gpu_float3 v0, gpu_float3 v1, gpu_float3 v2,
                                              gpu_float2 u)
{
    LightAreaSample s;
    float t = sqrt(u.x);
    float b0 = 1.0f - t, b1 = t * u.y;
    s.p = v0 * b0 + v1 * b1 + v2 * (1.0f - b0 - b1);
    gpu_float3 nn = cross(v1 - v0, v2 - v0);
    float area = fabs(length(nn)) * 0.5f;
    s.n = nn / (area > 0.0f ? 2.0f * area : 1e-8f);
    s.pdfArea = 1.0f / area;
    return s;
}

// Converts an area sample to a direction + solid-angle pdf from pi. Returns
// pdf 0 when the light faces away (one-sided).
GPU_FN inline float light_area_to_solid_angle(LightAreaSample s, gpu_float3 pi,
                                              GPU_THREAD gpu_float3& wi, GPU_THREAD float& dist)
{
    gpu_float3 wl = s.p - pi;
    float d2 = dot(wl, wl);
    dist = sqrt(d2);
    wi = wl / dist;
    float NoL = dot(-wi, s.n);
    if (NoL <= 0.0f)
        return 0.0f;
    return s.pdfArea * d2 / NoL;
}

// Solid-angle pdf of hitting light point pLight (unit outward normal nLight,
// 1/area pdfArea) from pi -- the MIS counterpart for BSDF-sampled hits.
GPU_FN inline float light_pdf_area(float pdfArea, gpu_float3 pLight, gpu_float3 nLight,
                                   gpu_float3 pi)
{
    gpu_float3 wl = pi - pLight;
    float d2 = dot(wl, wl);
    float cosL = dot(wl, nLight);
    if (cosL <= 0.0f || d2 <= 0.0f)
        return 0.0f;
    return pdfArea * d2 / (cosL / sqrt(d2));
}

GPU_FN inline float light_pdf_area_sphere(gpu_storage4x4 m)
{
    float R = light_sphere_radius(m);
    return 1.0f / (2.0f * GPU_PI * R * R);
}

GPU_FN inline float light_pdf_area_cube(gpu_storage4x4 m)
{
    gpu_float3 vx, vy, vz;
    float Axy, Axz, Ayz;
    return 1.0f / light_cube_area(m, vx, vy, vz, Axy, Axz, Ayz);
}

GPU_FN inline float light_pdf_area_tri(gpu_float3 v0, gpu_float3 v1, gpu_float3 v2)
{
    return 1.0f / (fabs(length(cross(v1 - v0, v2 - v0))) * 0.5f);
}

// Power heuristic (beta = 2), the MIS weight the CUDA renderer uses.
GPU_FN inline float mis_power2(float pa, float pb)
{
    float a = pa * pa, b = pb * pb;
    return (a + b) > 0.0f ? a / (a + b) : 0.0f;
}

#endif // CORE_LIGHT_SHARED_H
