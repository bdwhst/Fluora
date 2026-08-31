#ifndef CORE_BSDF_SHARED_H
#define CORE_BSDF_SHARED_H
// Portable BSDF sampling (design doc M3, tier-2 of the get_bxdf plan): plain
// functions over value types, no pointers, no tagged dispatch — per-material
// wavefront queues (tier-1) decide which function runs. RGB for now; the
// spectral port swaps SampledSpectrum in for float3 throughput later.
//
// Callers draw the uniform random numbers and pass them in, so megakernel and
// wavefront modes consume identical RNG streams (bitwise-image parity).
// Convention: `rd` is the incoming ray direction (pointing at the surface),
// updated in place to the scattered direction; `nF` is the normal flipped to
// face against rd. A false return means the sample is absorbed. The caller
// offsets the new origin by sign(dot(rdNew, nF)) * eps * nF.
//
// Single-source via the gpu_portable shim (docs/portable-device-code.md). The
// conductor is a port of the GGX VNDF sampling in microfacet.cu (Heitz 2018,
// "Sampling the GGX Distribution of Visible Normals"), Smith height-correlated
// weight F * G2/G1.

#ifndef __METAL_VERSION__
#include "../rhi/gpu_portable.h"
#endif

GPU_FN inline void bsdf_onb(gpu_float3 n, GPU_THREAD gpu_float3& b1, GPU_THREAD gpu_float3& b2)
{
    gpu_float3 t = fabs(n.x) > 0.9f ? gpu_float3(0, 1, 0) : gpu_float3(1, 0, 0);
    b1 = normalize(cross(n, t));
    b2 = cross(n, b1);
}

GPU_FN inline gpu_float3 bsdf_schlick(gpu_float3 f0, float cosTheta)
{
    return f0 + (gpu_float3(1.0f) - f0) * pow(1.0f - cosTheta, 5.0f);
}

GPU_FN inline bool bsdf_sample_lambert(gpu_float3 rgb, gpu_float3 nF, float u1, float u2,
                                GPU_THREAD gpu_float3& rd, GPU_THREAD gpu_float3& throughput)
{
    gpu_float3 b1, b2;
    bsdf_onb(nF, b1, b2);
    float r = sqrt(u1);
    float phi = 2.0f * GPU_PI * u2;
    rd = normalize(b1 * (r * cos(phi)) + b2 * (r * sin(phi))
                   + nF * sqrt(max(0.0f, 1.0f - u1)));
    throughput *= rgb;
    return true;
}

// Heitz 2018 VNDF sampling in tangent space (z = normal), isotropic alpha.
GPU_FN inline gpu_float3 bsdf_ggx_sample_vndf(gpu_float3 wo, float alpha, float u1, float u2)
{
    gpu_float3 Vh = normalize(gpu_float3(alpha * wo.x, alpha * wo.y, wo.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    gpu_float3 T1 = lensq > 0.0f ? gpu_float3(-Vh.y, Vh.x, 0.0f) * rsqrt(lensq)
                                 : gpu_float3(1, 0, 0);
    gpu_float3 T2 = cross(Vh, T1);
    float r = sqrt(u1);
    float phi = 2.0f * GPU_PI * u2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrt(max(0.0f, 1.0f - t1 * t1)) + s * t2;
    gpu_float3 Nh = t1 * T1 + t2 * T2
                  + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
    return normalize(gpu_float3(alpha * Nh.x, alpha * Nh.y, max(0.0f, Nh.z)));
}

GPU_FN inline float bsdf_ggx_lambda(gpu_float3 w, float alpha)
{
    float t = alpha * alpha * (w.x * w.x + w.y * w.y) / (w.z * w.z);
    return 0.5f * (-1.0f + sqrt(1.0f + t));
}

// Rough conductor via GGX VNDF sampling; f0 is the RGB reflectance at normal
// incidence (metallic-workflow stand-in until spectral eta/k lands).
GPU_FN inline bool bsdf_sample_conductor(gpu_float3 f0, float roughness, gpu_float3 nF,
                                  float u1, float u2,
                                  GPU_THREAD gpu_float3& rd, GPU_THREAD gpu_float3& throughput)
{
    if (roughness < 1e-3f) {  // delta: perfect mirror
        float cosTheta = fabs(dot(rd, nF));
        rd = reflect(rd, nF);
        throughput *= bsdf_schlick(f0, cosTheta);
        return true;
    }
    gpu_float3 b1, b2;
    bsdf_onb(nF, b1, b2);
    gpu_float3 wo = gpu_float3(-dot(rd, b1), -dot(rd, b2), -dot(rd, nF));
    float alpha = roughness * roughness;
    gpu_float3 wm = bsdf_ggx_sample_vndf(wo, alpha, u1, u2);
    gpu_float3 wi = 2.0f * dot(wo, wm) * wm - wo;
    if (wi.z <= 0.0f)
        return false;
    float lo = bsdf_ggx_lambda(wo, alpha);
    float li = bsdf_ggx_lambda(wi, alpha);
    // VNDF estimator: f * cos / pdf = F * G2 / G1
    throughput *= bsdf_schlick(f0, dot(wo, wm)) * (1.0f + lo) / (1.0f + lo + li);
    rd = normalize(b1 * wi.x + b2 * wi.y + nF * wi.z);
    return true;
}

// Smooth dielectric, Schlick fresnel (RTIOW-grade until the spectral
// DielectricBxDF port). `n` is the unoriented geometric normal.
GPU_FN inline bool bsdf_sample_dielectric(gpu_float3 rgb, float ior, gpu_float3 n, float u,
                                   GPU_THREAD gpu_float3& rd, GPU_THREAD gpu_float3& throughput)
{
    bool entering = dot(n, rd) < 0.0f;
    gpu_float3 nF = entering ? n : -n;
    float eta = entering ? 1.0f / ior : ior;
    float cosI = fabs(dot(rd, nF));
    float f0s = (1.0f - ior) / (1.0f + ior);
    f0s = f0s * f0s;
    float fresnel = f0s + (1.0f - f0s) * pow(1.0f - cosI, 5.0f);
    gpu_float3 refr = refract(rd, nF, eta);
    if (length_squared(refr) < 1e-8f || u < fresnel) {
        rd = reflect(rd, nF);
    } else {
        throughput *= rgb;
        rd = normalize(refr);
    }
    return true;
}

#endif // CORE_BSDF_SHARED_H
