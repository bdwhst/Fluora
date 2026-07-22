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
// MSL-only today (concatenated after accel_shared.h); the CUDA twin arrives
// with the M3 math shim. The conductor is a port of the GGX VNDF sampling in
// microfacet.cu (Heitz 2018, "Sampling the GGX Distribution of Visible
// Normals"), Smith height-correlated weight F * G2/G1.

#ifdef __METAL_VERSION__

inline void bsdf_onb(float3 n, thread float3& b1, thread float3& b2)
{
    float3 t = metal::abs(n.x) > 0.9f ? float3(0, 1, 0) : float3(1, 0, 0);
    b1 = metal::normalize(metal::cross(n, t));
    b2 = metal::cross(n, b1);
}

inline float3 bsdf_schlick(float3 f0, float cosTheta)
{
    return f0 + (float3(1.0f) - f0) * metal::pow(1.0f - cosTheta, 5.0f);
}

inline bool bsdf_sample_lambert(float3 rgb, float3 nF, float u1, float u2,
                                thread float3& rd, thread float3& throughput)
{
    float3 b1, b2;
    bsdf_onb(nF, b1, b2);
    float r = metal::sqrt(u1);
    float phi = 2.0f * M_PI_F * u2;
    rd = metal::normalize(b1 * (r * metal::cos(phi)) + b2 * (r * metal::sin(phi))
                          + nF * metal::sqrt(metal::max(0.0f, 1.0f - u1)));
    throughput *= rgb;
    return true;
}

// Heitz 2018 VNDF sampling in tangent space (z = normal), isotropic alpha.
inline float3 bsdf_ggx_sample_vndf(float3 wo, float alpha, float u1, float u2)
{
    float3 Vh = metal::normalize(float3(alpha * wo.x, alpha * wo.y, wo.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0f ? float3(-Vh.y, Vh.x, 0.0f) * metal::rsqrt(lensq)
                             : float3(1, 0, 0);
    float3 T2 = metal::cross(Vh, T1);
    float r = metal::sqrt(u1);
    float phi = 2.0f * M_PI_F * u2;
    float t1 = r * metal::cos(phi);
    float t2 = r * metal::sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * metal::sqrt(metal::max(0.0f, 1.0f - t1 * t1)) + s * t2;
    float3 Nh = t1 * T1 + t2 * T2
              + metal::sqrt(metal::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
    return metal::normalize(float3(alpha * Nh.x, alpha * Nh.y, metal::max(0.0f, Nh.z)));
}

inline float bsdf_ggx_lambda(float3 w, float alpha)
{
    float t = alpha * alpha * (w.x * w.x + w.y * w.y) / (w.z * w.z);
    return 0.5f * (-1.0f + metal::sqrt(1.0f + t));
}

// Rough conductor via GGX VNDF sampling; f0 is the RGB reflectance at normal
// incidence (metallic-workflow stand-in until spectral eta/k lands).
inline bool bsdf_sample_conductor(float3 f0, float roughness, float3 nF,
                                  float u1, float u2,
                                  thread float3& rd, thread float3& throughput)
{
    if (roughness < 1e-3f) {  // delta: perfect mirror
        float cosTheta = metal::fabs(metal::dot(rd, nF));
        rd = metal::reflect(rd, nF);
        throughput *= bsdf_schlick(f0, cosTheta);
        return true;
    }
    float3 b1, b2;
    bsdf_onb(nF, b1, b2);
    float3 wo = float3(-metal::dot(rd, b1), -metal::dot(rd, b2), -metal::dot(rd, nF));
    float alpha = roughness * roughness;
    float3 wm = bsdf_ggx_sample_vndf(wo, alpha, u1, u2);
    float3 wi = 2.0f * metal::dot(wo, wm) * wm - wo;
    if (wi.z <= 0.0f)
        return false;
    float lo = bsdf_ggx_lambda(wo, alpha);
    float li = bsdf_ggx_lambda(wi, alpha);
    // VNDF estimator: f * cos / pdf = F * G2 / G1
    throughput *= bsdf_schlick(f0, metal::dot(wo, wm)) * (1.0f + lo) / (1.0f + lo + li);
    rd = metal::normalize(b1 * wi.x + b2 * wi.y + nF * wi.z);
    return true;
}

// Smooth dielectric, Schlick fresnel (RTIOW-grade until the spectral
// DielectricBxDF port). `n` is the unoriented geometric normal.
inline bool bsdf_sample_dielectric(float3 rgb, float ior, float3 n, float u,
                                   thread float3& rd, thread float3& throughput)
{
    bool entering = metal::dot(n, rd) < 0.0f;
    float3 nF = entering ? n : -n;
    float eta = entering ? 1.0f / ior : ior;
    float cosI = metal::fabs(metal::dot(rd, nF));
    float f0s = (1.0f - ior) / (1.0f + ior);
    f0s = f0s * f0s;
    float fresnel = f0s + (1.0f - f0s) * metal::pow(1.0f - cosI, 5.0f);
    float3 refr = metal::refract(rd, nF, eta);
    if (metal::length_squared(refr) < 1e-8f || u < fresnel) {
        rd = metal::reflect(rd, nF);
    } else {
        throughput *= rgb;
        rd = metal::normalize(refr);
    }
    return true;
}

#endif // __METAL_VERSION__
#endif // CORE_BSDF_SHARED_H
