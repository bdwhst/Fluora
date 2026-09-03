#ifndef CORE_BSDF_SHARED_H
#define CORE_BSDF_SHARED_H
// Portable spectral BSDF sampling (design doc M3, tier-2 of the get_bxdf
// plan): plain functions over value types, no pointers, no tagged dispatch —
// per-material wavefront queues (tier-1) decide which function runs. Ports of
// the CUDA renderer's live BxDFs (bsdf.cu / microfacet.cu, the same float
// expressions): cosine-sampled DiffuseBxDF, smooth DielectricBxDF with real
// Fresnel, and ConductorBxDF on the Trowbridge-Reitz distribution with
// complex-IOR Fresnel. Spectra (GpuSpectrum = gpu_float4) are sampled by the
// caller — these functions never touch tables.
//
// Callers draw the uniform random numbers and pass them in, so megakernel and
// wavefront modes consume identical RNG streams (bitwise-image parity).
// Convention: `rd` is the incoming ray direction (pointing at the surface),
// updated in place to the scattered direction; `nF` is the normal flipped to
// face against rd. A false return means the sample is absorbed. The caller
// offsets the new origin by sign(dot(rdNew, nF)) * eps * nF.

#ifndef __METAL_VERSION__
#include "../rhi/gpu_portable.h"
#include "spectrum_shared.h"
#endif

GPU_FN inline void bsdf_onb(gpu_float3 n, GPU_THREAD gpu_float3& b1, GPU_THREAD gpu_float3& b2)
{
    gpu_float3 t = fabs(n.x) > 0.9f ? gpu_float3(0, 1, 0) : gpu_float3(1, 0, 0);
    b1 = normalize(cross(n, t));
    b2 = cross(n, b1);
}

// DiffuseBxDF: cosine-sampled Lambert; f*cos/pdf collapses to the sampled
// reflectance spectrum. pdf = cos(theta_i)/pi (for MIS against light samples).
GPU_FN inline bool bsdf_sample_lambert(GpuSpectrum reflectance, gpu_float3 nF,
                                float u1, float u2,
                                GPU_THREAD gpu_float3& rd,
                                GPU_THREAD GpuSpectrum& throughput,
                                GPU_THREAD float& pdf)
{
    gpu_float3 b1, b2;
    bsdf_onb(nF, b1, b2);
    float r = sqrt(u1);
    float phi = 2.0f * GPU_PI * u2;
    float cosT = sqrt(max(0.0f, 1.0f - u1));
    rd = normalize(b1 * (r * cos(phi)) + b2 * (r * sin(phi)) + nF * cosT);
    throughput *= reflectance;
    pdf = cosT * (1.0f / GPU_PI);
    return true;
}

// DiffuseBxDF::eval / pdf for a given incident direction wi (world space,
// pointing away from the surface): f is cos-premultiplied like the CUDA
// renderer's. Zero below the flipped normal (reflection only).
GPU_FN inline bool bsdf_eval_lambert(GpuSpectrum reflectance, gpu_float3 nF, gpu_float3 wi,
                                     GPU_THREAD GpuSpectrum& f, GPU_THREAD float& pdf)
{
    float cosI = dot(wi, nF);
    if (cosI <= 0.0f) {
        f = GpuSpectrum(0.0f);
        pdf = 0.0f;
        return false;
    }
    f = reflectance * (cosI * (1.0f / GPU_PI));
    pdf = cosI * (1.0f / GPU_PI);
    return true;
}

// ---------------------------------------------------------------------------
// Trowbridge-Reitz distribution (microfacet.cu TRDistribution, isotropic),
// tangent space with z = flipped normal so w.z > 0 for outgoing directions.
// ---------------------------------------------------------------------------

GPU_FN inline float bsdf_tr_D(gpu_float3 wm, float alpha)
{
    float cos2 = wm.z * wm.z;
    float sin2 = max(0.0f, 1.0f - cos2);
    if (cos2 == 0.0f)
        return 0.0f;   // tan2 infinite
    float tan2 = sin2 / cos2;
    float cos4 = cos2 * cos2;
    if (cos4 < 1e-16f)
        return 0.0f;
    float e = tan2 / (alpha * alpha);   // isotropic: (cosPhi^2+sinPhi^2)/a^2
    return 1.0f / (GPU_PI * alpha * alpha * cos4 * (1.0f + e) * (1.0f + e));
}

GPU_FN inline float bsdf_tr_lambda(gpu_float3 w, float alpha)
{
    float cos2 = w.z * w.z;
    float sin2 = max(0.0f, 1.0f - cos2);
    if (cos2 == 0.0f)
        return 0.0f;
    float tan2 = sin2 / cos2;
    return (sqrt(1.0f + alpha * alpha * tan2) - 1.0f) / 2.0f;
}

// TRDistribution::sample_wm — VNDF sampling via the uniform-disk warp.
GPU_FN inline gpu_float3 bsdf_tr_sample_wm(gpu_float3 w, float alpha, float u1, float u2)
{
    gpu_float3 wh = normalize(gpu_float3(w.x * alpha, w.y * alpha, w.z));
    if (wh.z < 0.0f)
        wh = -wh;
    gpu_float3 t1 = (wh.z < 0.99999f) ? normalize(cross(gpu_float3(0, 0, 1), wh))
                                      : gpu_float3(1, 0, 0);
    gpu_float3 t2 = cross(wh, t1);
    float r = sqrt(u1);
    float theta = 2.0f * GPU_PI * u2;
    float px = r * cos(theta);
    float py = r * sin(theta);
    float h = sqrt(1.0f - px * px);
    float s = (1.0f + wh.z) / 2.0f;
    py = (1.0f - s) * h + s * py;      // math::lerp(s, h, py)
    float pz = sqrt(max(0.0f, 1.0f - px * px - py * py));
    gpu_float3 nh = px * t1 + py * t2 + pz * wh;
    return normalize(gpu_float3(alpha * nh.x, alpha * nh.y, max(1e-6f, nh.z)));
}

// ConductorBxDF::sample_f: alpha < 1e-3 degenerates to a mirror
// (effectively_smooth); otherwise VNDF-sampled Trowbridge-Reitz with
// throughput f/pdf = F * G / G1.
GPU_FN inline bool bsdf_sample_conductor(GpuSpectrum eta, GpuSpectrum k, float alpha,
                                  gpu_float3 nF, float u1, float u2,
                                  GPU_THREAD gpu_float3& rd,
                                  GPU_THREAD GpuSpectrum& throughput,
                                  GPU_THREAD float& pdf)
{
    gpu_float3 b1, b2;
    bsdf_onb(nF, b1, b2);
    gpu_float3 wo = gpu_float3(-dot(rd, b1), -dot(rd, b2), -dot(rd, nF));
    if (alpha < 1e-3f) {
        // Specular: sample_f returns F at pdf 1 (delta f*cos cancels).
        throughput *= spd_fr_complex(fabs(wo.z), eta, k);
        rd = reflect(rd, nF);
        pdf = 1.0f;
        return true;
    }
    gpu_float3 wm = bsdf_tr_sample_wm(wo, alpha, u1, u2);
    gpu_float3 wi = -wo + 2.0f * dot(wo, wm) * wm;
    if (wo.z * wi.z <= 0.0f)
        return false;
    float cosO = fabs(wo.z), cosI = fabs(wi.z);
    if (cosO == 0.0f || cosI == 0.0f)
        return false;
    float D = bsdf_tr_D(wm, alpha);
    float lo = bsdf_tr_lambda(wo, alpha);
    float li = bsdf_tr_lambda(wi, alpha);
    float G = 1.0f / (1.0f + lo + li);
    float G1o = 1.0f / (1.0f + lo);
    float absDotOM = fabs(dot(wo, wm));
    // pdf = D_visible/(4|wo.wm|) with D_visible = G1(wo)/|cos wo| D |wo.wm|.
    pdf = (G1o / cosO * D * absDotOM) / (4.0f * absDotOM);
    if (pdf <= 0.0f)
        return false;
    GpuSpectrum F = spd_clamp_zero(spd_fr_complex(absDotOM, eta, k));
    // f (cos-premultiplied) = D F G / (4 cosO); throughput *= f / pdf.
    throughput *= F * (D * G / (4.0f * cosO) / pdf);
    rd = normalize(b1 * wi.x + b2 * wi.y + nF * wi.z);
    return true;
}

// ConductorBxDF::eval / pdf for incident wi (world, away from surface), given
// the incoming ray direction rd (pointing at the surface). Smooth conductors
// are delta distributions: zero. f is cos-premultiplied.
GPU_FN inline bool bsdf_eval_conductor(GpuSpectrum eta, GpuSpectrum k, float alpha,
                                       gpu_float3 nF, gpu_float3 rd, gpu_float3 wi,
                                       GPU_THREAD GpuSpectrum& f, GPU_THREAD float& pdf)
{
    f = GpuSpectrum(0.0f);
    pdf = 0.0f;
    if (alpha < 1e-3f)
        return false;
    gpu_float3 b1, b2;
    bsdf_onb(nF, b1, b2);
    gpu_float3 wo = gpu_float3(-dot(rd, b1), -dot(rd, b2), -dot(rd, nF));
    gpu_float3 wil = gpu_float3(dot(wi, b1), dot(wi, b2), dot(wi, nF));
    gpu_float3 wm = wo + wil;
    if (dot(wm, wm) < 1e-9f)
        return false;
    wm = normalize(wm);
    if (!(wil.z > 0.0f && dot(wm, wil) > 0.0f))
        return false;
    float cosO = fabs(wo.z);
    if (cosO == 0.0f)
        return false;
    float absDotOM = fabs(dot(wo, wm));
    float D = bsdf_tr_D(wm, alpha);
    float lo = bsdf_tr_lambda(wo, alpha);
    float li = bsdf_tr_lambda(wil, alpha);
    float G = 1.0f / (1.0f + lo + li);
    float G1o = 1.0f / (1.0f + lo);
    GpuSpectrum F = spd_clamp_zero(spd_fr_complex(absDotOM, eta, k));
    f = F * (D * G / (4.0f * cosO));
    pdf = (G1o / cosO * D * absDotOM) / (4.0f * absDotOM);
    return pdf > 0.0f;
}

// math::frensel_dielectric (etaI = 1).
GPU_FN inline float bsdf_fresnel_dielectric(float cosThetaI, float etaT)
{
    float sinThetaI = sqrt(max(1.0f - cosThetaI * cosThetaI, 0.0f));
    float sinThetaT = sinThetaI / etaT;
    if (sinThetaT >= 1.0f)
        return 1.0f;   // total internal reflection
    float cosThetaT = sqrt(max(1.0f - sinThetaT * sinThetaT, 0.0f));
    float rparll = (etaT * cosThetaI - cosThetaT) / (etaT * cosThetaI + cosThetaT);
    float rperpe = (cosThetaI - etaT * cosThetaT) / (cosThetaI + etaT * cosThetaT);
    return (rparll * rparll + rperpe * rperpe) * 0.5f;
}

// DielectricBxDF::sample_f: smooth glass, real Fresnel, radiance-scaled
// refraction ((1-F)/eta^2 over pdf 1-F). etaVal is the material's eta at the
// hero wavelength; the caller handles dispersion (terminate_secondary).
// `n` is the unoriented geometric/shading normal. Refraction that hits total
// internal reflection after winning the lottery is absorbed (CUDA pdf = 0).
GPU_FN inline bool bsdf_sample_dielectric(float etaVal, gpu_float3 n, float u,
                                   GPU_THREAD gpu_float3& rd,
                                   GPU_THREAD GpuSpectrum& throughput,
                                   GPU_THREAD float& pdf)
{
    bool entering = dot(n, rd) < 0.0f;
    gpu_float3 nF = entering ? n : -n;
    float eta = entering ? etaVal : 1.0f / etaVal;
    float cosI = fabs(dot(rd, nF));
    float fresnel = bsdf_fresnel_dielectric(cosI, eta);
    if (u < fresnel) {
        rd = reflect(rd, nF);   // throughput *= F/F = 1
        pdf = fresnel;
        return true;
    }
    pdf = 1.0f - fresnel;
    // geomerty_refract(wo, nF, 1/eta): wo = -rd, cosThetaI = cosI >= 0.
    float invEta = 1.0f / eta;
    float sin2ThetaI = max(0.0f, 1.0f - cosI * cosI);
    float sin2ThetaT = invEta * invEta * sin2ThetaI;
    if (sin2ThetaT >= 1.0f)
        return false;           // TIR on the refract branch: absorbed
    float cosThetaT = sqrt(1.0f - sin2ThetaT);
    rd = normalize(invEta * rd + (invEta * cosI - cosThetaT) * nF);
    throughput *= 1.0f / (eta * eta);   // (1-F)/eta^2 over pdf 1-F
    return true;
}

#endif // CORE_BSDF_SHARED_H
