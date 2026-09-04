#ifndef CORE_MEDIUM_SHARED_H
#define CORE_MEDIUM_SHARED_H
// Participating media for the portable renderer core (the CUDA renderer's
// media.h / medium.h port, M4 part 2 step 4): the device-side medium record
// and the Henyey-Greenstein phase function. Single-source across MSL, CUDA
// and host C++ via gpu_portable.h (docs/portable-device-code.md). Under MSL
// this file is concatenated after spectrum_shared.h and bsdf_shared.h
// (bsdf_onb); elsewhere the #includes resolve.
//
// Distance sampling itself lives with the integrator (mini/pathtrace_gpu.h
// miniTrace): a homogeneous medium is one majorant segment whose real
// collisions are the only events, so the tracker needs nothing beyond the
// coefficients here. Grid media (NanoVDB) will add a majorant iterator next
// to this record.

#ifndef __METAL_VERSION__
#include "../rhi/gpu_portable.h"
#include "spectrum_shared.h"
#include "bsdf_shared.h"
#endif

#define MEDIUM_HOMOGENEOUS 0
#define MEDIUM_NANOVDB     1   // parsed, not rendered yet: uploads as empty

// One "Media" entry. sigma_a/sigma_s are dense-spectrum offsets into the spd
// table (SIGMA_SCALE already applied on the host, RGB widened as an unbounded
// rgb2spec spectrum); 16 bytes, host/device layout identical.
struct MediumGpu {
    unsigned int sigmaASpd;
    unsigned int sigmaSSpd;
    float g;                 // Henyey-Greenstein asymmetry, (-1, 1)
    unsigned int type;       // MEDIUM_*
};

// Beer-Lambert transmittance over distance t. Written per component so an
// infinite t (the ray escaped while inside the medium) gives exactly 0 for
// sigma_t > 0 and exactly 1 for sigma_t == 0, never 0*inf.
GPU_FN inline GpuSpectrum medium_transmittance(GpuSpectrum sigmaT, float t)
{
    GpuSpectrum T;
    for (int i = 0; i < SPD_N_SAMPLES; i++)
        T[i] = sigmaT[i] > 0.0f ? exp(-sigmaT[i] * t) : 1.0f;
    return T;
}

// Henyey-Greenstein phase function, PBRT convention: wo points back toward
// the previous vertex, so cosTheta = dot(wo, wi) and g > 0 peaks at
// wi = -wo (forward scattering). Normalized over the sphere, so it is its own
// sampling pdf below.
GPU_FN inline float hg_phase(float cosTheta, float g)
{
    float denom = 1.0f + g * g + 2.0f * g * cosTheta;
    return (1.0f / (4.0f * GPU_PI)) * (1.0f - g * g) / (denom * sqrt(max(denom, 0.0f)));
}

// Samples wi with pdf = hg_phase(dot(wo, wi), g) (SampleHenyeyGreenstein;
// |g| < 1e-3 falls back to the uniform sphere). Returns the pdf/phase value.
GPU_FN inline float hg_sample(gpu_float3 wo, float g, float u1, float u2,
                              GPU_THREAD gpu_float3& wi)
{
    float cosTheta;
    if (fabs(g) < 1e-3f) {
        cosTheta = 1.0f - 2.0f * u1;
    } else {
        float sq = (1.0f - g * g) / (1.0f + g - 2.0f * g * u1);
        cosTheta = -1.0f / (2.0f * g) * (1.0f + g * g - sq * sq);
    }
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * GPU_PI * u2;
    gpu_float3 b1, b2;
    bsdf_onb(wo, b1, b2);
    wi = normalize(b1 * (sinTheta * cos(phi)) + b2 * (sinTheta * sin(phi)) + wo * cosTheta);
    return hg_phase(cosTheta, g);
}

#endif // CORE_MEDIUM_SHARED_H
