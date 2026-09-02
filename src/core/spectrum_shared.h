#ifndef CORE_SPECTRUM_SHARED_H
#define CORE_SPECTRUM_SHARED_H
// Portable spectral core (docs/portable-device-code.md): device-side ports of
// the CUDA renderer's spectrum.h / color.h / microfacet.cu spectral pieces,
// restructured onto flat buffers (invariant I-1 — offsets, not pointers):
//
//   - A spectrum sample is ONE gpu_float4 (spec::NSpectrumSamples == 4), so
//     all spectrum math is componentwise vector math on every backend.
//   - Dense spectra (CIE X/Y/Z, illuminants, measured eta/k) are 471-float
//     runs at fixed/parameter offsets in a single "spd" table buffer, built
//     by core/spectra.cpp.
//   - The sRGB RGB->sigmoid-coefficient table (PBRT RGBToSpectrumTable) is a
//     second buffer: [0,64) zNodes, then coeffs[3][64][64][64][3].
//
// Float expressions mirror the CUDA renderer's (lround via floor(x+.5) on
// positive lambdas, same complex-sqrt branch structure) so M4 can diff
// backends numerically.

#ifndef __METAL_VERSION__
#include "../rhi/gpu_portable.h"
#endif

#define SPD_N_SAMPLES 4
#define SPD_LAMBDA_MIN 360.0f
#define SPD_LAMBDA_MAX 830.0f
#define SPD_TABLE_SIZE 471u          // floats per dense spectrum (360..830)
#define SPD_CIE_Y_INTEGRAL 106.856895f
#define SPD_IMAGING_RATIO 0.03f      // PixelSensor imagingRatio (pathtrace.cu)

// Fixed dense-table offsets: core/spectra.cpp writes these four first, then
// scene-referenced named spectra at offsets carried in material params.
#define SPD_OFF_CIE_X 0u
#define SPD_OFF_CIE_Y 471u
#define SPD_OFF_CIE_Z 942u
#define SPD_OFF_ILLUM_D65 1413u
#define SPD_FIXED_TABLE_FLOATS 1884u
#define SPD_NONE 0xFFFFFFFFu

#define SPD_RGB2SPEC_RES 64

typedef gpu_float4 GpuSpectrum;

struct GpuWavelengths {
    gpu_float4 lambda;
    gpu_float4 pdf;
};

GPU_FN inline float spd_average(GpuSpectrum s)
{
    return (s.x + s.y + s.z + s.w) * (1.0f / SPD_N_SAMPLES);
}

// CUDA safe_div: pdf==0 components pass the numerator through.
GPU_FN inline GpuSpectrum spd_safe_div(GpuSpectrum s0, GpuSpectrum s1)
{
    GpuSpectrum s;
    for (int i = 0; i < SPD_N_SAMPLES; i++)
        s[i] = s1[i] == 0.0f ? s0[i] : s0[i] / s1[i];
    return s;
}

GPU_FN inline GpuSpectrum spd_clamp_zero(GpuSpectrum s)
{
    GpuSpectrum r;
    for (int i = 0; i < SPD_N_SAMPLES; i++)
        r[i] = max(s[i], 0.0f);
    return r;
}

// ---------------------------------------------------------------------------
// Wavelength sampling (SampledWavelengths::sample_visible): importance-sample
// the visible band with the PBRT analytic fit; stratified across the 4 slots.
// ---------------------------------------------------------------------------

GPU_FN inline float spd_visible_pdf(float lambda)
{
    if (lambda < SPD_LAMBDA_MIN || lambda > SPD_LAMBDA_MAX)
        return 0.0f;
    float c = coshf(0.0072f * (lambda - 538.0f));
    return 0.0039398042f / (c * c);
}

GPU_FN inline GpuWavelengths spd_sample_visible(float u)
{
    GpuWavelengths swl;
    for (int i = 0; i < SPD_N_SAMPLES; i++) {
        float up = u + (float)i / SPD_N_SAMPLES;
        if (up > 1.0f)
            up -= 1.0f;
        swl.lambda[i] = 538.0f - 138.888889f * atanhf(0.85691062f - 1.82750197f * up);
        swl.pdf[i] = spd_visible_pdf(swl.lambda[i]);
    }
    return swl;
}

GPU_FN inline bool spd_secondary_terminated(GpuWavelengths swl)
{
    return swl.pdf.y == 0.0f && swl.pdf.z == 0.0f && swl.pdf.w == 0.0f;
}

// Collapse to the hero wavelength (dispersion): pdf[1..3] = 0, pdf[0] /= N.
GPU_FN inline void spd_terminate_secondary(GPU_THREAD GpuWavelengths& swl)
{
    if (spd_secondary_terminated(swl))
        return;
    swl.pdf.y = swl.pdf.z = swl.pdf.w = 0.0f;
    swl.pdf.x /= SPD_N_SAMPLES;
}

// ---------------------------------------------------------------------------
// Dense spectra: 471 floats per spectrum, nearest-nanometer lookup
// (DenselySampledSpectrum::sample; lround == floor(x+.5) for positive x).
// ---------------------------------------------------------------------------

GPU_FN inline GpuSpectrum spd_dense_sample(GPU_DEVICE const float* spd, uint offset,
                                    GpuWavelengths swl)
{
    GpuSpectrum s;
    for (int i = 0; i < SPD_N_SAMPLES; i++) {
        int o = (int)(swl.lambda[i] + 0.5f) - (int)SPD_LAMBDA_MIN;
        s[i] = (o < 0 || o >= (int)SPD_TABLE_SIZE) ? 0.0f : spd[offset + o];
    }
    return s;
}

// ---------------------------------------------------------------------------
// RGB -> spectrum via sigmoid polynomials (PBRT RGBToSpectrumTable /
// RGBSigmoidPolynomial, ported from color.cu onto the rgb2spec buffer).
// ---------------------------------------------------------------------------

struct SpdPoly {
    float c0, c1, c2;
};

// math::lerp's exact form ((1-x)*a + x*b — not a+(b-a)*x) and integer clamp,
// spelled here because MSL mix() and host <algorithm> min/max don't line up
// across personalities.
GPU_FN inline float spd_lerp(float x, float a, float b)
{
    return (1.0f - x) * a + x * b;
}

GPU_FN inline int spd_clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

GPU_FN inline float spd_sigmoid(float x)
{
    if (gpu_isinf(x))
        return x > 0.0f ? 1.0f : 0.0f;
    return 0.5f + x / (2.0f * sqrt(1.0f + x * x));
}

// evaluate_polynomial(lambda, c2, c1, c0) spelled as the same fma chain.
GPU_FN inline float spd_poly_eval(SpdPoly p, float lambda)
{
    return spd_sigmoid(fmaf(lambda, fmaf(lambda, p.c0, p.c1), p.c2));
}

GPU_FN inline SpdPoly spd_rgb_to_coeffs(GPU_DEVICE const float* r2s, gpu_float3 rgb)
{
    rgb = max(rgb, gpu_float3(0.0f, 0.0f, 0.0f));
    // Uniform rgb: constant spectrum via the closed form.
    if (rgb.x == rgb.y && rgb.y == rgb.z) {
        SpdPoly p;
        p.c0 = 0.0f;
        p.c1 = 0.0f;
        p.c2 = (rgb.x - 0.5f) / sqrt(max(rgb.x * (1.0f - rgb.x), 0.0f));
        return p;
    }
    const int res = SPD_RGB2SPEC_RES;
    float c[3] = { rgb.x, rgb.y, rgb.z };
    int maxc = (c[0] > c[1]) ? ((c[0] > c[2]) ? 0 : 2) : ((c[1] > c[2]) ? 1 : 2);
    float z = c[maxc];
    float x = c[(maxc + 1) % 3] * (res - 1) / z;
    float y = c[(maxc + 2) % 3] * (res - 1) / z;

    int xi = spd_clampi((int)x, 0, res - 2);
    int yi = spd_clampi((int)y, 0, res - 2);
    // find_interval over zNodes (r2s[0..res)): largest i with zNodes[i] < z.
    // ("half" is an MSL type name, hence "hlf".)
    int size = res - 2, first = 1;
    while (size > 0) {
        int hlf = size >> 1, mid = first + hlf;
        bool pred = r2s[mid] < z;
        first = pred ? mid + 1 : first;
        size = pred ? size - hlf - 1 : hlf;
    }
    int zi = spd_clampi(first - 1, 0, res - 2);
    float dx = x - xi, dy = y - yi;
    float dz = (z - r2s[zi]) / (r2s[zi + 1] - r2s[zi]);

    // Trilinear interpolation of coeffs[maxc][zi..][yi..][xi..][k], laid out
    // after the res zNodes floats.
    SpdPoly p;
    for (int k = 0; k < 3; k++) {
        uint base = (uint)res
                  + ((((uint)maxc * res + (uint)zi) * res + (uint)yi) * res + (uint)xi) * 3u
                  + (uint)k;
        uint sx = 3u, sy = (uint)res * 3u, sz = (uint)res * res * 3u;
        float c000 = r2s[base], c100 = r2s[base + sx];
        float c010 = r2s[base + sy], c110 = r2s[base + sy + sx];
        float c001 = r2s[base + sz], c101 = r2s[base + sz + sx];
        float c011 = r2s[base + sz + sy], c111 = r2s[base + sz + sy + sx];
        float v = spd_lerp(dz,
                           spd_lerp(dy, spd_lerp(dx, c000, c100),
                                    spd_lerp(dx, c010, c110)),
                           spd_lerp(dy, spd_lerp(dx, c001, c101),
                                    spd_lerp(dx, c011, c111)));
        if (k == 0) p.c0 = v; else if (k == 1) p.c1 = v; else p.c2 = v;
    }
    return p;
}

// RGBAlbedoSpectrum: bounded reflectance.
GPU_FN inline GpuSpectrum spd_rgb_albedo_sample(GPU_DEVICE const float* r2s,
                                         gpu_float3 rgb, GpuWavelengths swl)
{
    SpdPoly p = spd_rgb_to_coeffs(r2s, rgb);
    GpuSpectrum s;
    for (int i = 0; i < SPD_N_SAMPLES; i++)
        s[i] = spd_poly_eval(p, swl.lambda[i]);
    return s;
}

// RGBIlluminantSpectrum: scale*sigmoid times the color space's illuminant
// (D65 for sRGB) — used for emitters and env-map texels.
GPU_FN inline GpuSpectrum spd_rgb_illuminant_sample(GPU_DEVICE const float* r2s,
                                             GPU_DEVICE const float* spd,
                                             gpu_float3 rgb, GpuWavelengths swl)
{
    float m = max(max(rgb.x, rgb.y), rgb.z);
    float scale = 2.0f * m;
    SpdPoly p = spd_rgb_to_coeffs(r2s, scale != 0.0f ? rgb / scale
                                                     : gpu_float3(0.0f, 0.0f, 0.0f));
    GpuSpectrum s;
    for (int i = 0; i < SPD_N_SAMPLES; i++)
        s[i] = scale * spd_poly_eval(p, swl.lambda[i]);
    return s * spd_dense_sample(spd, SPD_OFF_ILLUM_D65, swl);
}

// ---------------------------------------------------------------------------
// Complex Fresnel for conductors (microfacet.cu fr_complex, same branch
// structure and float expressions).
// ---------------------------------------------------------------------------

struct SpdComplex {
    float re, im;
};

GPU_FN inline SpdComplex spd_cmul(SpdComplex a, SpdComplex b)
{
    return SpdComplex{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}

GPU_FN inline SpdComplex spd_cdiv(SpdComplex a, SpdComplex b)
{
    float scale = 1.0f / (b.re * b.re + b.im * b.im);
    return SpdComplex{ scale * (a.re * b.re + a.im * b.im),
                       scale * (a.im * b.re - a.re * b.im) };
}

GPU_FN inline SpdComplex spd_csqrt(SpdComplex z)
{
    float n = sqrt(z.re * z.re + z.im * z.im);
    float t1 = sqrt(0.5f * (n + fabs(z.re)));
    float t2 = 0.5f * z.im / t1;
    if (n == 0.0f)
        return SpdComplex{ 0.0f, 0.0f };
    if (z.re >= 0.0f)
        return SpdComplex{ t1, t2 };
    return SpdComplex{ fabs(t2), copysignf(t1, z.im) };
}

GPU_FN inline float spd_fr_complex1(float cos_theta_i, SpdComplex eta)
{
    cos_theta_i = min(max(cos_theta_i, 0.0f), 1.0f);
    float sin2_theta_i = 1.0f - cos_theta_i * cos_theta_i;
    SpdComplex sin2_theta_t = spd_cdiv(SpdComplex{ sin2_theta_i, 0.0f },
                                       spd_cmul(eta, eta));
    SpdComplex cos_theta_t = spd_csqrt(SpdComplex{ 1.0f - sin2_theta_t.re,
                                                   -sin2_theta_t.im });
    SpdComplex eci = SpdComplex{ eta.re * cos_theta_i, eta.im * cos_theta_i };
    SpdComplex r_parl = spd_cdiv(
        SpdComplex{ eci.re - cos_theta_t.re, eci.im - cos_theta_t.im },
        SpdComplex{ eci.re + cos_theta_t.re, eci.im + cos_theta_t.im });
    SpdComplex ect = spd_cmul(eta, cos_theta_t);
    SpdComplex r_perp = spd_cdiv(
        SpdComplex{ cos_theta_i - ect.re, -ect.im },
        SpdComplex{ cos_theta_i + ect.re, ect.im });
    float norm_parl = r_parl.re * r_parl.re + r_parl.im * r_parl.im;
    float norm_perp = r_perp.re * r_perp.re + r_perp.im * r_perp.im;
    return (norm_parl + norm_perp) * 0.5f;
}

GPU_FN inline GpuSpectrum spd_fr_complex(float cos_theta_i, GpuSpectrum eta, GpuSpectrum k)
{
    GpuSpectrum res;
    for (int i = 0; i < SPD_N_SAMPLES; i++)
        res[i] = spd_fr_complex1(cos_theta_i, SpdComplex{ eta[i], k[i] });
    return res;
}

// ---------------------------------------------------------------------------
// Pixel sensor (camera.h PixelSensor with CIE X/Y/Z matching curves, no white
// balance): spectral radiance -> sensor XYZ. The film's output matrix
// (sRGB RGBFromXYZ, computed host-side by core/spectra.cpp) is applied by the
// caller.
// ---------------------------------------------------------------------------

GPU_FN inline gpu_float3 spd_to_sensor_xyz(GpuSpectrum L, GpuWavelengths swl,
                                    GPU_DEVICE const float* spd)
{
    L = spd_safe_div(L, swl.pdf);
    GpuSpectrum X = spd_dense_sample(spd, SPD_OFF_CIE_X, swl);
    GpuSpectrum Y = spd_dense_sample(spd, SPD_OFF_CIE_Y, swl);
    GpuSpectrum Z = spd_dense_sample(spd, SPD_OFF_CIE_Z, swl);
    return SPD_IMAGING_RATIO * gpu_float3(spd_average(X * L), spd_average(Y * L),
                                          spd_average(Z * L));
}

#endif // CORE_SPECTRUM_SHARED_H
