#ifndef TEST_SHARED_PROBE_H
#define TEST_SHARED_PROBE_H
// Probe body for SharedHostTest: evaluates the shared BSDF/RNG/env/tonemap/
// spectral functions on fixed inputs, writing float4 slots. Spelled once as a
// macro so the exact same text runs in the host personality (#include) and the
// MSL personality (concatenated before the probe kernel). Uses only gpu_*
// types and shared functions — this file is itself portable device code.
//
// R2S/SPD are the rgb2spec and dense-spectra buffers (core/spectra.cpp builds
// them host-side; the test uploads the same bytes for the GPU pass).
//
// Slots: 0-1 rng draws; 2-3 lambert rd/thr (w of 2 = pdf); 4-5 rough
// conductor rd/thr; 6-7 mirror conductor rd/thr; 8-9 dielectric refract
// rd/thr; 10 dielectric reflect rd; 11 equirect uv; 12 ACES; 13-14
// visible-wavelength lambda/pdf; 15 sigmoid-poly + complex-Fresnel +
// visible-pdf scalars; 16 pdf after terminate_secondary; 17 dense CIE-Y
// sample; 18 rgb->albedo spectrum; 19 rgb->illuminant spectrum; 20 sensor
// XYZ; 21 lambert eval f (xyz) + pdf (w); 22 rough conductor eval f (xyz) +
// pdf (w). The test sizes its buffers from PROBE_SLOTS — bump it with every
// added slot.

#define PROBE_SLOTS 23

// RNG draws land in named locals first: constructor-argument evaluation order
// is unspecified in C++, and nvcc/MSVC need not match clang's left-to-right.
#define PROBE_BODY(OUT, R2S, SPD)                                              \
    {                                                                          \
        GpuRng rng;                                                            \
        rng.state = 12345u;                                                    \
        float r0 = gpu_rand(rng), r1 = gpu_rand(rng);                          \
        float r2 = gpu_rand(rng), r3 = gpu_rand(rng);                          \
        OUT[0] = gpu_float4(r0, r1, r2, r3);                                   \
        float r4 = gpu_rand(rng), r5 = gpu_rand(rng);                          \
        float r6 = gpu_rand(rng), r7 = gpu_rand(rng);                          \
        OUT[1] = gpu_float4(r4, r5, r6, r7);                                   \
        gpu_float3 nF = gpu_float3(0.0f, 0.0f, 1.0f);                          \
        gpu_float3 rdIn = normalize(gpu_float3(0.5f, 0.3f, -1.0f));            \
        gpu_float3 rd;                                                         \
        GpuSpectrum thr;                                                       \
        float pdf;                                                             \
        GpuSpectrum cEta = gpu_float4(1.1f, 1.0f, 0.9f, 1.2f);                 \
        GpuSpectrum cK = gpu_float4(2.0f, 3.0f, 2.5f, 1.8f);                   \
        rd = rdIn; thr = GpuSpectrum(1.0f);                                    \
        bsdf_sample_lambert(gpu_float4(0.8f, 0.6f, 0.4f, 0.5f), nF, 0.3f,      \
                            0.7f, rd, thr, pdf);                               \
        OUT[2] = gpu_float4(rd, pdf);                                          \
        OUT[3] = thr;                                                          \
        rd = rdIn; thr = GpuSpectrum(1.0f);                                    \
        bool aliveRough = bsdf_sample_conductor(cEta, cK, 0.5f, nF, 0.4f,      \
                                                0.6f, rd, thr, pdf);           \
        OUT[4] = gpu_float4(rd, aliveRough ? 1.0f : 0.0f);                     \
        OUT[5] = thr;                                                          \
        rd = rdIn; thr = GpuSpectrum(1.0f);                                    \
        bsdf_sample_conductor(cEta, cK, 0.0f, nF, 0.4f, 0.6f, rd, thr, pdf);   \
        OUT[6] = gpu_float4(rd, 0.0f);                                         \
        OUT[7] = thr;                                                          \
        rd = rdIn; thr = GpuSpectrum(1.0f);                                    \
        bsdf_sample_dielectric(1.5f, nF, 0.9f, rd, thr, pdf);                  \
        OUT[8] = gpu_float4(rd, 0.0f);                                         \
        OUT[9] = thr;                                                          \
        rd = rdIn; thr = GpuSpectrum(1.0f);                                    \
        bsdf_sample_dielectric(1.5f, nF, 0.01f, rd, thr, pdf);                 \
        OUT[10] = gpu_float4(rd, 0.0f);                                        \
        gpu_float2 uv = env_equirect_uv(normalize(gpu_float3(0.3f, 0.5f,       \
                                                             -0.8f)));         \
        OUT[11] = gpu_float4(uv.x, uv.y, 0.0f, 0.0f);                          \
        OUT[12] = gpu_float4(tonemap_aces(gpu_float3(0.2f, 1.5f, 8.0f)),       \
                             0.0f);                                            \
        GpuWavelengths swl = spd_sample_visible(0.35f);                        \
        OUT[13] = swl.lambda;                                                  \
        OUT[14] = swl.pdf;                                                     \
        SpdPoly poly; poly.c0 = -0.0001f; poly.c1 = 0.1f; poly.c2 = -26.0f;    \
        OUT[15] = gpu_float4(spd_poly_eval(poly, 550.0f),                      \
                             spd_fr_complex1(0.7f, SpdComplex{ 1.5f, 2.0f }),  \
                             spd_fr_complex1(0.3f, SpdComplex{ 0.2f, 3.0f }),  \
                             spd_visible_pdf(550.0f));                         \
        GpuWavelengths swlT = spd_sample_visible(0.1f);                        \
        spd_terminate_secondary(swlT);                                         \
        OUT[16] = swlT.pdf;                                                    \
        OUT[17] = spd_dense_sample(SPD, SPD_OFF_CIE_Y, swl);                   \
        OUT[18] = spd_rgb_albedo_sample(R2S, gpu_float3(0.7f, 0.3f, 0.2f),     \
                                        swl);                                  \
        OUT[19] = spd_rgb_illuminant_sample(R2S, SPD,                          \
                                            gpu_float3(5.0f, 3.0f, 0.5f),     \
                                            swl);                              \
        OUT[20] = gpu_float4(spd_to_sensor_xyz(gpu_float4(0.9f, 1.4f, 0.6f,    \
                                                          1.1f),               \
                                               swl, SPD),                      \
                             0.0f);                                            \
        gpu_float3 wiE = normalize(gpu_float3(-0.2f, 0.4f, 0.9f));             \
        GpuSpectrum fE;                                                        \
        float pdfE;                                                            \
        bsdf_eval_lambert(gpu_float4(0.8f, 0.6f, 0.4f, 0.5f), nF, wiE, fE,     \
                          pdfE);                                               \
        OUT[21] = gpu_float4(fE.x, fE.y, fE.z, pdfE);                          \
        bsdf_eval_conductor(cEta, cK, 0.5f, nF, rdIn, wiE, fE, pdfE);          \
        OUT[22] = gpu_float4(fE.x, fE.y, fE.z, pdfE);                          \
    }

// The GPU-side probe kernel, single-source (concatenated for MSL after this
// header, #included by the generated CUDA registration TU). The parameter
// block is unused; PrimParams keeps the dispatch convention uniform.
#if defined(__METAL_VERSION__) || defined(__CUDACC__)
GPU_KERNEL(shared_probe)(GPU_KERNEL_PARAMS(PrimParams, P)
    GPU_BUFFER(gpu_float4, outv, 1)
    GPU_BUFFER(const float, r2s, 2)
    GPU_BUFFER(const float, spd, 3)
    GPU_TID_1D)
{
    if (GPU_GLOBAL_ID_X != 0)
        return;
    PROBE_BODY(outv, r2s, spd)
}
#endif

#endif  // TEST_SHARED_PROBE_H
