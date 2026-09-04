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
// XF/ED are the light-sampling fixtures: two gpu_storage4x4 object-to-world
// transforms (XF[0] sphere, XF[1] cube) and a 4x2 envdist buffer built by
// core/lights.cpp buildEnvDistribution — the test uploads the same bytes for
// the GPU pass, so the dist1d binary searches compare identical floats in
// both personalities.
//
// Slots are named by the ProbeSlot enum: PROBE_BODY writes OUT[slot] and the
// test's identity checks read host[slot] through the same names, so inserting
// a slot renumbers both sides consistently at compile time. The test sizes
// its buffers from PROBE_SLOT_COUNT.

enum ProbeSlot {
    PROBE_RNG_A = 0,              // first four gpu_rand draws
    PROBE_RNG_B,                  // next four
    PROBE_LAMBERT_RD,             // lambert sample rd (w = pdf)
    PROBE_LAMBERT_THR,
    PROBE_CONDUCTOR_RD,           // rough conductor sample rd (w = alive)
    PROBE_CONDUCTOR_THR,
    PROBE_MIRROR_RD,              // mirror conductor sample rd
    PROBE_MIRROR_THR,
    PROBE_DIELECTRIC_REFRACT_RD,
    PROBE_DIELECTRIC_REFRACT_THR,
    PROBE_DIELECTRIC_REFLECT_RD,
    PROBE_EQUIRECT_UV,
    PROBE_ACES,
    PROBE_VISIBLE_LAMBDA,         // spd_sample_visible lambdas
    PROBE_VISIBLE_PDF,            // ...and their pdfs
    PROBE_SPD_SCALARS,            // sigmoid-poly / complex-Fresnel x2 / visible-pdf
    PROBE_TERMINATED_PDF,         // pdf after spd_terminate_secondary
    PROBE_DENSE_CIE_Y,
    PROBE_RGB_ALBEDO,
    PROBE_RGB_ILLUMINANT,
    PROBE_SENSOR_XYZ,
    PROBE_LAMBERT_EVAL,           // eval f (xyz) + pdf (w)
    PROBE_CONDUCTOR_EVAL,         // eval f (xyz) + pdf (w)
    PROBE_TRI_SAMPLE,             // tri sample p + pdfArea
    PROBE_TRI_WI,                 // tri wi + solid-angle pdf
    PROBE_TRI_PDFS,               // dist / reverse light_pdf_area /
                                  // light_pdf_area_tri / mis_power2
    PROBE_SPHERE_SAMPLE,          // sphere sample p + pdfArea
    PROBE_SPHERE_N,               // ...n + light_pdf_area_sphere
    PROBE_CUBE_SAMPLE,            // cube sample p + pdfArea (second face branch)
    PROBE_CUBE_N,                 // ...n + light_pdf_area_cube
    PROBE_ONE_SIDED_ZEROS,        // back-face sample/pdf zeros, mis_power2(0,0),
                                  // + first-branch cube face fingerprint
    PROBE_ENVDIST_SAMPLE,         // envdist_sample uv/pdf + envdist_pdf (z == w)
    PROBE_ENV_SAMPLE_DIR,         // env_sample_dir wi + pdf
    PROBE_ENV_PDF_DIR,            // env_pdf_dir (== PROBE_ENV_SAMPLE_DIR.w) /
                                  // marginal find_interval / equirect round trip
    PROBE_SLOT_COUNT
};

// RNG draws land in named locals first: constructor-argument evaluation order
// is unspecified in C++, and nvcc/MSVC need not match clang's left-to-right.
#define PROBE_BODY(OUT, R2S, SPD, XF, ED)                                      \
    {                                                                          \
        GpuRng rng;                                                            \
        rng.state = 12345u;                                                    \
        float r0 = gpu_rand(rng), r1 = gpu_rand(rng);                          \
        float r2 = gpu_rand(rng), r3 = gpu_rand(rng);                          \
        OUT[PROBE_RNG_A] = gpu_float4(r0, r1, r2, r3);                         \
        float r4 = gpu_rand(rng), r5 = gpu_rand(rng);                          \
        float r6 = gpu_rand(rng), r7 = gpu_rand(rng);                          \
        OUT[PROBE_RNG_B] = gpu_float4(r4, r5, r6, r7);                         \
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
        OUT[PROBE_LAMBERT_RD] = gpu_float4(rd, pdf);                           \
        OUT[PROBE_LAMBERT_THR] = thr;                                          \
        rd = rdIn; thr = GpuSpectrum(1.0f);                                    \
        bool aliveRough = bsdf_sample_conductor(cEta, cK, 0.5f, nF, 0.4f,      \
                                                0.6f, rd, thr, pdf);           \
        OUT[PROBE_CONDUCTOR_RD] = gpu_float4(rd, aliveRough ? 1.0f : 0.0f);    \
        OUT[PROBE_CONDUCTOR_THR] = thr;                                        \
        rd = rdIn; thr = GpuSpectrum(1.0f);                                    \
        bsdf_sample_conductor(cEta, cK, 0.0f, nF, 0.4f, 0.6f, rd, thr, pdf);   \
        OUT[PROBE_MIRROR_RD] = gpu_float4(rd, 0.0f);                           \
        OUT[PROBE_MIRROR_THR] = thr;                                           \
        rd = rdIn; thr = GpuSpectrum(1.0f);                                    \
        bsdf_sample_dielectric(1.5f, nF, 0.9f, rd, thr, pdf);                  \
        OUT[PROBE_DIELECTRIC_REFRACT_RD] = gpu_float4(rd, 0.0f);               \
        OUT[PROBE_DIELECTRIC_REFRACT_THR] = thr;                               \
        rd = rdIn; thr = GpuSpectrum(1.0f);                                    \
        bsdf_sample_dielectric(1.5f, nF, 0.01f, rd, thr, pdf);                 \
        OUT[PROBE_DIELECTRIC_REFLECT_RD] = gpu_float4(rd, 0.0f);               \
        gpu_float2 uv = env_equirect_uv(normalize(gpu_float3(0.3f, 0.5f,       \
                                                             -0.8f)));         \
        OUT[PROBE_EQUIRECT_UV] = gpu_float4(uv.x, uv.y, 0.0f, 0.0f);           \
        OUT[PROBE_ACES] = gpu_float4(tonemap_aces(gpu_float3(0.2f, 1.5f,       \
                                                             8.0f)),           \
                                     0.0f);                                    \
        GpuWavelengths swl = spd_sample_visible(0.35f);                        \
        OUT[PROBE_VISIBLE_LAMBDA] = swl.lambda;                                \
        OUT[PROBE_VISIBLE_PDF] = swl.pdf;                                      \
        SpdPoly poly; poly.c0 = -0.0001f; poly.c1 = 0.1f; poly.c2 = -26.0f;    \
        OUT[PROBE_SPD_SCALARS] =                                               \
            gpu_float4(spd_poly_eval(poly, 550.0f),                            \
                       spd_fr_complex1(0.7f, SpdComplex{ 1.5f, 2.0f }),        \
                       spd_fr_complex1(0.3f, SpdComplex{ 0.2f, 3.0f }),        \
                       spd_visible_pdf(550.0f));                               \
        GpuWavelengths swlT = spd_sample_visible(0.1f);                        \
        spd_terminate_secondary(swlT);                                         \
        OUT[PROBE_TERMINATED_PDF] = swlT.pdf;                                  \
        OUT[PROBE_DENSE_CIE_Y] = spd_dense_sample(SPD, SPD_OFF_CIE_Y, swl);    \
        OUT[PROBE_RGB_ALBEDO] = spd_rgb_albedo_sample(R2S,                     \
                                                      gpu_float3(0.7f, 0.3f,   \
                                                                 0.2f),        \
                                                      swl);                    \
        OUT[PROBE_RGB_ILLUMINANT] =                                            \
            spd_rgb_illuminant_sample(R2S, SPD, gpu_float3(5.0f, 3.0f, 0.5f),  \
                                      swl);                                    \
        OUT[PROBE_SENSOR_XYZ] =                                                \
            gpu_float4(spd_to_sensor_xyz(gpu_float4(0.9f, 1.4f, 0.6f, 1.1f),   \
                                         swl, SPD),                            \
                       0.0f);                                                  \
        gpu_float3 wiE = normalize(gpu_float3(-0.2f, 0.4f, 0.9f));             \
        GpuSpectrum fE;                                                        \
        float pdfE;                                                            \
        bsdf_eval_lambert(gpu_float4(0.8f, 0.6f, 0.4f, 0.5f), nF, wiE, fE,     \
                          pdfE);                                               \
        OUT[PROBE_LAMBERT_EVAL] = gpu_float4(fE.x, fE.y, fE.z, pdfE);          \
        bsdf_eval_conductor(cEta, cK, 0.5f, nF, rdIn, wiE, fE, pdfE);          \
        OUT[PROBE_CONDUCTOR_EVAL] = gpu_float4(fE.x, fE.y, fE.z, pdfE);        \
        gpu_float3 tv0 = gpu_float3(1.0f, 0.0f, 2.0f);                         \
        gpu_float3 tv1 = gpu_float3(3.0f, 0.5f, 2.0f);                         \
        gpu_float3 tv2 = gpu_float3(1.5f, 2.0f, 1.0f);                         \
        LightAreaSample ls = light_sample_tri(tv0, tv1, tv2,                   \
                                              gpu_float2(0.3f, 0.7f));         \
        OUT[PROBE_TRI_SAMPLE] = gpu_float4(ls.p, ls.pdfArea);                  \
        gpu_float3 piL = gpu_float3(0.0f, 3.0f, 5.0f);                         \
        gpu_float3 wiL;                                                        \
        float distL;                                                           \
        float pdfSA = light_area_to_solid_angle(ls, piL, wiL, distL);          \
        OUT[PROBE_TRI_WI] = gpu_float4(wiL, pdfSA);                            \
        OUT[PROBE_TRI_PDFS] =                                                  \
            gpu_float4(distL,                                                  \
                       light_pdf_area(ls.pdfArea, ls.p, ls.n, piL),            \
                       light_pdf_area_tri(tv0, tv1, tv2),                      \
                       mis_power2(pdfSA, 0.37f));                              \
        LightAreaSample ss = light_sample_sphere(XF[0],                        \
                                                 gpu_float3(6.0f, 2.0f,        \
                                                            -0.5f),            \
                                                 gpu_float2(0.6f, 0.25f));     \
        OUT[PROBE_SPHERE_SAMPLE] = gpu_float4(ss.p, ss.pdfArea);               \
        OUT[PROBE_SPHERE_N] = gpu_float4(ss.n, light_pdf_area_sphere(XF[0]));  \
        LightAreaSample cs = light_sample_cube(XF[1],                          \
                                               gpu_float3(0.55f, 0.3f,         \
                                                          0.8f));              \
        OUT[PROBE_CUBE_SAMPLE] = gpu_float4(cs.p, cs.pdfArea);                 \
        OUT[PROBE_CUBE_N] = gpu_float4(cs.n, light_pdf_area_cube(XF[1]));      \
        LightAreaSample cs2 = light_sample_cube(XF[1],                         \
                                                gpu_float3(0.02f, 0.5f,        \
                                                           0.5f));             \
        gpu_float3 piB = gpu_float3(3.0f, -2.0f, 0.0f);                        \
        OUT[PROBE_ONE_SIDED_ZEROS] =                                           \
            gpu_float4(light_area_to_solid_angle(ls, piB, wiL, distL),         \
                       light_pdf_area(ls.pdfArea, ls.p, ls.n, piB),            \
                       mis_power2(0.0f, 0.0f),                                 \
                       dot(cs2.n, gpu_float3(1.0f, 2.0f, 3.0f)));              \
        float pdfMap;                                                          \
        gpu_float2 uvE = envdist_sample(ED, 4u, 2u,                            \
                                        gpu_float2(0.83f, 0.4f), pdfMap);      \
        OUT[PROBE_ENVDIST_SAMPLE] = gpu_float4(uvE.x, uvE.y, pdfMap,           \
                                               envdist_pdf(ED, 4u, 2u, uvE));  \
        gpu_float3 wiEnv = gpu_float3(0.0f);                                   \
        float pdfEnv = 0.0f;                                                   \
        bool okEnv = env_sample_dir(ED, 4u, 2u, gpu_float2(0.15f, 0.65f),      \
                                    wiEnv, pdfEnv);                            \
        OUT[PROBE_ENV_SAMPLE_DIR] = gpu_float4(wiEnv,                          \
                                               okEnv ? pdfEnv : -1.0f);        \
        gpu_float2 uvRt = env_equirect_uv(                                     \
            env_equirect_dir(gpu_float2(0.7f, 0.4f)));                         \
        OUT[PROBE_ENV_PDF_DIR] =                                               \
            gpu_float4(env_pdf_dir(ED, 4u, 2u, wiEnv),                         \
                       (float)dist1d_find_interval(                            \
                           ED + ENVDIST_OFF_MARG_CDF(4u, 2u), 2u,              \
                           0.9f),                                              \
                       uvRt.x, uvRt.y);                                        \
    }

// The GPU-side probe kernel, single-source (concatenated for MSL after this
// header, #included by the generated CUDA registration TU). The parameter
// block is unused; PrimParams keeps the dispatch convention uniform.
#if defined(__METAL_VERSION__) || defined(__CUDACC__)
GPU_KERNEL(shared_probe, GPU_TID_1D)(GPU_KERNEL_PARAMS(PrimParams, P),
    GPU_BUFFER(gpu_float4, outv),
    GPU_BUFFER(const float, r2s),
    GPU_BUFFER(const float, spd),
    GPU_BUFFER(const gpu_storage4x4, xf),
    GPU_BUFFER(const float, env))
{
    if (GPU_GLOBAL_ID_X != 0)
        return;
    PROBE_BODY(outv, r2s, spd, xf, env)
}
#endif

#endif  // TEST_SHARED_PROBE_H
