#ifndef RHI_SHARED_PROBE_H
#define RHI_SHARED_PROBE_H
// Probe body for SharedHostTest: evaluates the shared BSDF/RNG/env/tonemap
// functions on fixed inputs, writing 13 float4 slots. Spelled once as a macro
// so the exact same text runs in the host personality (#include) and the MSL
// personality (concatenated before the probe kernel). Uses only gpu_* types
// and shared functions — this file is itself portable device code.
//
// Slots: 0-1 rng draws; 2-3 lambert rd/thr; 4-5 rough conductor rd/thr;
// 6-7 mirror conductor rd/thr; 8-9 dielectric refract rd/thr; 10 dielectric
// reflect rd; 11 equirect uv; 12 ACES. The test sizes its buffers from
// PROBE_SLOTS — bump it with every added slot.

#define PROBE_SLOTS 13

// RNG draws land in named locals first: constructor-argument evaluation order
// is unspecified in C++, and nvcc/MSVC need not match clang's left-to-right.
#define PROBE_BODY(OUT)                                                        \
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
        gpu_float3 rd, thr;                                                    \
        rd = rdIn; thr = gpu_float3(1.0f);                                     \
        bsdf_sample_lambert(gpu_float3(0.8f, 0.6f, 0.4f), nF, 0.3f, 0.7f, rd,  \
                            thr);                                              \
        OUT[2] = gpu_float4(rd, 0.0f);                                         \
        OUT[3] = gpu_float4(thr, 0.0f);                                        \
        rd = rdIn; thr = gpu_float3(1.0f);                                     \
        bool aliveRough = bsdf_sample_conductor(gpu_float3(0.9f, 0.8f, 0.7f),  \
                                                0.5f, nF, 0.4f, 0.6f, rd, thr);\
        OUT[4] = gpu_float4(rd, aliveRough ? 1.0f : 0.0f);                     \
        OUT[5] = gpu_float4(thr, 0.0f);                                        \
        rd = rdIn; thr = gpu_float3(1.0f);                                     \
        bsdf_sample_conductor(gpu_float3(0.9f, 0.8f, 0.7f), 0.0f, nF, 0.4f,    \
                              0.6f, rd, thr);                                  \
        OUT[6] = gpu_float4(rd, 0.0f);                                         \
        OUT[7] = gpu_float4(thr, 0.0f);                                        \
        rd = rdIn; thr = gpu_float3(1.0f);                                     \
        bsdf_sample_dielectric(gpu_float3(0.95f, 0.95f, 0.95f), 1.5f, nF, 0.9f,\
                               rd, thr);                                       \
        OUT[8] = gpu_float4(rd, 0.0f);                                         \
        OUT[9] = gpu_float4(thr, 0.0f);                                        \
        rd = rdIn; thr = gpu_float3(1.0f);                                     \
        bsdf_sample_dielectric(gpu_float3(0.95f, 0.95f, 0.95f), 1.5f, nF,      \
                               0.01f, rd, thr);                                \
        OUT[10] = gpu_float4(rd, 0.0f);                                        \
        gpu_float2 uv = env_equirect_uv(normalize(gpu_float3(0.3f, 0.5f,       \
                                                             -0.8f)));         \
        OUT[11] = gpu_float4(uv.x, uv.y, 0.0f, 0.0f);                          \
        OUT[12] = gpu_float4(tonemap_aces(gpu_float3(0.2f, 1.5f, 8.0f)),       \
                             0.0f);                                            \
    }

#endif
