#ifndef CORE_TONEMAP_SHARED_H
#define CORE_TONEMAP_SHARED_H
// ACES filmic curve shared between the device present/tonemap kernel and the
// host PNG writer — the same curve as util_postprocess_ACESFilm in
// sendImageToPBO (no extra gamma, matching the CUDA preview). Compiled by host
// clang (simd types) and the Metal compiler (invariant I-3); self-contained
// because MSL sources are concatenated, not #included.

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef metal::float3 tm_float3;
#else
#include <simd/simd.h>
typedef simd_float3 tm_float3;
#endif

inline tm_float3 tonemap_aces(tm_float3 x)
{
    float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    tm_float3 v = (x * (a * x + b)) / (x * (c * x + d) + e);
#ifdef __METAL_VERSION__
    return metal::clamp(v, 0.0f, 1.0f);
#else
    return simd_clamp(v, simd_make_float3(0.0f, 0.0f, 0.0f),
                      simd_make_float3(1.0f, 1.0f, 1.0f));
#endif
}

#endif
