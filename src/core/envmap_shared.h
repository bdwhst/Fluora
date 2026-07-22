#ifndef CORE_ENVMAP_SHARED_H
#define CORE_ENVMAP_SHARED_H
// Environment-map math shared between renderer kernels and (later) CUDA device
// code — the same equirectangular mapping as math::equirectangular_dir_to_uv
// in mathUtils.h, so env lookups agree across backends. Self-contained because
// MSL sources are concatenated, not #included.

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef metal::float2 env_float2;
typedef metal::float3 env_float3;
#else
#include <simd/simd.h>
typedef simd_float2 env_float2;
typedef simd_float3 env_float3;
#endif

// dir must be normalized. uv.x wraps at the atan2 seam (sample with wrap
// addressing); uv.y spans the poles.
inline env_float2 env_equirect_uv(env_float3 dir)
{
#ifdef __METAL_VERSION__
    env_float2 uv = env_float2(metal::atan2(dir.z, dir.x), metal::asin(dir.y));
#else
    env_float2 uv = simd_make_float2(atan2f(dir.z, dir.x), asinf(dir.y));
#endif
    uv = uv * env_float2{ 0.1591f, 0.3183f } + env_float2{ 0.5f, 0.5f };
    return uv;
}

#endif
