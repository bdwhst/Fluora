#ifndef CORE_TONEMAP_SHARED_H
#define CORE_TONEMAP_SHARED_H
// ACES filmic curve shared between the device present/tonemap kernel and the
// host PNG writer — the same curve as util_postprocess_ACESFilm in
// sendImageToPBO (no extra gamma, matching the CUDA preview). Single-source
// via the gpu_portable shim (docs/portable-device-code.md).

#ifndef __METAL_VERSION__
#include "../rhi/gpu_portable.h"
#endif

GPU_FN inline gpu_float3 tonemap_aces(gpu_float3 x)
{
    float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    gpu_float3 v = (x * (a * x + b)) / (x * (c * x + d) + e);
    return clamp(v, 0.0f, 1.0f);
}

#endif
