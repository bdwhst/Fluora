#ifndef CORE_ENVMAP_SHARED_H
#define CORE_ENVMAP_SHARED_H
// Environment-map math shared between renderer kernels and CUDA device code —
// the same equirectangular mapping as math::equirectangular_dir_to_uv in
// mathUtils.h, so env lookups agree across backends. Single-source via the
// gpu_portable shim (docs/portable-device-code.md).

#ifndef __METAL_VERSION__
#include "../rhi/gpu_portable.h"
#endif

// dir must be normalized. uv.x wraps at the atan2 seam (sample with wrap
// addressing); uv.y spans the poles.
GPU_FN inline gpu_float2 env_equirect_uv(gpu_float3 dir)
{
    gpu_float2 uv = gpu_float2(atan2(dir.z, dir.x), asin(dir.y));
    uv = uv * gpu_float2(0.1591f, 0.3183f) + gpu_float2(0.5f, 0.5f);
    return uv;
}

#endif
