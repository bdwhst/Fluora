#pragma once
// Host-side glue between glm value math and the gpu_storage layout types that
// device-visible structs carry. The loaders compute in glm — the same
// functions scene.cpp/utilityCore use — so host math is identical across
// backends in M4; only the storage spelling differs per platform.
#include <glm/glm.hpp>

#include "../rhi/gpu_portable.h"

inline gpu_storage3 hostStore3(const glm::vec3& v)
{
#if defined(__APPLE__)
    return simd_make_float3(v.x, v.y, v.z);
#else
    return gpu_storage3{ v.x, v.y, v.z };
#endif
}

inline gpu_storage4x4 hostStore4x4(const glm::mat4& m)
{
#if defined(__APPLE__)
    return simd_matrix(simd_make_float4(m[0][0], m[0][1], m[0][2], m[0][3]),
                       simd_make_float4(m[1][0], m[1][1], m[1][2], m[1][3]),
                       simd_make_float4(m[2][0], m[2][1], m[2][2], m[2][3]),
                       simd_make_float4(m[3][0], m[3][1], m[3][2], m[3][3]));
#else
    return m;   // gpu_storage4x4 is glm::mat4 off-Apple
#endif
}
