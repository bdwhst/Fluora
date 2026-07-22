#pragma once
// Portable mesh loading for the renderer core (no backend headers, invariant
// I-4). Currently OBJ via tinyobjloader; the glTF/PLY paths in scene.cpp
// migrate here as the real loader is made host-portable (design doc M3).
#include <cstdint>
#include <string>
#include <vector>
#include <simd/simd.h>

// Appends the OBJ's triangles (world-space baked through `transform`) to
// positions/tris. tris are {i0, i1, i2, userData}. Returns false on failure.
bool loadObjMesh(const std::string& path, const simd_float4x4& transform,
                 uint32_t userData,
                 std::vector<simd_float3>& positions,
                 std::vector<simd_uint4>& tris);
