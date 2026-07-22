#pragma once
// Portable mesh loading for the renderer core (no backend headers, invariant
// I-4). Currently OBJ via tinyobjloader; the glTF/PLY paths in scene.cpp
// migrate here as the real loader is made host-portable (design doc M3).
#include <cstdint>
#include <string>
#include <vector>
#include <simd/simd.h>

// Material pulled from the OBJ's .mtl (used when the scene binds material -1,
// like Scene::loadModel): rendered as diffuse Kd, optionally textured.
struct MeshMaterial {
    simd_float3 kd = { 0.5f, 0.5f, 0.5f };
    std::string diffuseTexPath;  // joined with the .mtl directory; empty if none
    std::string name;
};

// Appends the OBJ's triangles (world-space baked through `transform`) to the
// unified vertex arrays: positions/normals/uvs share indices (vertices deduped
// on the OBJ index triple), so tris {i0, i1, i2, userData} stay valid when the
// BVH builder reorders them. Normals are zero when absent or when
// useVertexNormal is false (shading falls back to the geometric normal); uvs
// are (-1,-1) when absent, matching Scene::loadModel.
//
// userData per triangle: sceneMaterialId if >= 0 (whole mesh bound to one
// scene material, like the "material N" scene line); otherwise
// localMaterialBase + the face's MTL material index, with the MTL materials
// appended to outMaterials in index order.
bool loadObjMesh(const std::string& path, const simd_float4x4& transform,
                 int sceneMaterialId, uint32_t localMaterialBase,
                 bool useVertexNormal,
                 std::vector<simd_float3>& positions,
                 std::vector<simd_float3>& normals,
                 std::vector<simd_float2>& uvs,
                 std::vector<simd_uint4>& tris,
                 std::vector<MeshMaterial>& outMaterials);
