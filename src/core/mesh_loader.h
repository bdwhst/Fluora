#pragma once
// Portable mesh loading for the renderer core (no backend headers, invariant
// I-4). OBJ via tinyobjloader, PLY via tinyply, plus the .json format's inline
// vertex/index lists. The glTF path in scene.cpp is dead code and does not
// migrate.
#include <cstdint>
#include <string>
#include <vector>
#include <glm/glm.hpp>

#include "../rhi/gpu_portable.h"

// Material pulled from the OBJ's .mtl (used when the scene binds material -1,
// like Scene::loadModel): rendered as diffuse Kd, optionally textured.
struct MeshMaterial {
    glm::vec3 kd { 0.5f, 0.5f, 0.5f };
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
bool loadObjMesh(const std::string& path, const glm::mat4& transform,
                 int sceneMaterialId, uint32_t localMaterialBase,
                 bool useVertexNormal,
                 std::vector<gpu_storage3>& positions,
                 std::vector<gpu_storage3>& normals,
                 std::vector<gpu_float2>& uvs,
                 std::vector<gpu_uint4>& tris,
                 std::vector<MeshMaterial>& outMaterials);

// Appends a PLY mesh (Scene::loadPly's subset: float x/y/z, optional nx/ny/nz
// and u/v, triangle faces with int32/uint32 vertex_indices) bound to one
// scene material. PLY vertices are already unified, so they append 1:1; no
// dedupe. Fails on non-triangle faces.
bool loadPlyMesh(const std::string& path, const glm::mat4& transform,
                 uint32_t materialId,
                 std::vector<gpu_storage3>& positions,
                 std::vector<gpu_storage3>& normals,
                 std::vector<gpu_float2>& uvs,
                 std::vector<gpu_uint4>& tris);

// Appends an inline mesh (.json "model_inline": flat xyz vertex list, flat
// triangle index list) bound to one scene material. No normals or uvs (zero
// / (-1,-1)), like Scene::loadJSON. Fails on an index out of range or a
// count that is not a multiple of three.
bool appendInlineMesh(const std::vector<float>& xyz, const std::vector<uint32_t>& indices,
                      const glm::mat4& transform, uint32_t materialId,
                      std::vector<gpu_storage3>& positions,
                      std::vector<gpu_storage3>& normals,
                      std::vector<gpu_float2>& uvs,
                      std::vector<gpu_uint4>& tris,
                      std::string& err);
