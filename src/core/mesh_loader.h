#pragma once
// Portable mesh loading for the renderer core (no backend headers, invariant
// I-4). OBJ via tinyobjloader, PLY via tinyply, glTF/GLB via tinygltf
// (gltf_loader.cpp), plus the .json format's inline vertex/index lists.
#include <cstdint>
#include <string>
#include <vector>
#include <glm/glm.hpp>

#include "../rhi/gpu_portable.h"
#include "image_loader.h"

// A material carried by a model file, appended to the scene list when the
// scene binds material -1 (like Scene::loadModel). OBJ .mtl entries are
// diffuse Kd + optional texture; glTF materials also map to the other core
// types (see loadGltfMesh).
enum class MeshMaterialType { Diffuse, Emissive, Dielectric, Conductor };
struct MeshMaterial {
    MeshMaterialType type = MeshMaterialType::Diffuse;
    glm::vec3 kd { 0.5f, 0.5f, 0.5f };
    float roughness = 0.0f;      // conductor GGX alpha
    float ior = 1.5f;            // dielectric eta
    float emittance = 0.0f;      // emissive scale on kd
    std::string diffuseTexPath;  // joined with the model's directory; empty if none
    LdrImage embeddedTex;        // decoded in-file image (glTF); width 0 = none
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

// Appends a glTF or GLB file's triangle meshes (gltf_loader.cpp): walks the
// default scene's node hierarchy, bakes node-global * `transform` into the
// unified vertex arrays, and reads POSITION / NORMAL / TEXCOORD_0 (indexed
// u8/u16/u32 or non-indexed TRIANGLES primitives).
//
// userData per triangle: sceneMaterialId if >= 0; otherwise localMaterialBase
// + the primitive's glTF material index, with the glTF materials appended to
// outMaterials in index order (a trailing default is added if any primitive
// has no material). Material mapping, per the metallic-roughness model and
// the intent of the old scene.cpp glTF path:
//   KHR_materials_transmission / _volume or alphaMode BLEND -> Dielectric
//     (eta from KHR_materials_ior, default 1.5);
//   else metallicFactor >= 0.5 -> Conductor in reflectance mode (kd =
//     baseColorFactor, GGX alpha = roughnessFactor^2) — the base-color
//     texture is carried but conductor shading currently ignores it;
//   else -> Diffuse (baseColorFactor x baseColor texture).
// Emissive factors, normal/metallicRoughness textures and per-texel metal
// masks are ignored (single-lobe materials; normal maps are parse-only in
// the core). glTF images (embedded or external) are decoded by tinygltf and
// returned in MeshMaterial::embeddedTex, bottom-row-first like every
// LdrImage (the loader pins stbi's global flip flag rather than inheriting
// whatever loaded before it); uvs are converted to the matching bottom-left
// origin (v -> 1-v) with any base-color KHR_texture_transform baked in first.
bool loadGltfMesh(const std::string& path, const glm::mat4& transform,
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
