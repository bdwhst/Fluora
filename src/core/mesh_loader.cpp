#include "mesh_loader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace {

// Key for unified-vertex dedupe: the OBJ (v, vt, vn) index triple. Cheaper and
// slightly more correct than Scene::loadModel's (pos, uv) float hashing — two
// corners sharing position+uv but not normal stay distinct vertices.
struct IndexTriple {
    int v, t, n;
    bool operator==(const IndexTriple& o) const { return v == o.v && t == o.t && n == o.n; }
};
struct IndexTripleHash {
    size_t operator()(const IndexTriple& k) const
    {
        size_t h = (size_t)(uint32_t)k.v;
        h = h * 0x9E3779B97F4A7C15ull + (uint32_t)k.t;
        h = h * 0x9E3779B97F4A7C15ull + (uint32_t)k.n;
        return h;
    }
};

} // namespace

bool loadObjMesh(const std::string& path, const simd_float4x4& transform,
                 int sceneMaterialId, uint32_t localMaterialBase,
                 bool useVertexNormal,
                 std::vector<simd_float3>& positions,
                 std::vector<simd_float3>& normals,
                 std::vector<simd_float2>& uvs,
                 std::vector<simd_uint4>& tris,
                 std::vector<MeshMaterial>& outMaterials)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objMaterials;
    std::string warn, err;
    std::string dir = path.substr(0, path.find_last_of('/') + 1);
    if (!tinyobj::LoadObj(&attrib, &shapes, &objMaterials, &warn, &err, path.c_str(),
                          dir.c_str(), /*triangulate=*/true)) {
        std::cout << "core: OBJ load failed for " << path << ": " << err << "\n";
        return false;
    }

    if (sceneMaterialId < 0) {
        for (const auto& m : objMaterials) {
            MeshMaterial mm;
            mm.kd = simd_make_float3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
            if (!m.diffuse_texname.empty())
                mm.diffuseTexPath = dir + m.diffuse_texname;
            mm.name = m.name;
            outMaterials.push_back(std::move(mm));
        }
    }

    // Normals transform with the inverse transpose (matches invTranspose on
    // analytic objects); normalized at interpolation time on device.
    simd_float4x4 invT = simd_transpose(simd_inverse(transform));

    std::unordered_map<IndexTriple, uint32_t, IndexTripleHash> vertexSet;
    size_t triCount = 0;
    for (const auto& shape : shapes) {
        for (size_t f = 0; f + 2 < shape.mesh.indices.size(); f += 3) {
            uint32_t matId = (uint32_t)sceneMaterialId;
            if (sceneMaterialId < 0) {
                int local = shape.mesh.material_ids[f / 3];
                matId = localMaterialBase + (uint32_t)std::max(local, 0);
            }
            uint32_t corner[3];
            for (int k = 0; k < 3; k++) {
                tinyobj::index_t idx = shape.mesh.indices[f + k];
                IndexTriple key{ idx.vertex_index,
                                 idx.texcoord_index,
                                 useVertexNormal ? idx.normal_index : -1 };
                auto it = vertexSet.find(key);
                if (it == vertexSet.end()) {
                    simd_float4 p = { attrib.vertices[3 * idx.vertex_index + 0],
                                      attrib.vertices[3 * idx.vertex_index + 1],
                                      attrib.vertices[3 * idx.vertex_index + 2], 1.0f };
                    simd_float3 n = { 0, 0, 0 };
                    if (useVertexNormal && idx.normal_index >= 0) {
                        simd_float4 nn = { attrib.normals[3 * idx.normal_index + 0],
                                           attrib.normals[3 * idx.normal_index + 1],
                                           attrib.normals[3 * idx.normal_index + 2], 0.0f };
                        n = simd_mul(invT, nn).xyz;
                    }
                    simd_float2 uv = { -1.0f, -1.0f };
                    if (idx.texcoord_index >= 0)
                        uv = simd_make_float2(attrib.texcoords[2 * idx.texcoord_index + 0],
                                              attrib.texcoords[2 * idx.texcoord_index + 1]);
                    it = vertexSet.emplace(key, (uint32_t)positions.size()).first;
                    positions.push_back(simd_mul(transform, p).xyz);
                    normals.push_back(n);
                    uvs.push_back(uv);
                }
                corner[k] = it->second;
            }
            tris.push_back(simd_make_uint4(corner[0], corner[1], corner[2], matId));
            triCount++;
        }
    }
    std::cout << "core: loaded " << path << " (" << triCount << " tris, "
              << outMaterials.size() << " mtl materials)\n";
    return true;
}
