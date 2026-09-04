#include "mesh_loader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#define TINYPLY_IMPLEMENTATION
#include <tinyply.h>

#include <glm/gtc/matrix_inverse.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "host_math.h"

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

bool loadObjMesh(const std::string& path, const glm::mat4& transform,
                 int sceneMaterialId, uint32_t localMaterialBase,
                 bool useVertexNormal,
                 std::vector<gpu_storage3>& positions,
                 std::vector<gpu_storage3>& normals,
                 std::vector<gpu_float2>& uvs,
                 std::vector<gpu_uint4>& tris,
                 std::vector<MeshMaterial>& outMaterials)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objMaterials;
    std::string warn, err;
    std::string dir;
    if (auto slash = path.find_last_of("/\\"); slash != std::string::npos)
        dir = path.substr(0, slash + 1);
    if (!tinyobj::LoadObj(&attrib, &shapes, &objMaterials, &warn, &err, path.c_str(),
                          dir.c_str(), /*triangulate=*/true)) {
        std::cout << "core: OBJ load failed for " << path << ": " << err << "\n";
        return false;
    }

    if (sceneMaterialId < 0) {
        for (const auto& m : objMaterials) {
            MeshMaterial mm;
            mm.kd = glm::vec3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
            if (!m.diffuse_texname.empty())
                mm.diffuseTexPath = dir + m.diffuse_texname;
            mm.name = m.name;
            outMaterials.push_back(std::move(mm));
        }
    }

    // Normals transform with the inverse transpose (glm::inverseTranspose,
    // like Scene::loadModel); normalized at interpolation time on device.
    glm::mat4 invT = glm::inverseTranspose(transform);

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
                    glm::vec4 p(attrib.vertices[3 * idx.vertex_index + 0],
                                attrib.vertices[3 * idx.vertex_index + 1],
                                attrib.vertices[3 * idx.vertex_index + 2], 1.0f);
                    glm::vec3 n(0.0f);
                    if (useVertexNormal && idx.normal_index >= 0) {
                        glm::vec4 nn(attrib.normals[3 * idx.normal_index + 0],
                                     attrib.normals[3 * idx.normal_index + 1],
                                     attrib.normals[3 * idx.normal_index + 2], 0.0f);
                        n = glm::vec3(invT * nn);
                    }
                    gpu_float2 uv(-1.0f, -1.0f);
                    if (idx.texcoord_index >= 0)
                        uv = gpu_float2(attrib.texcoords[2 * idx.texcoord_index + 0],
                                        attrib.texcoords[2 * idx.texcoord_index + 1]);
                    it = vertexSet.emplace(key, (uint32_t)positions.size()).first;
                    positions.push_back(hostStore3(glm::vec3(transform * p)));
                    normals.push_back(hostStore3(n));
                    uvs.push_back(uv);
                }
                corner[k] = it->second;
            }
            tris.push_back(gpu_uint4(corner[0], corner[1], corner[2], matId));
            triCount++;
        }
    }
    std::cout << "core: loaded " << path << " (" << triCount << " tris, "
              << outMaterials.size() << " mtl materials)\n";
    return true;
}

// ---- PLY ----------------------------------------------------------------

bool loadPlyMesh(const std::string& path, const glm::mat4& transform,
                 uint32_t materialId,
                 std::vector<gpu_storage3>& positions,
                 std::vector<gpu_storage3>& normals,
                 std::vector<gpu_float2>& uvs,
                 std::vector<gpu_uint4>& tris)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        std::cout << "core: cannot open PLY " << path << "\n";
        return false;
    }
    tinyply::PlyFile file;
    std::shared_ptr<tinyply::PlyData> verts, norms, texcoords, faces;
    try {
        file.parse_header(stream);
        verts = file.request_properties_from_element("vertex", { "x", "y", "z" });
        // Optional attributes: tinyply throws when an element/property is
        // missing, so each is requested on its own.
        try { norms = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
        catch (const std::exception&) {}
        try { texcoords = file.request_properties_from_element("vertex", { "u", "v" }); }
        catch (const std::exception&) {}
        faces = file.request_properties_from_element("face", { "vertex_indices" }, 0);
        file.read(stream);
    } catch (const std::exception& e) {
        std::cout << "core: PLY load failed for " << path << ": " << e.what() << "\n";
        return false;
    }
    if (!verts || verts->t != tinyply::Type::FLOAT32 || !faces) {
        std::cout << "core: PLY " << path << " needs float32 vertices and faces\n";
        return false;
    }
    if (norms && norms->t != tinyply::Type::FLOAT32)
        norms.reset();
    if (texcoords && texcoords->t != tinyply::Type::FLOAT32)
        texcoords.reset();
    if (faces->t != tinyply::Type::INT32 && faces->t != tinyply::Type::UINT32) {
        std::cout << "core: PLY " << path << " has non-32-bit face indices\n";
        return false;
    }
    // With no list-size hint tinyply packs variable-length lists back to back;
    // only all-triangle files have exactly 3 indices per face.
    if (faces->buffer.size_bytes() != faces->count * 3 * sizeof(uint32_t)) {
        std::cout << "core: PLY " << path << " has non-triangle faces\n";
        return false;
    }

    glm::mat4 invT = glm::inverseTranspose(transform);
    const uint32_t base = (uint32_t)positions.size();
    const float* p = reinterpret_cast<const float*>(verts->buffer.get());
    const float* n = norms ? reinterpret_cast<const float*>(norms->buffer.get()) : nullptr;
    const float* t = texcoords ? reinterpret_cast<const float*>(texcoords->buffer.get()) : nullptr;
    for (size_t i = 0; i < verts->count; i++) {
        glm::vec4 pos(p[3 * i], p[3 * i + 1], p[3 * i + 2], 1.0f);
        positions.push_back(hostStore3(glm::vec3(transform * pos)));
        glm::vec3 nn(0.0f);
        if (n)
            nn = glm::vec3(invT * glm::vec4(n[3 * i], n[3 * i + 1], n[3 * i + 2], 0.0f));
        normals.push_back(hostStore3(nn));
        uvs.push_back(t ? gpu_float2(t[2 * i], t[2 * i + 1]) : gpu_float2(-1.0f, -1.0f));
    }
    const uint32_t* idx = reinterpret_cast<const uint32_t*>(faces->buffer.get());
    for (size_t f = 0; f < faces->count; f++) {
        uint32_t a = idx[3 * f], b = idx[3 * f + 1], c = idx[3 * f + 2];
        if (a >= verts->count || b >= verts->count || c >= verts->count) {
            std::cout << "core: PLY " << path << " face index out of range\n";
            return false;
        }
        tris.push_back(gpu_uint4(base + a, base + b, base + c, materialId));
    }
    std::cout << "core: loaded " << path << " (" << faces->count << " tris)\n";
    return true;
}

// ---- inline (.json) ------------------------------------------------------

bool appendInlineMesh(const std::vector<float>& xyz, const std::vector<uint32_t>& indices,
                      const glm::mat4& transform, uint32_t materialId,
                      std::vector<gpu_storage3>& positions,
                      std::vector<gpu_storage3>& normals,
                      std::vector<gpu_float2>& uvs,
                      std::vector<gpu_uint4>& tris,
                      std::string& err)
{
    if (xyz.size() % 3 != 0 || indices.size() % 3 != 0) {
        err = "inline mesh: VERTICES/INDICES counts must be multiples of three";
        return false;
    }
    const uint32_t base = (uint32_t)positions.size();
    const uint32_t count = (uint32_t)(xyz.size() / 3);
    for (uint32_t i = 0; i < count; i++) {
        glm::vec4 pos(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], 1.0f);
        positions.push_back(hostStore3(glm::vec3(transform * pos)));
        normals.push_back(hostStore3(glm::vec3(0.0f)));
        uvs.push_back(gpu_float2(-1.0f, -1.0f));
    }
    for (size_t f = 0; f + 2 < indices.size(); f += 3) {
        uint32_t a = indices[f], b = indices[f + 1], c = indices[f + 2];
        if (a >= count || b >= count || c >= count) {
            err = "inline mesh: index out of range";
            return false;
        }
        tris.push_back(gpu_uint4(base + a, base + b, base + c, materialId));
    }
    return true;
}
