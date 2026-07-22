#include "mesh_loader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>

bool loadObjMesh(const std::string& path, const simd_float4x4& transform,
                 uint32_t userData,
                 std::vector<simd_float3>& positions,
                 std::vector<simd_uint4>& tris)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objMaterials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &objMaterials, &warn, &err, path.c_str(),
                          nullptr, /*triangulate=*/true)) {
        std::cout << "core: OBJ load failed for " << path << ": " << err << "\n";
        return false;
    }
    uint32_t baseVertex = (uint32_t)positions.size();
    for (size_t i = 0; i + 2 < attrib.vertices.size(); i += 3) {
        simd_float4 v = { attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2], 1.0f };
        positions.push_back(simd_mul(transform, v).xyz);
    }
    size_t triCount = 0;
    for (const auto& shape : shapes) {
        for (size_t f = 0; f + 2 < shape.mesh.indices.size(); f += 3) {
            tris.push_back(simd_make_uint4(
                baseVertex + (uint32_t)shape.mesh.indices[f + 0].vertex_index,
                baseVertex + (uint32_t)shape.mesh.indices[f + 1].vertex_index,
                baseVertex + (uint32_t)shape.mesh.indices[f + 2].vertex_index,
                userData));
            triCount++;
        }
    }
    std::cout << "core: loaded " << path << " (" << triCount << " tris)\n";
    return true;
}
