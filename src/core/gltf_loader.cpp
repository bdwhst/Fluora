// glTF/GLB loading for the renderer core (loadGltfMesh in mesh_loader.h).
// Host-only: tinygltf parses the file and decodes every referenced image
// (embedded buffers and external files alike) through the stb_image
// implementation in src/stb.cpp. Kept out of mesh_loader.cpp so the tinygltf
// implementation macro lives in exactly one translation unit.
#include "mesh_loader.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <unordered_set>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include <stb_image.h>

#include "host_math.h"

namespace {

// KHR_texture_transform on the base-color texture, baked into the vertex uvs
// at load (exact for an affine transform with repeat addressing; every
// primitive here owns its vertices, so per-material baking is safe).
struct UvTransform {
    glm::vec2 offset { 0.0f, 0.0f };
    glm::vec2 scale { 1.0f, 1.0f };
    float rotation = 0.0f;  // radians, clockwise per the extension spec
    bool active = false;
};

UvTransform uvTransformOf(const tinygltf::Material& gm)
{
    UvTransform t;
    const auto& ext = gm.pbrMetallicRoughness.baseColorTexture.extensions;
    auto it = ext.find("KHR_texture_transform");
    if (it == ext.end())
        return t;
    const tinygltf::Value& v = it->second;
    if (v.Has("offset")) {
        t.offset.x = (float)v.Get("offset").Get(0).GetNumberAsDouble();
        t.offset.y = (float)v.Get("offset").Get(1).GetNumberAsDouble();
    }
    if (v.Has("scale")) {
        t.scale.x = (float)v.Get("scale").Get(0).GetNumberAsDouble();
        t.scale.y = (float)v.Get("scale").Get(1).GetNumberAsDouble();
    }
    if (v.Has("rotation"))
        t.rotation = (float)v.Get("rotation").GetNumberAsDouble();
    t.active = true;
    return t;
}

// Node-local transform: `matrix` when present, else TRS composed like
// GLTFNodeGetLocalTransform in scene.cpp (T * R * S).
glm::mat4 nodeLocalTransform(const tinygltf::Node& node)
{
    if (node.matrix.size() == 16) {
        glm::mat4 m;
        for (int c = 0; c < 4; c++)
            for (int r = 0; r < 4; r++)
                m[c][r] = (float)node.matrix[c * 4 + r];  // glTF is column-major
        return m;
    }
    glm::mat4 m(1.0f);
    if (node.translation.size() == 3)
        m = glm::translate(m, glm::vec3((float)node.translation[0],
                                        (float)node.translation[1],
                                        (float)node.translation[2]));
    if (node.rotation.size() == 4)  // glTF quaternion is x, y, z, w
        m = m * glm::toMat4(glm::quat((float)node.rotation[3], (float)node.rotation[0],
                                      (float)node.rotation[1], (float)node.rotation[2]));
    if (node.scale.size() == 3)
        m = glm::scale(m, glm::vec3((float)node.scale[0], (float)node.scale[1],
                                    (float)node.scale[2]));
    return m;
}

// Strided accessor reader: returns a pointer to element i of the accessor.
const unsigned char* accessorElem(const tinygltf::Model& model,
                                  const tinygltf::Accessor& acc, size_t elemBytes, size_t i)
{
    const tinygltf::BufferView& bv = model.bufferViews[acc.bufferView];
    size_t stride = bv.byteStride ? bv.byteStride : elemBytes;
    return model.buffers[bv.buffer].data.data() + bv.byteOffset + acc.byteOffset + i * stride;
}

// glTF material -> MeshMaterial, per the mapping documented in mesh_loader.h.
// The decoded base-color image is copied into the MeshMaterial, so materials
// sharing one image each get their own copy (rare; simpler than an
// image-index indirection through the scene loader).
MeshMaterial convertMaterial(const tinygltf::Model& model, const tinygltf::Material& gm)
{
    MeshMaterial m;
    m.name = gm.name;
    const auto& pbr = gm.pbrMetallicRoughness;
    m.kd = glm::vec3((float)pbr.baseColorFactor[0], (float)pbr.baseColorFactor[1],
                     (float)pbr.baseColorFactor[2]);
    if (gm.extensions.count("KHR_materials_transmission")
        || gm.extensions.count("KHR_materials_volume") || gm.alphaMode == "BLEND") {
        m.type = MeshMaterialType::Dielectric;
        m.kd = glm::vec3(0.98f);  // like the old scene.cpp glTF path
        auto it = gm.extensions.find("KHR_materials_ior");
        m.ior = it != gm.extensions.end() && it->second.Has("ior")
                    ? (float)it->second.Get("ior").GetNumberAsDouble()
                    : 1.5f;
        return m;
    }
    // Conductor only when the file says "uniformly metallic": real assets
    // usually set metallicFactor 1 and mask per texel through the
    // metallicRoughness texture, which a single-lobe material can't honor —
    // those render better as textured diffuse than as untextured metal.
    if (pbr.metallicFactor >= 0.5 && pbr.metallicRoughnessTexture.index < 0) {
        m.type = MeshMaterialType::Conductor;
        // glTF roughness is perceptual; GGX alpha = roughness^2.
        float r = (float)pbr.roughnessFactor;
        m.roughness = r * r;
        return m;
    }
    if (pbr.baseColorTexture.index >= 0) {
        const tinygltf::Texture& tex = model.textures[pbr.baseColorTexture.index];
        if (pbr.baseColorTexture.texCoord != 0) {
            std::cout << "core: glTF material '" << gm.name
                      << "': only TEXCOORD_0 is supported, dropping texture\n";
        } else if (tex.source >= 0 && tex.source < (int)model.images.size()) {
            const tinygltf::Image& img = model.images[tex.source];
            if (img.width > 0 && img.component == 4 && img.bits == 8) {
                m.embeddedTex.width = img.width;
                m.embeddedTex.height = img.height;
                m.embeddedTex.rgba = img.image;  // top-row-first, see mesh_loader.h
            } else {
                std::cout << "core: glTF material '" << gm.name
                          << "': unsupported image format (" << img.component << "x"
                          << img.bits << " bit), dropping texture\n";
            }
        }
    }
    return m;
}

} // namespace

bool loadGltfMesh(const std::string& path, const glm::mat4& transform,
                  int sceneMaterialId, uint32_t localMaterialBase,
                  bool useVertexNormal,
                  std::vector<gpu_storage3>& positions,
                  std::vector<gpu_storage3>& normals,
                  std::vector<gpu_float2>& uvs,
                  std::vector<gpu_uint4>& tris,
                  std::vector<MeshMaterial>& outMaterials)
{
    // tinygltf decodes images through stbi, whose flip-on-load flag is global
    // state that loadLdrImage/loadHdrImage set and never clear — whether it is
    // set here depends on what loaded before us (a preview scene hot-swap
    // arrives with it set, a fresh launch doesn't). Pin it: decode
    // bottom-row-first like every other image in the pipeline (the LdrImage
    // contract) and convert the uvs below from glTF's top-left origin to
    // match (v -> 1-v), which lands on the exact same sample points.
    stbi_set_flip_vertically_on_load(1);

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool isGlb = path.size() >= 4
                 && (path.compare(path.size() - 4, 4, ".glb") == 0
                     || path.compare(path.size() - 4, 4, ".GLB") == 0);
    bool ok = isGlb ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
                    : loader.LoadASCIIFromFile(&model, &err, &warn, path);
    if (!warn.empty())
        std::cout << "core: glTF warning for " << path << ": " << warn << "\n";
    if (!ok) {
        std::cout << "core: failed to parse glTF " << path
                  << (err.empty() ? "" : ": " + err) << "\n";
        return false;
    }

    // Materials, appended in glTF index order so primitive.material maps to
    // localMaterialBase + index. A primitive without a material gets a
    // trailing default-diffuse entry.
    bool needDefault = false;
    if (sceneMaterialId < 0) {
        for (const auto& gm : model.materials)
            outMaterials.push_back(convertMaterial(model, gm));
        for (const auto& mesh : model.meshes)
            for (const auto& prim : mesh.primitives)
                if (prim.material < 0)
                    needDefault = true;
        if (needDefault) {
            MeshMaterial def;
            def.name = "gltf default";
            outMaterials.push_back(def);
        }
    }
    uint32_t defaultMatId = localMaterialBase + (uint32_t)model.materials.size();
    std::vector<UvTransform> uvXfs;
    uvXfs.reserve(model.materials.size());
    for (const auto& gm : model.materials)
        uvXfs.push_back(uvTransformOf(gm));

    // Walk the default scene's node hierarchy, accumulating global transforms.
    const glm::mat3 nrmXf = glm::inverseTranspose(glm::mat3(transform));
    size_t trisBefore = tris.size();
    auto emitPrimitive = [&](const tinygltf::Primitive& prim, const glm::mat4& nodeGlobal) {
        if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
            std::cout << "core: glTF: skipping non-triangle primitive in " << path << "\n";
            return true;
        }
        auto posIt = prim.attributes.find("POSITION");
        if (posIt == prim.attributes.end())
            return true;
        const tinygltf::Accessor& posAcc = model.accessors[posIt->second];
        if (posAcc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT
            || posAcc.type != TINYGLTF_TYPE_VEC3) {
            std::cout << "core: glTF: unsupported POSITION format in " << path << "\n";
            return false;
        }

        uint32_t base = (uint32_t)positions.size();
        glm::mat4 xf = transform * nodeGlobal;
        glm::mat3 nXf = nrmXf * glm::inverseTranspose(glm::mat3(nodeGlobal));
        for (size_t i = 0; i < posAcc.count; i++) {
            const float* p = (const float*)accessorElem(model, posAcc, 12, i);
            glm::vec3 wp = glm::vec3(xf * glm::vec4(p[0], p[1], p[2], 1.0f));
            positions.push_back(hostStore3(wp));
        }

        auto nrmIt = prim.attributes.find("NORMAL");
        bool haveNormals = useVertexNormal && nrmIt != prim.attributes.end();
        if (haveNormals) {
            const tinygltf::Accessor& nAcc = model.accessors[nrmIt->second];
            haveNormals = nAcc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
                          && nAcc.type == TINYGLTF_TYPE_VEC3 && nAcc.count == posAcc.count;
            if (haveNormals)
                for (size_t i = 0; i < posAcc.count; i++) {
                    const float* n = (const float*)accessorElem(model, nAcc, 12, i);
                    normals.push_back(hostStore3(glm::normalize(nXf * glm::vec3(n[0], n[1], n[2]))));
                }
        }
        if (!haveNormals)
            for (size_t i = 0; i < posAcc.count; i++)
                normals.push_back(hostStore3(glm::vec3(0.0f)));  // geometric fallback

        auto uvIt = prim.attributes.find("TEXCOORD_0");
        bool haveUvs = uvIt != prim.attributes.end();
        if (haveUvs) {
            const tinygltf::Accessor& uvAcc = model.accessors[uvIt->second];
            haveUvs = uvAcc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
                      && uvAcc.type == TINYGLTF_TYPE_VEC2 && uvAcc.count == posAcc.count;
            UvTransform xf;  // own-material texture transform, if any
            if (sceneMaterialId < 0 && prim.material >= 0
                && prim.material < (int)uvXfs.size())
                xf = uvXfs[prim.material];
            float cr = std::cos(xf.rotation), sr = std::sin(xf.rotation);
            if (haveUvs)
                for (size_t i = 0; i < posAcc.count; i++) {
                    const float* t = (const float*)accessorElem(model, uvAcc, 8, i);
                    float u = t[0], v = t[1];
                    if (xf.active) {
                        // KHR_texture_transform: scale, then rotate (clockwise),
                        // then offset, in glTF's top-left uv space.
                        float su = u * xf.scale.x, sv = v * xf.scale.y;
                        u = cr * su + sr * sv + xf.offset.x;
                        v = -sr * su + cr * sv + xf.offset.y;
                    }
                    // Top-left glTF origin -> the pipeline's bottom-left
                    // (images are decoded bottom-row-first above).
                    uvs.push_back(gpu_float2(u, 1.0f - v));
                }
        }
        if (!haveUvs)
            for (size_t i = 0; i < posAcc.count; i++)
                uvs.push_back(gpu_float2(-1.0f, -1.0f));

        uint32_t matId = sceneMaterialId >= 0 ? (uint32_t)sceneMaterialId
                         : prim.material >= 0 ? localMaterialBase + (uint32_t)prim.material
                                              : defaultMatId;
        auto emitTri = [&](uint32_t i0, uint32_t i1, uint32_t i2) {
            tris.push_back(gpu_uint4{ base + i0, base + i1, base + i2, matId });
        };
        if (prim.indices < 0) {
            for (size_t i = 0; i + 2 < posAcc.count; i += 3)
                emitTri((uint32_t)i, (uint32_t)i + 1, (uint32_t)i + 2);
            return true;
        }
        const tinygltf::Accessor& idxAcc = model.accessors[prim.indices];
        size_t compBytes = idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE ? 1
                           : idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? 2
                           : idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT ? 4
                                                                                          : 0;
        if (compBytes == 0) {
            std::cout << "core: glTF: unsupported index type in " << path << "\n";
            return false;
        }
        auto idxAt = [&](size_t i) -> uint32_t {
            const unsigned char* p = accessorElem(model, idxAcc, compBytes, i);
            return compBytes == 1 ? *p
                   : compBytes == 2 ? *(const uint16_t*)p
                                    : *(const uint32_t*)p;
        };
        for (size_t i = 0; i + 2 < (size_t)idxAcc.count; i += 3)
            emitTri(idxAt(i), idxAt(i + 1), idxAt(i + 2));
        return true;
    };

    // Recursive scene traversal (parent transform composed in); nodes outside
    // the scene graph are ignored per the spec. Guards against cycles.
    std::unordered_set<int> visiting;
    bool failed = false;
    std::function<void(int, const glm::mat4&)> walk = [&](int nodeIdx, const glm::mat4& parent) {
        if (failed || nodeIdx < 0 || nodeIdx >= (int)model.nodes.size()
            || !visiting.insert(nodeIdx).second)
            return;
        const tinygltf::Node& node = model.nodes[nodeIdx];
        glm::mat4 global = parent * nodeLocalTransform(node);
        if (node.mesh >= 0 && node.mesh < (int)model.meshes.size())
            for (const auto& prim : model.meshes[node.mesh].primitives)
                if (!emitPrimitive(prim, global))
                    failed = true;
        for (int child : node.children)
            walk(child, global);
        visiting.erase(nodeIdx);
    };
    int sceneIdx = model.defaultScene >= 0 ? model.defaultScene : (model.scenes.empty() ? -1 : 0);
    if (sceneIdx < 0) {
        std::cout << "core: glTF " << path << " has no scene\n";
        return false;
    }
    for (int root : model.scenes[sceneIdx].nodes)
        walk(root, glm::mat4(1.0f));

    if (failed)
        return false;
    if (tris.size() == trisBefore) {
        std::cout << "core: glTF " << path << " contains no triangles\n";
        return false;
    }
    return true;
}
