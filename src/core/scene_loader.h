#pragma once
// The renderer-core scene loader: parses Fluora's .txt scene format into a
// backend-neutral CoreScene (invariant I-4 — no backend headers). Covers the
// full key set scene.cpp reads, including the spectral material parameters
// (REFRIOR_NAMED / REFRIOR_RGB / REFRIOR_REAL_NAMED / REFRIOR_IMAG_NAMED),
// which are carried through for the spectral port even though FluoraMini does
// not consume them yet. Out of scope here: glTF (dead code in scene.cpp) and
// PLY + .json volume scenes, which migrate with the volume/spectral work.
#include <cstdint>
#include <string>
#include <vector>
#include <glm/glm.hpp>

#include "bvh_builder.h"

constexpr uint32_t kCoreTexNone = 0xFFFFFFFFu;

// Scene-format material types; the .txt names map as in scene.cpp
// (frenselSpecular -> Dielectric, microfacet/conductor -> Conductor).
// Unsupported types (asymMicrofacet, ...) degrade to Diffuse with a warning.
enum class CoreMaterialType { Diffuse, Emissive, Dielectric, Conductor };

struct CoreMaterial {
    CoreMaterialType type = CoreMaterialType::Diffuse;
    glm::vec3 rgb { 0.5f, 0.5f, 0.5f };
    float ior = 1.5f;          // REFRIOR
    float emittance = 0.0f;
    float roughness = 0.0f;
    uint32_t texIdx = kCoreTexNone;  // index into CoreScene::texturePaths

    // Spectral parameters, parsed but unconsumed until the spectral port:
    // named spectra resolve against SpectrumConsts/ tables there.
    std::string etaNamed;      // REFRIOR_NAMED / REFRIOR_REAL_NAMED
    std::string kNamed;        // REFRIOR_IMAG_NAMED
    glm::vec3 etaRgb { 0.0f, 0.0f, 0.0f };  // REFRIOR_RGB
    bool hasEtaRgb = false;
};

enum CoreGeomType { CORE_GEOM_CUBE = 0, CORE_GEOM_SPHERE = 1 };

// Analytic unit primitive (cube [-0.5,0.5]^3, sphere r=0.5) with baked
// transforms, in device layout. Built with utilityCore's exact composition
// (T * Rx * Ry * Rz * S, degrees) and glm::inverse/inverseTranspose.
struct CoreObject {
    int geomType = CORE_GEOM_CUBE;
    int materialId = -1;
    gpu_storage4x4 transform, invTransform, invTranspose;
};

struct CoreCamera {
    int width = 800;
    int height = 800;
    float fovyDeg = 45.0f;     // effectively a half-angle (scene.cpp quirk)
    int iterations = 1000;
    int maxDepth = 8;
    glm::vec3 eye { 0.0f, 5.0f, 10.5f };
    glm::vec3 lookAt { 0.0f, 5.0f, 0.0f };
    glm::vec3 up { 0.0f, 1.0f, 0.0f };
    std::string outputName = "mini";
};

struct CoreScene {
    std::vector<CoreMaterial> materials;
    std::vector<CoreObject> objects;   // analytic cubes/spheres
    CoreCamera camera;

    // Mesh geometry, world-space baked at load (no instancing). tris are
    // {i0, i1, i2, materialId}; positions/normals/uvs are unified vertex
    // arrays sharing those indices (see mesh_loader.h).
    std::vector<gpu_storage3> positions;
    std::vector<gpu_storage3> normals;
    std::vector<gpu_float2> uvs;
    std::vector<gpu_uint4> tris;
    BvhBuildResult bvh;

    // Base-color texture files (scene-dir joined), deduped — index ==
    // CoreMaterial::texIdx, so the host must create heap textures in exactly
    // this order for heap indices to line up.
    std::vector<std::string> texturePaths;

    // SKYBOX line: equirectangular .hdr path (scene-dir relative), or empty.
    std::string envMapPath;
};

// Returns false and sets err on failure (missing file, bad material
// reference, no renderable geometry, ...).
bool loadTxtScene(const std::string& path, CoreScene& out, std::string& err);
