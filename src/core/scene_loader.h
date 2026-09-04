#pragma once
// The renderer-core scene loader: parses Fluora's two scene formats into a
// backend-neutral CoreScene (invariant I-4 — no backend headers).
//   .txt  — the CIS-5650 line format (MATERIAL / CAMERA / OBJECT / SKYBOX
//           blocks), the full key set scene.cpp reads, including the spectral
//           material parameters.
//   .json — the newer format (Scene::loadJSON): named materials with
//           spectral eta/k, DOF camera parameters, env-map scale/clamp, PLY
//           and inline meshes, and media + medium interfaces.
// Everything a file says is carried, but not everything is rendered yet:
// media, medium interfaces and DOF are parsed for the M4 part-2 volume/DOF
// steps (docs/metal-rhi-design.md) and ignored by FluoraMini until then. Out
// of scope: glTF (dead code in scene.cpp).
#include <cstdint>
#include <string>
#include <vector>
#include <glm/glm.hpp>

#include "bvh_builder.h"

constexpr uint32_t kCoreTexNone = 0xFFFFFFFFu;

// Scene-format material types; the .txt names map as in scene.cpp
// (frenselSpecular -> Dielectric, microfacet/conductor -> Conductor).
// Unsupported types (asymMicrofacet, ...) degrade to Diffuse with a warning.
// Interface is a pass-through boundary: .json objects that only carry a
// MEDIUM_INTERFACE, no surface. Their geometry is not emitted until media
// land — a surfaceless boundary still blocks shadow rays, which no BSDF-side
// pass-through undoes — but the material stays, so the medium pair it names
// survives into the scene.
enum class CoreMaterialType { Diffuse, Emissive, Dielectric, Conductor, Interface };

struct CoreMaterial {
    CoreMaterialType type = CoreMaterialType::Diffuse;
    glm::vec3 rgb { 0.5f, 0.5f, 0.5f };
    float ior = 1.5f;          // REFRIOR / ETA const
    float emittance = 0.0f;
    float roughness = 0.0f;
    uint32_t texIdx = kCoreTexNone;  // index into CoreScene::texturePaths

    // Spectral parameters: named spectra resolve against the SpectrumConsts
    // tables (core/spectra.cpp) at upload.
    std::string etaNamed;      // REFRIOR_NAMED / REFRIOR_REAL_NAMED / ETA named
    std::string kNamed;        // REFRIOR_IMAG_NAMED / K named
    glm::vec3 etaRgb { 0.0f, 0.0f, 0.0f };  // REFRIOR_RGB
    bool hasEtaRgb = false;

    // .json extras, carried but not rendered yet.
    std::string normalMapPath; // NORMAL_MAP (scene-dir joined)
    int mediumIn = -1;         // MEDIUM_INTERFACE: indices into CoreScene::media,
    int mediumOut = -1;        // -1 = vacuum. Attached per material because
                               // triangles carry only a material id.
};

enum class CoreMediumType { Homogeneous, NanoVdb };

// One "Media" entry of a .json scene, as written (rgb coefficients become
// spectra at upload; the grid file is not opened here).
struct CoreMedium {
    CoreMediumType type = CoreMediumType::Homogeneous;
    std::string name;
    std::string vdbPath;       // NanoVdb: PATH (scene-dir joined)
    glm::vec3 sigmaA { 0.0f, 0.0f, 0.0f };
    glm::vec3 sigmaS { 0.0f, 0.0f, 0.0f };
    float sigmaScale = 1.0f;   // SIGMA_SCALE
    float g = 0.0f;            // Henyey-Greenstein asymmetry
    float leScale = 0.0f;      // LESCALE (emission scale; 0 = none)
    float temperatureScale = 1.0f;   // TEMPSCALE (NanoVdb blackbody)
    float temperatureOffset = 0.0f;  // TEMPOFFSET
    glm::mat4 worldFromMedium { 1.0f };
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
    // Thin-lens DOF (.json LENS_RADIUS / FOCAL_LEN); 0 radius = pinhole.
    float lensRadius = 0.0f;
    float focalLength = 0.0f;
    int mediumId = -1;         // .json Camera MEDIUM: the medium the eye sits in
};

// scene.cpp's default clamp on env-map radiance (environmentMapMaxLumin).
constexpr float kCoreEnvMaxRadiance = 1e5f;

struct CoreScene {
    std::vector<CoreMaterial> materials;
    std::vector<CoreObject> objects;   // analytic cubes/spheres
    std::vector<CoreMedium> media;     // .json only; empty otherwise
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

    // SKYBOX line / Background PATH: equirectangular image (scene-dir
    // relative), or empty. Radiance is min(rgb * envScale, envMaxRadiance),
    // as scene.cpp's ImageInfiniteLight applies it.
    std::string envMapPath;
    float envScale = 1.0f;
    glm::vec3 envMaxRadiance { kCoreEnvMaxRadiance, kCoreEnvMaxRadiance, kCoreEnvMaxRadiance };
};

// Dispatches on the extension (.txt / .json). Returns false and sets err on
// failure (missing file, bad material reference, no renderable geometry, ...).
bool loadScene(const std::string& path, CoreScene& out, std::string& err);
bool loadTxtScene(const std::string& path, CoreScene& out, std::string& err);
bool loadJsonScene(const std::string& path, CoreScene& out, std::string& err);
