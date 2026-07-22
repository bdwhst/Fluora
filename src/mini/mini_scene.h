#pragma once
// Loader for the subset of Fluora's .txt scene format that FluoraMini (M1)
// supports: MATERIAL / CAMERA / OBJECT blocks with cube and sphere geometry.
// Mesh-based objects and unknown material types degrade gracefully (skip /
// diffuse fallback) with a warning, so most existing scenes at least load.
#include <string>
#include <vector>
#include <simd/simd.h>
#include "mini_shared.h"
#include "../core/bvh_builder.h"

struct MiniCamera {
    int width = 800;
    int height = 800;
    float fovyDeg = 45.0f;
    int iterations = 1000;
    int maxDepth = 8;
    simd_float3 eye = { 0.0f, 5.0f, 10.5f };
    simd_float3 lookAt = { 0.0f, 5.0f, 0.0f };
    simd_float3 up = { 0.0f, 1.0f, 0.0f };
    std::string outputName = "mini";
};

struct MiniScene {
    std::vector<MiniMaterial> materials;
    std::vector<MiniObject> objects;      // analytic cubes/spheres
    MiniCamera camera;

    // Mesh geometry, world-space baked at load (no instancing); loading and
    // BVH construction live in src/core. tris are {i0, i1, i2, materialId}.
    std::vector<simd_float3> positions;
    std::vector<simd_uint4> tris;
    BvhBuildResult bvh;

    // SKYBOX line: equirectangular .hdr path (scene-dir relative), or empty.
    std::string envMapPath;
};

// Returns false and sets err on failure (missing file, no objects, ...).
bool miniLoadScene(const std::string& path, MiniScene& out, std::string& err);
