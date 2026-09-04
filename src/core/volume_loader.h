#pragma once
// Host-side volume loading for the renderer core: reads a NanoVDB grid with
// NanoVDB's own reader and re-bricks it into the flat, pointer-free layout
// the device samples (core/medium_shared.h VolGrid): 8^3 bricks — NanoVDB's
// leaf size, so leaves copy straight across — addressed by a dense table over
// the grid's index-space bounding box, with a conservative per-brick
// majorant. Backend-neutral: no GPU types.
#include <cstdint>
#include <string>
#include <vector>
#include <glm/glm.hpp>

struct BrickGrid {
    glm::ivec3 origin { 0, 0, 0 };    // index-space coordinate of brick (0,0,0), multiple of 8
    glm::uvec3 brickDim { 0, 0, 0 };  // bricks per axis
    std::vector<uint32_t> table;      // brickDim.x*y*z entries: voxel-run index or VOL_BRICK_EMPTY
    std::vector<float> voxels;        // 512 floats per stored brick (x-major, z fastest)
    std::vector<float> majorants;     // per brick: max over the brick and its 26 neighbours
                                      // (trilinear samples reach one voxel across the boundary)
    float maxValue = 0.0f;
    glm::mat4 indexFromWorld { 1.0f };   // the grid's own map (grid world -> voxel index)
    size_t activeVoxels = 0;
};

// Loads grid `gridName` ("density", "temperature") from a .nvdb file. False
// with `err` set when the file cannot be read or has no such grid; a missing
// optional grid is reported the same way, the caller decides.
bool loadNanoVdbGrid(const std::string& path, const std::string& gridName, BrickGrid& out,
                     std::string& err);
